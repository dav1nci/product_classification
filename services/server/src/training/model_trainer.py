from internal.comm_protocol_types import TrainingConfig
from minio_utils.download import download_data_to_file
from minio_utils.connection import minio_client
from training.data_loader import TextClassificationDataset
from training.utils import GetActiveRunCallback, get_best_step, parse_s3_objectname
import pandas
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, f1_score
from mlflow.tracking import MlflowClient
from dependencies import logger
import torch
import gc



class AutomaticModelTrainer(object):
    def __init__(self, training_config: TrainingConfig, db_record_id):
        self.val_dataset = None
        self.best_step_info = None
        self.train_dataset = None
        self.training_config = training_config

        # TODO make it configurable!
        self.labelmap = {
            'Dry Goods & Pantry Staples': 0,
            'Fresh & Perishable Items': 1,
            'Household & Personal Care': 2,
            'Beverages': 3,
            'Specialty & Miscellaneous': 4
        }
        self.model_name_base = 'bert-base-uncased'
        self.db_record_id = db_record_id

        self.data_train=None
        self.tokenizer = None
        self.run_id = None


    def init_trainer(self):
        logger.info(f"[init_trainer]:{self.db_record_id} started")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_base)

    def setup_mlflow_logging(self):
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "test1"
        os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "True"

        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

    def train_model(self):
        logger.info(f"[train_model]:{self.db_record_id} started")
        self.setup_mlflow_logging()

        model = BertForSequenceClassification.from_pretrained(self.model_name_base, num_labels=5)
        run_name = f"{self.model_name_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        training_args = TrainingArguments(
            output_dir='/tmp/training_results/',
            num_train_epochs=self.training_config.epoch_num,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=200,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            run_name=run_name,
        )
        active_run_id_callback = GetActiveRunCallback()

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            f1_for_each_class = f1_score(labels, preds, average=None)
            return {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'f1_for_each_class': f1_for_each_class.tolist(),
                'f1_min': min(f1_for_each_class),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[active_run_id_callback]
        )

        trainer.train()

        self.run_id = active_run_id_callback.run_id
        # Free resources
        del model
        del trainer
        torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Collect unused Python objects



    def finalize_training(self):
        logger.info(f"[finalize_training]:{self.db_record_id} started")
        client = MlflowClient()
        self.best_step_info = get_best_step(client, self.run_id, self.training_config.best_model_metric)


    def retrieve_data(self):
        logger.info(f"[retrieve_data]:{self.db_record_id} started")
        bucket_name, s3_file_object_path = parse_s3_objectname(self.training_config.train_file_s3)
        out_fname = download_data_to_file(minio_client, bucket_name, s3_file_object_path)
        self.data_train = pandas.read_csv(out_fname)


    def validate_data(self):
        logger.info(f"[validate_data]:{self.db_record_id} started")
        required_columns = ['product_description', 'Category']
        expected_dtypes = {'product_description': 'object', 'Category': 'object'}
        required_target_values = set(self.labelmap.keys())
        target_column = 'Category'

        # Check if required columns are in dataset
        missing_columns = [col for col in required_columns if col not in self.data_train.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        logger.info("All required columns are present.")

        # Check if datatypes match the expected types
        for column, expected_dtype in expected_dtypes.items():
            if column in self.data_train.columns and not pandas.api.types.is_dtype_equal(self.data_train[column].dtype, expected_dtype):
                raise ValueError(
                    f"Column '{column}' has wrong dtype. Expected: {expected_dtype}, Got: {self.data_train[column].dtype}")
        logger.info("All columns have the correct datatypes.")

        # Check for NaN values in the required columns
        for column in required_columns:
            if self.data_train[column].isnull().any():
                raise ValueError(f"Column '{column}' contains NaN values.")
        logger.info("No NaN values in the required columns.")

        # Check if unique values in the target column intersect with the required target values
        if target_column in self.data_train.columns:
            unique_values = set(self.data_train[target_column].unique())
            if not unique_values.issubset(required_target_values):
                raise ValueError(f"Unique values in column '{target_column}' do not match the required target values.")
        logger.info(f"Unique values in column '{target_column}' are valid.")


    def preprocess_data(self):
        logger.info(f"[preprocess_data]:{self.db_record_id} started")
        self.data_train['category_index'] = self.data_train.apply(lambda x: self.labelmap[x['Category']], axis=1)

        # # TODO DEBUG! comment this out!
        # self.data_train = self.data_train.sample(500)

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.data_train['product_description'], self.data_train['category_index'], test_size=0.2, random_state=42
        )

        self.train_dataset = TextClassificationDataset(train_texts.tolist(), train_labels.tolist(), self.tokenizer)
        self.val_dataset = TextClassificationDataset(val_texts.tolist(), val_labels.tolist(), self.tokenizer)



    def run(self):
        self.init_trainer()
        self.retrieve_data()
        self.validate_data()
        self.preprocess_data()
        self.train_model()
        self.finalize_training()
