# from fastapi import FastAPI, Depends
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
# app = FastAPI()
#
# class ModelLoader:
#     def __init__(self):
#         self.model = None
#         self.tokenizer = None
#
#     def get_model(self):
#         if self.model is None or self.tokenizer is None:
#             print("Loading model for the first time...")
#             self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
#             self.model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda" if torch.cuda.is_available() else "cpu")
#         return self.model, self.tokenizer
#
# # Global singleton instance
# model_loader = ModelLoader()
#
# @app.post("/predict/")
# def predict(input_text: str, model_and_tokenizer=Depends(model_loader.get_model)):
#     model, tokenizer = model_and_tokenizer
#     inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_length=50)
#     return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}
#
#
import time
from datetime import datetime
import pandas
import pytz
from sklearn.metrics import f1_score

from dependencies import get_db_session
from internal.comm_protocol_types import InferenceConfig
from minio_utils.download import download_data_to_file
from minio_utils.upload import upload_data_to_bucket
from serving.utils import process_in_batches, get_class_titles
from sql_utils.crud import finalize_inference_job_in_db, report_model_metrics_to_db
from training.utils import parse_s3_objectname
from minio_utils.connection import minio_client



class InferencePipeline(object):
    def __init__(self, tokenizer, model, model_name, db_job_id, inference_config: InferenceConfig):
        self.data_inference = None
        self.db_job_id = db_job_id
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        self.inference_config = inference_config
        # TODO make it configurable!
        self.labelmap = {
            'Dry Goods & Pantry Staples': 0,
            'Fresh & Perishable Items': 1,
            'Household & Personal Care': 2,
            'Beverages': 3,
            'Specialty & Miscellaneous': 4
        }

    def fetch_inference_data(self):
        bucket_name, s3_file_object_path = parse_s3_objectname(self.inference_config.input_file_s3)
        out_fname = download_data_to_file(minio_client, bucket_name, s3_file_object_path)
        self.data_inference = pandas.read_csv(out_fname)

    def validate_inference_data(self):
        required_columns = ['product_description']
        expected_dtypes = {'product_description': 'object'}

        # Check if required columns are in dataset
        missing_columns = [col for col in required_columns if col not in self.data_inference.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        print("All required columns are present.")

        # Check if datatypes match the expected types
        for column, expected_dtype in expected_dtypes.items():
            if column in self.data_inference.columns and not pandas.api.types.is_dtype_equal(self.data_inference[column].dtype,
                                                                                         expected_dtype):
                raise ValueError(
                    f"Column '{column}' has wrong dtype. Expected: {expected_dtype}, Got: {self.data_inference[column].dtype}")
        print("All columns have the correct datatypes.")

        # Check for NaN values in the required columns
        for column in required_columns:
            if self.data_inference[column].isnull().any():
                raise ValueError(f"Column '{column}' contains NaN values.")
        print("No NaN values in the required columns.")



    def run_predictions(self):
        time_start = time.time()
        predictions = process_in_batches(self.tokenizer, self.model,
                                         self.data_inference['product_description'].values.tolist(), 64)
        self.inference_time = time.time() - time_start
        class_titles = get_class_titles(self.labelmap, predictions)
        self.data_inference['predictions'] = predictions
        self.data_inference['class_titles'] = class_titles

    def store_predictions(self):
        # prediction_uuid = uuid4()
        upload_data_to_bucket(minio_client, 'inference', f"{str(self.db_job_id)}/predictions.csv",
                              self.data_inference[['product_description', 'class_titles']])

        db_session_generator = get_db_session()
        db_session = next(get_db_session())

        finalize_inference_job_in_db(db_session, self.db_job_id, datetime.now(pytz.utc),
                                     f"s3://inference/{str(self.db_job_id)}/predictions.csv")
        db_session_generator.close()



    def finalize(self):
        pass


    def human_feedback_present(self):
        if 'HUMAN_VERIFIED_Category' in self.data_inference.columns:
            return True
        return False

    def handle_human_feedback(self):
        self.human_feedback_handler = HumanFeedbackHandler(self)
        self.human_feedback_handler.run()


    def run(self):
        self.fetch_inference_data()
        self.validate_inference_data()
        self.run_predictions()
        self.store_predictions()
        if self.human_feedback_present():
            self.handle_human_feedback()
        self.finalize()
#


class HumanFeedbackHandler(object):
    def __init__(self, parent_inference_pipeline: InferencePipeline):
        self.pip = parent_inference_pipeline

    def validate_data(self):
        required_columns = ['product_description', 'HUMAN_VERIFIED_Category']
        expected_dtypes = {'product_description': 'object', 'HUMAN_VERIFIED_Category': 'object'}
        required_target_values = set(self.pip.labelmap.keys())
        target_column = 'HUMAN_VERIFIED_Category'

        # Check if required columns are in dataset
        missing_columns = [col for col in required_columns if col not in self.filtered_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        print("All required columns are present.")

        # Check if datatypes match the expected types
        for column, expected_dtype in expected_dtypes.items():
            if column in self.filtered_data.columns and not pandas.api.types.is_dtype_equal(
                    self.filtered_data[column].dtype,
                    expected_dtype):
                raise ValueError(
                    f"Column '{column}' has wrong dtype. Expected: {expected_dtype}, Got: {self.filtered_data[column].dtype}")
        print("All columns have the correct datatypes.")

        # Check for NaN values in the required columns
        # for column in required_columns:
        #     if self.pip.data_inference[column].isnull().any():
        #         raise ValueError(f"Column '{column}' contains NaN values.")
        # print("No NaN values in the required columns.")

        # Check if unique values in the target column intersect with the required target values
        if target_column in self.filtered_data.columns:
            unique_values = set(self.filtered_data[target_column].unique())
            if not unique_values.issubset(required_target_values):
                raise ValueError(f"Unique values in column '{target_column}' do not match the required target values. Target values: {required_target_values}, unique_values: {unique_values}")
        print(f"Unique values in column '{target_column}' are valid.")

    def filter_data(self):
        self.filtered_data = self.pip.data_inference[~pandas.isnull(self.pip.data_inference['HUMAN_VERIFIED_Category'])]
        self.filtered_data = self.filtered_data[~pandas.isnull(self.filtered_data['product_description'])]
        assert self.filtered_data.shape[0] > 0, "After all filtering dataset size is 0"

    def add_to_training_datasets(self):
        upload_data_to_bucket(minio_client, 'dataset', f"train/human_verified_subset_from_job_{self.pip.db_job_id}.csv",
                              self.filtered_data[['product_description', 'HUMAN_VERIFIED_Category']])

        # db_session_generator = get_db_session()
        # db_session = next(get_db_session())
        #
        # finalize_inference_job_in_db(db_session, self.db_job_id, datetime.now(pytz.utc),
        #                              f"s3://inference/{str(self.db_job_id)}/predictions.csv")
        # db_session_generator.close()

    def calculate_model_metrics(self):
        """
        CREATE TABLE model_metrics (
        id INT AUTO_INCREMENT PRIMARY KEY,
        associated_job_id
        timestamp DATETIME NOT NULL,
        model_name VARCHAR(100),
        f1 FLOAT ,
        min_f1 FLOAT ,
        total_predictions INT,
        latency_avg_ms FLOAT
    );
        :return:
        """
        self.filtered_data['human_category_index'] = self.filtered_data.apply(lambda x: self.pip.labelmap[x['HUMAN_VERIFIED_Category']], axis=1)

        y_true = self.filtered_data['human_category_index']
        y_pred = self.filtered_data['predictions']

        self.f1_per_class = f1_score(y_true, y_pred, average=None)
        self.f1_weighted = f1_score(y_true, y_pred, average="weighted")


    def report_model_health_metrics_to_db(self):
        db_session_generator = get_db_session()
        db_session = next(get_db_session())
        #
        # finalize_inference_job_in_db(db_session, self.db_job_id, datetime.now(pytz.utc),
        #                              f"s3://inference/{str(self.db_job_id)}/predictions.csv")
        report_model_metrics_to_db(db_session, self.pip.db_job_id, datetime.now(pytz.utc), self.pip.model_name,
                                   self.f1_weighted, min(self.f1_per_class), self.pip.data_inference.shape[0],
                                   self.pip.inference_time * 1000 / self.pip.data_inference.shape[0])
        db_session_generator.close()

    def run(self):
        self.filter_data()
        self.validate_data()
        self.add_to_training_datasets()
        self.calculate_model_metrics()
        self.report_model_health_metrics_to_db()
