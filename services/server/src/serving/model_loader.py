import torch
from sql_utils.crud import fetch_recent_model
import os
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from transformers import BertTokenizer, BertForSequenceClassification


class ModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.mlflow_run_id = None
        self.best_checkpoint = None
        self.saved_checkpoint_path = None


        os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

    def new_model_available(self):
        from dependencies import get_db_session
        db_session_generator = get_db_session()
        db_session = next(get_db_session())

        result = fetch_recent_model(db_session)
        db_session_generator.close()
        run_id, best_checkpoint = None, None
        if result:
            run_id, best_checkpoint = result[0]
        else:
            raise NotImplementedError("no model available for serving")

        if run_id == self.mlflow_run_id and best_checkpoint == self.best_checkpoint:
            return False
        else:
            self.mlflow_run_id = run_id
            self.best_checkpoint = best_checkpoint
            return True



    def download_model_from_mlflow(self):
        artifacts_download_dir = os.path.join(f'/tmp/{self.mlflow_run_id}')
        client = MlflowClient()

        if not os.path.exists(artifacts_download_dir):
            os.makedirs(artifacts_download_dir)

        try:
            client.download_artifacts(self.mlflow_run_id, self.best_checkpoint, artifacts_download_dir)
        except MlflowException as e:
            print(f"Exception occured: {e}")

            # s3://mlflow/1/9acee285129b429ba9243103cac82ff4/artifacts/checkpoint-6750

        self.saved_checkpoint_path = os.path.join(artifacts_download_dir, self.best_checkpoint, "artifacts", self.best_checkpoint)

    def init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.saved_checkpoint_path)
        self.model = BertForSequenceClassification.from_pretrained(self.saved_checkpoint_path).to(self.device)
        self.model.eval()


    def get_model(self):
        if self.new_model_available():
            self.download_model_from_mlflow()
        self.init_model()
        return self.tokenizer, self.model, f"{self.mlflow_run_id}:{self.best_checkpoint}"

