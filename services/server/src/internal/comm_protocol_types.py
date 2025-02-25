from pydantic import BaseModel

class InferenceConfig(BaseModel):
    input_file_s3: str


class BaseTrainingConfig(BaseModel):
    epoch_num: int
    best_model_metric: str

class TrainingConfig(BaseTrainingConfig):
    train_file_s3: str

class HFTrainingConfig(BaseTrainingConfig):
    train_dir_s3: str


# class InputForTrainingHF(BaseModel):
#     hf_file_s3: str