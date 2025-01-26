from pydantic import BaseModel
from typing import List, Union

class InferenceConfig(BaseModel):
    input_file_s3: str

class TrainingConfig(BaseModel):
    train_file_s3: str
    epoch_num: int
    best_model_metric: str

class InputForTrainingHF(BaseModel):
    hf_file_s3: str