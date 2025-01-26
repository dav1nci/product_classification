from fastapi import APIRouter, BackgroundTasks

from dependencies import get_db_session
from internal.comm_protocol_types import TrainingConfig
from sql_utils.crud import add_training_job
from training.train_workers import regular_train_worker
from training.model_trainer import AutomaticModelTrainer
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException, Depends
from datetime import datetime
import pytz
from uuid import uuid4

router = APIRouter(
    prefix="/v1",
    tags=["training"],
    dependencies=[],
    responses={404: {"description": "Not found"}}
)


@router.post("/train/csv")
async def train_ds(training_config: TrainingConfig,
                   bg_tasks: BackgroundTasks,
                   db: Session = Depends(get_db_session)):
    """
    This endpoint is called when there's a new official dataset to train model on
    :param bg_tasks:
    :return:
    """

    train_job_id = uuid4()

    add_training_job(db, train_job_id, datetime.now(pytz.utc),
                     "06d184b35e4a3ba4a6c9f7143157997fa3e7e13f", # 06d184b35e4a3ba4a6c9f7143157997fa3e7e13f - is a mock, TODO add dataset versioning suppport
                     training_config.train_file_s3,
                     "PENDING")
    automatic_model_train = AutomaticModelTrainer(training_config, db_record_id=train_job_id)
    bg_tasks.add_task(regular_train_worker, automatic_model_train)

    return {
        "db_record_id": train_job_id,
        "status": "PENDING"
    }


@router.post("/train/human_feedback")
async def train_hf(bg_task: BackgroundTasks):
    pass
