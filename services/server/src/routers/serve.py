import pytz
from fastapi import APIRouter, BackgroundTasks
from internal.comm_protocol_types import InferenceConfig
from serving.inference_pipeline import InferencePipeline
from serving.inference_workers import regular_inference_worker
from sql_utils.crud import add_inference_job
from sqlalchemy.orm import Session
from fastapi import Depends
from uuid import uuid4
from datetime import datetime
from dependencies import get_db_session, model_loader

router = APIRouter(
    prefix="/v1",
    tags=["serving"],
    dependencies=[],
    responses={404: {"description": "Not found"}}
)

@router.post("/predict/csv")
async def predict_csv(inference_config: InferenceConfig,
                      bg_tasks: BackgroundTasks,
                      db: Session = Depends(get_db_session),
                      tokenizer_and_model=Depends(model_loader.get_model)):

    job_id = uuid4()
    add_inference_job(db, str(job_id), datetime.now(pytz.utc), "STATUS_PENDING", inference_config.input_file_s3)

    inference_pipeline = InferencePipeline(tokenizer_and_model[0], tokenizer_and_model[1], tokenizer_and_model[2],
                                           job_id, inference_config)
    bg_tasks.add_task(regular_inference_worker, inference_pipeline)

    return {
        "job_id": job_id,
        "status": "STATUS_PENDING"
    }



