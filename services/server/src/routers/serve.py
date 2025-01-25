import pytz
from fastapi import APIRouter, BackgroundTasks
from internal.comm_protocol_types import InputForInference
from sql_utils.crud import add_inference_job
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException, Depends
from uuid import uuid4
from datetime import datetime
from dependencies import get_db_session

router = APIRouter(
    prefix="/v1",
    tags=["serving"],
    dependencies=[],
    responses={404: {"description": "Not found"}}
)

@router.post("/predict/csv")
async def predict_csv(input_for_inference: InputForInference,
                      bg_tasks: BackgroundTasks,
                      db: Session = Depends(get_db_session)):

    job_id = uuid4()
    add_inference_job(db, str(job_id), datetime.now(pytz.utc), "STATUS_PENDING", input_for_inference.input_file_s3)
    # args = process_job_description(query)
    # print(f"Args: {args}")
    # bg_tasks.add_task(worker, input_for_inference)

    return {
        "job_id": job_id,
        "status": "STATUS_PENDING"
    }

