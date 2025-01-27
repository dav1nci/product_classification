from serving.inference_pipeline import InferencePipeline
import torch
from dependencies import logger


def regular_inference_worker(inference_pipeline: InferencePipeline):
    try:
        logger.info(f"Starting inference pipeline with id {inference_pipeline.db_job_id}")
        inference_pipeline.run()
    except torch.OutOfMemoryError as e:
        logger.error(f"OOM Error. Details:\n {e}")
    except Exception as e:
        logger.error(f"Exception occured: {e}")