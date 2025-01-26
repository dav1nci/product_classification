from serving.inference_pipeline import InferencePipeline
import torch


def regular_inference_worker(inference_pipeline: InferencePipeline):
    try:
        inference_pipeline.run()
    except torch.OutOfMemoryError as e:
        print(f"OOM Error. Details:\n {e}")
    except Exception as e:
        print(f"Exception occured: {e}")