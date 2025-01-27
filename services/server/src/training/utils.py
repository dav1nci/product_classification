from transformers import TrainerCallback
import mlflow
import numpy as np

class GetActiveRunCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if not mlflow.active_run():
            mlflow.start_run()
        self.run_id = mlflow.active_run().info.run_id

    def on_train_end(self, args, state, control, **kwargs):
        mlflow.end_run()


def get_best_step(client, run_id, metric):
    metric_history = client.get_metric_history(run_id, metric)
    metric_list = [i.value for i in metric_history]
    max_metric_index = np.argmax(metric_list)

    f1_history = client.get_metric_history(run_id, 'eval_f1')
    f1_min_history = client.get_metric_history(run_id, 'eval_f1_min')

    return {
        "best_checkpoint": f"checkpoint-{metric_history[max_metric_index].step}",
        "best_f1": f1_history[max_metric_index].value,
        "best_f1_min": f1_min_history[max_metric_index].value
    }


def parse_s3_objectname(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI. Must start with 's3://'.")

    #remove the "s3://" prefix
    s3_path = s3_uri[5:]

    #split into bucket name and object path
    parts = s3_path.split("/", 1)
    bucket_name = parts[0]
    object_path = parts[1] if len(parts) > 1 else ""

    return bucket_name, object_path