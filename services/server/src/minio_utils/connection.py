from minio import Minio
import os
from minio_utils.upload import upload_file_to_bucket

minio_client = Minio(f"{os.environ.get('MINIO_HOST')}:{os.environ.get('MINIO_PORT')}",
                     access_key=os.environ.get('AWS_ACCESS_KEY_ID'),
                     secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                     secure=False
                     )

def init_minio_filestructure():
    try:
        upload_file_to_bucket(minio_client, "dataset",
                              "train/Training_Data.csv",
                              "/data/Training_Data.csv")
    except ValueError as e:
        print("[init_minio_filestructure] object already exists")

    try:
        upload_file_to_bucket(minio_client, "dataset",
                              "human_feedback/Query_and_Validation_Data.csv",
                              "/data/Query_and_Validation_Data.csv")
    except ValueError as e:
        print("[init_minio_filestructure] object already exists")