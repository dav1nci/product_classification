from minio.error import S3Error
from dependencies import logger
import os
import shutil

def download_data_from_bucket(client, bucket_name, obj_name):
    try:
        response = client.get_object(bucket_name, obj_name)
        return response.json()
        # Read data from response.
    finally:
        response.close()
        response.release_conn()



def download_data_to_file(client, bucket_name, obj_path):
    try:
        out_fname = f"/tmp/{obj_path.split('/')[-1]}"
        client.fget_object(bucket_name, obj_path, out_fname)
        logger.info(f"File '{bucket_name}:{obj_path}' downloaded successfully to '{out_fname}'.")
        return out_fname
    except S3Error as err:
        logger.error(f"Error downloading file: {err}")

def download_files_to_dir(client, bucket_name, obj_path):
    objects = client.list_objects(bucket_name, obj_path, recursive=False)
    #filter for CSV files
    csv_files = [obj.object_name for obj in objects if obj.object_name.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: s3://{bucket_name}/{obj_path}")

    #local directory to save CSVs
    local_dir = f'/tmp/{bucket_name}/{obj_path}'
    if os.path.exists(local_dir):
        logger.info(f"Found existing {local_dir}. Removing... ")
        shutil.rmtree(local_dir)

    os.makedirs(local_dir, exist_ok=True)

    #download CSV files
    local_files = []
    for csv_file in csv_files:
        local_path = os.path.join(local_dir, os.path.basename(csv_file))
        client.fget_object(bucket_name, csv_file, local_path)
        logger.info(f"File {csv_file} downloaded to local storage at {local_path}")
        local_files.append(local_path)

    return local_files


