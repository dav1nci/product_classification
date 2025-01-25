from minio.error import S3Error

def download_data_from_bucket(client, bucket_name, obj_name):
    try:
        response = client.get_object(bucket_name, obj_name)
        return response.json()
        # Read data from response.
    finally:
        response.close()
        response.release_conn()

    # return response

def download_data_to_file(client, bucket_name, obj_path):
    try:
        out_fname = f"/tmp/{obj_path.split('/')[-1]}"
        client.fget_object(bucket_name, obj_path, out_fname)
        print(f"File '{bucket_name}:{obj_path}' downloaded successfully to '{out_fname}'.")
        return out_fname
    except S3Error as err:
        print(f"Error downloading file: {err}")
