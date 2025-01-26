import io
import json
from minio.error import S3Error


def bucket_exists_or_create(client, bucket_name):
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)

def object_exists(client, bucket_name, object_name):
    try:
        # Use stat_object to check if the object exists
        client.stat_object(bucket_name, object_name)
        return True

    except S3Error as err:
        # print(f"Error occurred: {err}")
        return False
    except Exception as e:
        print(f"Non defined exception {e}")
        return False

def upload_data_to_bucket(client, bucket_name, dest_file, data):
    source_file_bytes = data.to_csv(index=False).encode('utf-8')
    source_file_stream = io.BytesIO(source_file_bytes)

    bucket_exists_or_create(client, bucket_name)
    if not object_exists(client, bucket_name, dest_file):
        client.put_object(
            bucket_name, dest_file, source_file_stream, length=len(source_file_bytes)
        )
    else:
        raise ValueError(f"Object {bucket_name}:{dest_file} already exists in s3")

def upload_file_to_bucket(client, bucket_name, dest_file, fname_local):
    # upload_report_to_bucket(minio_client,
    #                         'ai-bucket',
    #                         f"job/{job_description.job_id}/report.json", report)
    bucket_exists_or_create(client, bucket_name)
    if not object_exists(client, bucket_name, dest_file):
        print(f"Creating new object {bucket_name}:{dest_file}")
        client.fput_object(bucket_name, dest_file, fname_local)
    else:
        raise ValueError(f"Object {bucket_name}:{dest_file} already exists in s3")




if __name__ == '__main__':
    from minio import Minio
    minio_client = Minio("minio:9000",
                         access_key="Cu2rKKmg0dvx42Ril6k0",
                         secret_key="cEM5D3EI2cgY4OQNMOwqCOiepPFdIXbuDxa4NH5X",
                         secure=False
                         )
    # source_file = "/tmp/test-file.txt"
    source_file = "{'custom_key': 'custom_value}"
    source_file_bytes = source_file.encode('utf-8')
    source_file_stream = io.BytesIO(source_file_bytes)
    # The destination bucket and filename on the MinIO server
    bucket_name = "ai-bucket"
    destination_file = "job/2/report.json"

    # Make the bucket if it doesn't exist.
    found = minio_client.bucket_exists(bucket_name)
    if not found:
        minio_client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # Upload the file, renaming it in the process
    minio_client.put_object(
        bucket_name, destination_file, source_file_stream, length=len(source_file_bytes)
    )
    print(
        source_file, "successfully uploaded as object",
        destination_file, "to bucket", bucket_name,
    )