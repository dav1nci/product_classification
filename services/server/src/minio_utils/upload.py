import io
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
    bucket_exists_or_create(client, bucket_name)
    if not object_exists(client, bucket_name, dest_file):
        print(f"Creating new object {bucket_name}:{dest_file}")
        client.fput_object(bucket_name, dest_file, fname_local)
    else:
        raise ValueError(f"Object {bucket_name}:{dest_file} already exists in s3")

