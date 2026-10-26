from urllib.parse import urlparse
import os
import shutil

# Returns a local path for the checkpoint, downloading it first when it lives in S3
def resolve_checkpoint(location, cache_dir="/tmp/echo-models"):
  if not location.startswith("s3://"):
    return location
  import boto3
  
  url = urlparse(location)
  bucket, key = url.netloc, url.path.lstrip("/")
  local_path = os.path.join(cache_dir, key)
  if not os.path.exists(local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    boto3.client("s3").download_file(bucket, key, local_path)
  return local_path

# Writes an uploaded file to disk, keeps its extension so OpenCV picks the right decoder
def save_upload(upload, folder, name):
  extension = os.path.splitext(upload.filename or "")[1]
  path = os.path.join(folder, name + extension)
  with open(path, "wb") as f:
    shutil.copyfileobj(upload.file, f)
  return path

