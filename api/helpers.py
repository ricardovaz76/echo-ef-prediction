from urllib.parse import urlparse
import os

class UploadTooLarge(Exception):
  pass

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
def save_upload(upload, folder, name, max_bytes, chunk_size=1024*1024):
  extension = os.path.splitext(upload.filename or "")[1]
  path = os.path.join(folder, name + extension)

  written = 0
  with open(path, "wb") as f:
    while True:
      chunk = upload.file.read(chunk_size)  # reads 1 MB at a time
      if not chunk:   # empty chunk means we've reached the end of the file
        break
      
      written += len(chunk)
      if written > max_bytes:
        raise UploadTooLarge(f"{upload.filename} is larger than {max_bytes // (1024*1024)} MB")
      f.write(chunk)
  
  return path

