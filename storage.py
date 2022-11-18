from datetime import timedelta
from os import environ
from google import auth
from google.auth import compute_engine
from google.auth.transport import requests
from google.cloud import storage

credentials, project = auth.default()
signing_credentials = compute_engine.IDTokenCredentials(
    requests.Request(),
    "",
    service_account_email=credentials.service_account_email
)

storage_client = storage.Client(credentials=signing_credentials)

def _get_bucket() -> storage.Bucket:
    bucket_name = environ.get("BUCKET_NAME")

    return storage_client.bucket(bucket_name)

def _get_object(obj_name: str) -> storage.Blob:
    bucket = _get_bucket()
    return bucket.blob(obj_name)

def get_object_signed_url(obj_name: str) -> str:
    bucket_obj = _get_object(obj_name)
    bucket_obj_expiry = timedelta(minutes=5)

    return bucket_obj.generate_signed_url(
        expiration=bucket_obj_expiry,
        credentials=signing_credentials,
    )
