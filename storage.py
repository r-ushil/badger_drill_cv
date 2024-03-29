from google import auth
from google.cloud import storage
from datetime import timedelta
from os import environ

credentials, project = auth.default()

storage_client = storage.Client()

def _get_bucket() -> storage.Bucket:
	bucket_name = environ.get("BUCKET_NAME")

	return storage_client.bucket(bucket_name)

def _get_object(obj_name: str) -> storage.Blob:
	bucket = _get_bucket()
	return bucket.blob(obj_name)

def get_object_signed_url(obj_name: str) -> str:
	bucket_obj = _get_object(obj_name)
	bucket_obj_expiry = timedelta(minutes=5)

	from google.auth.transport import requests
	request = requests.Request()
	credentials.refresh(request)

	if not hasattr(credentials, "service_account_email"):
		print("Missing service_account_email")

	return bucket_obj.generate_signed_url(
		expiration=bucket_obj_expiry,
		service_account_email=credentials.service_account_email,
		access_token=credentials.token,
	)
