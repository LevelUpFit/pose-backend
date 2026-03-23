from minio import Minio

from app.core.config import settings

MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
MINIO_URL = settings.MINIO_URL

# MinIO 클라이언트 생성
client = Minio(
    endpoint=MINIO_URL,  # MinIO 서버 주소와 포트
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=True  # http면 False, https면 True
)

bucket_name = settings.MINIO_BUCKET_NAME

# 버킷이 없으면 생성
if not client.bucket_exists(bucket_name=bucket_name):
    client.make_bucket(bucket_name=bucket_name)

# 버킷을 public read로 설정 (브라우저에서 직접 재생 가능)
policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
        }
    ]
}

try:
    import json
    client.set_bucket_policy(bucket_name=bucket_name, policy=json.dumps(policy))
    print(f"Bucket '{bucket_name}' is now public for read access")
except Exception as e:
    print(f"Warning: Could not set bucket policy: {e}")