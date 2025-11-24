from minio import Minio
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()
print("MINIO_URL:", os.getenv("MINIO_URL"))
# 환경변수 사용
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_URL = os.getenv("MINIO_URL")

# MinIO 클라이언트 생성
client = Minio(
    endpoint=MINIO_URL,  # MinIO 서버 주소와 포트
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=True  # http면 False, https면 True
)

bucket_name = "levelupfit-videos"

# 버킷이 없으면 생성
if not client.bucket_exists(bucket_name=bucket_name):
    client.make_bucket(bucket_name=bucket_name)