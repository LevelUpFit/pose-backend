from minio import Minio

# MinIO 클라이언트 생성
client = Minio(
    "minio.local:9000",  # MinIO 서버 주소와 포트
    access_key="YOUR_ACCESS_KEY",
    secret_key="YOUR_SECRET_KEY",
    secure=False  # http면 False, https면 True
)

bucket_name = "processed"
object_name = "your_video.mp4"
file_path = "저장된_동영상_경로.mp4"

# 버킷이 없으면 생성
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)

# 동영상 업로드
client.fput_object(bucket_name, object_name, file_path)

# 업로드된 파일의 URL 생성
video_url = f"http://minio.local/{bucket_name}/{object_name}"