# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MinIO 전용 필드
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_URL: str  # .env의 MINIO_URL 이 여기에 바인딩

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 이 한 줄로 .env를 읽고 속성에 바인딩해 줍니다
settings = Settings()
