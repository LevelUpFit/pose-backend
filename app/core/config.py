from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MinIO settings
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_URL: str
    MINIO_BUCKET_NAME: str = "levelupfit-videos"

    # OpenAI API Key (LLM feedback)
    OPENAI_API_KEY: Optional[str] = None

    # Lunge analysis thresholds
    LUNGE_RANGE_TOLERANCE_PX: int = 25
    LUNGE_VERTICAL_TOLERANCE_DEG: int = 10

    # Korean font search paths used by PIL text rendering
    KOREAN_FONT_PATHS: str = ",".join(
        [
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/malgunbd.ttf",
            "C:/Windows/Fonts/gulim.ttc",
            "C:/Windows/Fonts/batang.ttc",
            "C:/Windows/Fonts/ngulim.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/NanumGothic.ttf",
        ]
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
