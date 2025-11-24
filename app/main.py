# app/main.py
from fastapi import FastAPI
from app.routers import pose
from app.core.config import settings  # ← 여기에 import

app = FastAPI()

# pose 라우터 등록
app.include_router(pose.router, prefix="/pose", tags=["Pose"])

@app.get("/")
def read_root():
    return {
        "message": "Hello from FastAPI!",
        "minio_endpoint": settings.MINIO_URL
    }

# 헬스체크 엔드포인트 (Docker 헬스체크용)
@app.get("/health")
async def health_check():
    """
    컨테이너 헬스체크 엔드포인트
    """
    return {
        "status": "healthy",
        "service": "pose-analysis",
        "version": "1.0.0"
    }