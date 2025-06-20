from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from app.services.mediapipe_service import analyze_and_render_video
from app.services.lunge_analyzer import lunge_video
from app.services.lunge_analyzer import analyze_lunge
from app.services.squat_analyzer import squat_video
from app.services.squat_analyzer import analyze_squat
from app.services.lunge_analyzer_ver2 import lunge_video_ver2
from app.utils.video_utils import get_video_info
from fastapi import HTTPException
from pydantic import BaseModel
import tempfile

class AnalyzeRequest(BaseModel):
    exercise_id: int
    image_url: str

router = APIRouter()

@router.post("/analyze/lunge")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    video_bytes = await file.read()
    output_path = lunge_video(video_bytes)
    return FileResponse(path=output_path, media_type="video/mp4", filename="analyzed_output.mp4")

@router.post("/analyze/squat")
async def analyze_uploaded_squat_video(file: UploadFile = File(...),feedback_id: int = Form(...)):
    video_bytes = await file.read()
    output_path = squat_video(video_bytes)
    return FileResponse(path=output_path, media_type="video/mp4", filename="analyzed_output_squat.mp4")


exercise_analyzers = {
    19: lunge_video_ver2,
    17: squat_video,
}

@router.post("/analyze")
async def analyze(
    exercise_id: int = Form(...),
    feedback_id: int = Form(...),
    file: UploadFile = File(...)
):
    if exercise_id not in exercise_analyzers:
        raise HTTPException(status_code=400, detail="알 수 없는 운동 ID입니다")
    
    video_bytes = await file.read()
    # 임시 파일로 저장

    # 파일 경로로 get_video_info 호출
    result = exercise_analyzers[exercise_id](video_bytes,feedback_id)
    return JSONResponse(result)
