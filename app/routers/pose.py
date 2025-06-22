from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from app.services.mediapipe_service import analyze_and_render_video
from app.services.lunge_analyzer import lunge_video
from app.services.lunge_analyzer import analyze_lunge
from app.services.squat_analyzer import squat_video
from app.services.squat_analyzer import analyze_squat
from app.services.lunge_analyzer_ver2 import lunge_video_ver2
from app.services.lunge_analyzer_level2 import lunge_video_level2
from app.services.lunge_analyzer_level3 import lunge_video_level2 as lunge_video_level3
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
    19: lunge_video_level2,
    17: squat_video,
}

@router.post("/analyze")
async def analyze(
    exercise_id: int = Form(...),
    level: int = Form(...),
    feedback_id: int = Form(...),
    file: UploadFile = File(...)
):
    if exercise_id not in exercise_analyzers:
        raise HTTPException(status_code=400, detail="알 수 없는 운동 ID입니다")
    
    video_bytes = await file.read()

    # 운동별 난이도 분기
    if exercise_id == 19:  # 런지
        if level == 1:
            result = lunge_video_ver2(video_bytes, feedback_id)
        elif level == 2:
            result = lunge_video_level2(video_bytes, feedback_id)
        elif level == 3:
            result = lunge_video_level3(video_bytes, feedback_id)
        else:
            raise HTTPException(status_code=400, detail="알 수 없는 난이도입니다")
    elif exercise_id == 17:  # 스쿼트
        result = squat_video(video_bytes)
    else:
        raise HTTPException(status_code=400, detail="알 수 없는 운동 ID입니다")

    return JSONResponse(result)
