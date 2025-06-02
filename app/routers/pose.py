from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from app.services.mediapipe_service import analyze_and_render_video
from app.services.lunge_analyzer import lunge_video
from app.services.squat_analyzer import squat_video

router = APIRouter()

@router.post("/analyze/lunge")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    video_bytes = await file.read()
    output_path = lunge_video(video_bytes)
    return FileResponse(path=output_path, media_type="video/mp4", filename="analyzed_output.mp4")

@router.post("/analyze/squat")
async def analyze_uploaded_squat_video(file: UploadFile = File(...)):
    video_bytes = await file.read()
    output_path = squat_video(video_bytes)
    return FileResponse(path=output_path, media_type="video/mp4", filename="analyzed_output_squat.mp4")