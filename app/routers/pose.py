from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from app.services.mediapipe_service import analyze_and_render_video

router = APIRouter()

@router.post("/analyze")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    video_bytes = await file.read()
    output_path = analyze_and_render_video(video_bytes)
    return FileResponse(path=output_path, media_type="video/mp4", filename="analyzed_output.mp4")