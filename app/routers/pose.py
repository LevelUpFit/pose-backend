from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.lunge_analyzer_level2 import lunge_video_level2
from app.services.lunge_analyzer_level3 import lunge_video_level3
from app.services.lunge_analyzer_ver2 import lunge_video_ver2
from app.services.squat_analyzer import squat_video

router = APIRouter()

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
