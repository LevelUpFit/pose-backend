from fastapi import FastAPI
from app.routers import pose

app = FastAPI()

# pose 라우터 등록
app.include_router(pose.router, prefix="/pose", tags=["Pose"])

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}
