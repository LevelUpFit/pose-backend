from pydantic import BaseModel

class PoseRequest(BaseModel):
    user_id: str
    video_name: str
