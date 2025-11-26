import cv2
import numpy as np
import tempfile
import mediapipe as mp
import math

from app.services.person import Person
from app.utils.video_utils import get_rotation_from_ffprobe, rotate_frame, correct_landmark_rotation
from app.utils.minio_client import client as minio_client, bucket_name
import app.utils.minio_client as minio_client_module

import uuid

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def analyze_lunge(landmarks, rot_width, rot_height, rotation=0):
    left_knee = correct_landmark_rotation(landmarks[25], rotation, rot_width, rot_height)
    right_knee = correct_landmark_rotation(landmarks[26], rotation, rot_width, rot_height)
    left_foot = correct_landmark_rotation(landmarks[31], rotation, rot_width, rot_height)
    right_foot = correct_landmark_rotation(landmarks[32], rotation, rot_width, rot_height)
    nose_x, _ = correct_landmark_rotation(landmarks[0], rotation, rot_width, rot_height)
    left_shoulder_x, _ = correct_landmark_rotation(landmarks[11], rotation, rot_width, rot_height)
    right_shoulder_x, _ = correct_landmark_rotation(landmarks[12], rotation, rot_width, rot_height)

    if nose_x > right_shoulder_x:
        look_direction = "right"
    else:
        look_direction = "left"

    if look_direction == "right":
        if right_foot[0] > left_foot[0]:
            front = "right"
            rear = "left"
            front_knee = right_knee
            front_foot = right_foot
        else:
            front = "left"
            rear = "right"
            front_knee = left_knee
            front_foot = left_foot
    else:
        if left_foot[0] < right_foot[0]:
            front = "left"
            rear = "right"
            front_knee = left_knee
            front_foot = left_foot
        else:
            front = "right"
            rear = "left"
            front_knee = right_knee
            front_foot = right_foot

    knee_ahead = front_knee > front_foot if front == "right" else front_knee < front_foot

    return {
        "front": front,
        "rear": rear,
        "front_knee": front_knee,
        "front_foot": front_foot,
        "knee_ahead": knee_ahead,
    }

def calc_penalty(over_distance, threshold=10, max_penalty=100):
    x = max(0, over_distance)
    exp_input = min((x / threshold) ** 2, 10)
    penalty = math.exp(exp_input) - 1
    return min(penalty, max_penalty)

def lunge_video(video_bytes: bytes, feedback_id: int) -> dict:
    person = Person()
    total_penalty = 0
    frame_count = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    # 회전 정보만 추출
    rotation = get_rotation_from_ffprobe(input_path)

    cap = cv2.VideoCapture(input_path)
    # fps는 OpenCV에서 추출 (없으면 30)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    # 첫 프레임에서 회전 후 크기 결정
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("영상 읽기 실패")
    rotated_frame = rotate_frame(frame, rotation)
    rot_height, rot_width = rotated_frame.shape[:2]

    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_tmp.name
    output_tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (rot_width, rot_height))
    
    if not out.isOpened():
        cap.release()
        raise Exception("VideoWriter 초기화 실패")

    with mp_pose.Pose(static_image_mode=False) as pose:
        # 첫 프레임 처리
        while ret:
            rotated_frame = rotate_frame(frame, rotation)
            image_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
            result1 = pose.process(image_rgb)

            if result1.pose_landmarks:
                landmarks = result1.pose_landmarks.landmark
                result = analyze_lunge(landmarks, rot_width, rot_height, rotation)
                front_knee = result["front_knee"]
                front_ankle = result["front_foot"]

                if result["front"] == "right":
                    over_distance = max(0, front_knee[0] - front_ankle[0])
                    vertical_line_x = int(front_ankle[0])
                else:
                    over_distance = max(0, front_ankle[0] - front_knee[0])
                    vertical_line_x = int(front_ankle[0])

                penalty = calc_penalty(over_distance, threshold=10, max_penalty=100) if over_distance > 0 else 0
                total_penalty += penalty
                frame_count += 1

                line_color = (0, 0, 255) if over_distance > 0 else (0, 255, 0)
                cv2.line(rotated_frame, (vertical_line_x, 0), (vertical_line_x, rot_height), line_color, 2)

                lm_dict = {i: l for i, l in enumerate(landmarks)}
                person.update_from_landmarks(lm_dict)

                cv2.putText(
                    rotated_frame,
                    f"Accuracy: {max(0, 100 - (total_penalty / frame_count if frame_count > 0 else 0)):.1f}%",
                    (rot_width - 250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 0), 3
                )

                mp_drawing.draw_landmarks(rotated_frame, result1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(rotated_frame)
            ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()  # OpenCV 리소스 정리
    print("Saved video to:", output_path)

    bucket_name = "levelupfit-videos"

    avg_penalty = total_penalty / frame_count if frame_count > 0 else 0
    accuracy = max(0, 100 - avg_penalty)

    object_name = f"{uuid.uuid4()}.mp4"
    
    try:
        # MinIO 업로드
        minio_client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=output_path,
            content_type="video/mp4"
        )
        video_url = f"https://{minio_client_module.MINIO_URL}/{bucket_name}/{object_name}"
    finally:
        # 임시 파일 정리
        import os
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

    return {
        "feedback_id": feedback_id,
        "video_url": video_url,
        "feedback_text": "sample",
        "accuracy": accuracy
    }