import cv2
import numpy as np
import tempfile
import mediapipe as mp
import os
import math

from app.utils.angle_utils import calculate_angle
from app.services.person import Person
from app.utils.video_utils import get_video_info, correct_video_orientation
from app.utils.minio_client import client as minio_client, bucket_name
import app.utils.minio_client as minio_client_module

import uuid

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def is_angle_within(angle, target=90, tolerance=5):
    return abs(angle - target) <= tolerance


def analyze_lunge(landmarks, width, height):
    # 좌표 추출
    left_knee = (landmarks[25].x * width, landmarks[25].y * height)
    right_knee = (landmarks[26].x * width, landmarks[26].y * height)
    left_foot = (landmarks[31].x * width, landmarks[31].y * height)   # left_foot_index
    right_foot = (landmarks[32].x * width, landmarks[32].y * height) # right_foot_index

    # 바라보는 방향 판단: 코와 양쪽 어깨의 x좌표 비교
    nose_x = landmarks[0].x * width
    left_shoulder_x = landmarks[11].x * width
    right_shoulder_x = landmarks[12].x * width

    # 바라보는 방향: 코가 오른쪽 어깨보다 오른쪽이면 오른쪽 바라봄, 아니면 왼쪽
    if nose_x > right_shoulder_x:
        look_direction = "right"
    else:
        look_direction = "left"

    # 바라보는 방향 쪽으로 더 나와있는 다리가 앞다리
    if look_direction == "right":
        # 오른발이 더 오른쪽에 있으면 오른발이 앞, 아니면 왼발이 앞
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
        # 왼발이 더 왼쪽에 있으면 왼발이 앞, 아니면 오른발이 앞
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

    # 무릎이 발끝보다 앞으로 나갔는지 판단 (발끝 기준)
    knee_ahead = front_knee[0] > front_foot[0] if front == "right" else front_knee[0] < front_foot[0]

    # # 각도 계산
    # left_hip = (landmarks[23].x * width, landmarks[23].y * height)
    # left_angle = calculate_angle(left_hip, left_knee, left_foot)
    # right_hip = (landmarks[24].x * width, landmarks[24].y * height)
    # right_angle = calculate_angle(right_hip, right_knee, right_foot)

    # # 각도 기준으로 올바른 자세 판별 (예: 80~120도)
    # front_correct = is_angle_within(left_angle if front == "left" else right_angle)
    # rear_correct = is_angle_within(right_angle if front == "left" else left_angle)
    # is_lunge_pose = front_correct and rear_correct

    return {
        "front": front,
        "rear": rear,
        "front_knee": front_knee,
        "front_foot": front_foot,
        "knee_ahead": knee_ahead,
        # "front_angle": left_angle if front == "left" else right_angle,
        # "rear_angle": right_angle if front == "left" else left_angle,

    }

def check_pose(landmarks, width, height):
    left_hip = (landmarks[23].x * width, landmarks[23].y * height)
    left_knee = (landmarks[25].x * width, landmarks[25].y * height)
    left_ankle = (landmarks[27].x * width, landmarks[27].y * height)

    right_hip = (landmarks[24].x * width, landmarks[24].y * height)
    right_knee = (landmarks[26].x * width, landmarks[26].y * height)
    right_ankle = (landmarks[28].x * width, landmarks[28].y * height)

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    if 80 <= right_knee_angle <= 120:
        return True
    elif (80 <= left_knee_angle <= 120):
        return True
    else:
        return False

def calc_penalty(over_distance, threshold=10, max_penalty=100):
    # x: over_distance, threshold: 기준 픽셀, max_penalty: 최대 감점
    # clamp exp 입력값 (예: 0~10)으로 제한하여 overflow 방지
    x = max(0, over_distance)
    exp_input = min((x / threshold) ** 2, 10)  # 10 이상이면 exp(10) ≈ 22026, 충분히 큼
    penalty = math.exp(exp_input) - 1
    return min(penalty, max_penalty)

def lunge_video(video_bytes: bytes, feedback_id: int) -> dict:
    person = Person()
    total_penalty = 0  # 감점 누적
    frame_count = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_tmp.name
    output_tmp.close()

    info = get_video_info(input_path)
    rotate = info["rotate"]
    width = info["width"]
    height = info["height"]
    fps = info["fps"] or 30

    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("영상 읽기 실패")
    frame = correct_video_orientation(frame)
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while ret:
            proc_frame = correct_video_orientation(frame)
            image_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            result1 = pose.process(image_rgb)

            if result1.pose_landmarks:
                landmarks = result1.pose_landmarks.landmark
                result = analyze_lunge(landmarks, width, height)
                front_knee = result["front_knee"]
                front_ankle = result["front_foot"]

                # 기준선 감점 로직 (절대 픽셀 기준)
                if result["front"] == "right":
                    over_distance = max(0, front_knee[0] - front_ankle[0])  # 픽셀 단위
                    vertical_line_x = int(front_ankle[0])
                    knee_ahead = front_knee[0] > front_ankle[0]
                else:
                    over_distance = max(0, front_ankle[0] - front_knee[0])  # 픽셀 단위
                    vertical_line_x = int(front_ankle[0])
                    knee_ahead = front_knee[0] < front_ankle[0]

                # 감점: exp((x/10)**2) - 1, clamp 적용
                penalty = calc_penalty(over_distance, threshold=10, max_penalty=100) if over_distance > 0 else 0
                total_penalty += penalty
                frame_count += 1

                # 기준선(발끝) 수직선 그리기: 넘었으면 빨간색, 아니면 초록색
                line_color = (0, 0, 255) if over_distance > 0 else (0, 255, 0)
                cv2.line(proc_frame, (vertical_line_x, 0), (vertical_line_x, height), line_color, 2)

                # 시각화 및 텍스트 출력
                lm_dict = {i: l for i, l in enumerate(landmarks)}
                person.update_from_landmarks(lm_dict)

                cv2.putText(
                    proc_frame,
                    f"Accuracy: {max(0, 100 - (total_penalty / frame_count if frame_count > 0 else 0)):.1f}%",
                    (width - 250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 0), 3
                )

                mp_drawing.draw_landmarks(proc_frame, result1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(proc_frame)
            ret, frame = cap.read()

    cap.release()
    out.release()
    print("Saved video to:", output_path)

    bucket_name = "levelupfit-videos"

    # 정확도 계산 (100점에서 평균 감점)
    avg_penalty = total_penalty / frame_count if frame_count > 0 else 0
    accuracy = max(0, 100 - avg_penalty)

    # MinIO 업로드
    object_name = f"{uuid.uuid4()}.mp4"
    minio_client.fput_object(bucket_name, object_name, output_path, content_type="video/mp4")
    video_url = f"https://{minio_client_module.MINIO_URL}/{bucket_name}/{object_name}"

    return {
        "feedback_id": feedback_id,
        "video_url": video_url,
        "feedback_text": "sample",
        "accuracy": accuracy
    }