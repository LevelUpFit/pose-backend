import cv2
import numpy as np
import tempfile
import mediapipe as mp
import os

from app.utils.angle_utils import calculate_angle


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def is_angle_within(angle, target=90, tolerance=5):
    return abs(angle - target) <= tolerance


def analyze_lunge(landmarks, width, height):
    left_hip = (landmarks[23].x * width, landmarks[23].y * height)
    left_knee = (landmarks[25].x * width, landmarks[25].y * height)
    left_ankle = (landmarks[27].x * width, landmarks[27].y * height)

    right_hip = (landmarks[24].x * width, landmarks[24].y * height)
    right_knee = (landmarks[26].x * width, landmarks[26].y * height)
    right_ankle = (landmarks[28].x * width, landmarks[28].y * height)

    # Z 비교: front knee 판단
    front = "left" if landmarks[25].z < landmarks[26].z else "right"
    rear = "right" if front == "left" else "left"

    front_angle = calculate_angle(
        left_hip if front == "left" else right_hip,
        left_knee if front == "left" else right_knee,
        left_ankle if front == "left" else right_ankle
    )

    rear_angle = calculate_angle(
        left_hip if rear == "left" else right_hip,
        left_knee if rear == "left" else right_knee,
        left_ankle if rear == "left" else right_ankle
    )

    if 80 <= front_angle <= 100:
        is_lunge_pose =  True
    elif (80 <= rear_angle <= 100):
        is_lunge_pose = True
    else:
        is_lunge_pose = False

    # 분석 기준 범위
    front_correct = is_angle_within(front_angle)
    rear_correct = is_angle_within(rear_angle)

    return {
        "front": front,
        "rear": rear,
        "front_angle": front_angle,
        "rear_angle": rear_angle,
        "front_correct": front_correct,
        "rear_correct": rear_correct,
        "is_lunge_pose": is_lunge_pose
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


def lunge_video(video_bytes: bytes) -> str:
    # 입력 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    # 출력 영상 저장 경로
    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_tmp.name
    output_tmp.close()

    cap = cv2.VideoCapture(input_path)

    # ⭐ 디버깅용 출력
    print("Video Opened:", cap.isOpened())
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print("Video Info:", fps, width, height)

    # 기본값 보정
    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_pose.Pose(static_image_mode=False) as pose: #false가 훨 나음
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No more frames or failed to read.")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result1 = pose.process(image_rgb)

            if result1.pose_landmarks:
                landmarks = result1.pose_landmarks.landmark
                
                #if check_pose(landmarks, width, height):
                result = analyze_lunge(landmarks, width, height)


                # 앞/뒤 무릎 좌표
                front_knee = (landmarks[25] if result["front"] == "left" else landmarks[26])
                rear_knee = (landmarks[25] if result["rear"] == "left" else landmarks[26])

                front_coord = (int(front_knee.x * width), int(front_knee.y * height))
                rear_coord = (int(rear_knee.x * width), int(rear_knee.y * height))
                    
                if result["is_lunge_pose"]:
                    # 색상 결정
                    front_color = (0, 255, 0) if result["front_correct"] else (0, 0, 255)
                    rear_color = (0, 255, 0) if result["rear_correct"] else (0, 0, 255)
                else:
                    front_color = (255, 255, 255)
                    rear_color = (255, 255, 255)
                
                

                # 텍스트 출력
                cv2.putText(
                    frame,
                    f"{int(result['front_angle'])}",
                    (front_coord[0], front_coord[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, front_color, 2
                )

                cv2.putText(
                    frame,
                    f"{int(result['rear_angle'])}",
                    (rear_coord[0], rear_coord[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, rear_color, 2
                )

                # 랜드마크 표시
                mp_drawing.draw_landmarks(frame, result1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(frame)

    cap.release()
    out.release()

    print("Saved video to:", output_path)
    return output_path