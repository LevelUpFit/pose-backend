import cv2
import numpy as np
import tempfile
import mediapipe as mp

from app.utils.angle_utils import calculate_angle
from app.services.person import Person

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def analyze_squat(landmarks, width, height):
    left_hip = (landmarks[23].x * width, landmarks[23].y * height)
    left_knee = (landmarks[25].x * width, landmarks[25].y * height)
    left_ankle = (landmarks[27].x * width, landmarks[27].y * height)

    right_hip = (landmarks[24].x * width, landmarks[24].y * height)
    right_knee = (landmarks[26].x * width, landmarks[26].y * height)
    right_ankle = (landmarks[28].x * width, landmarks[28].y * height)

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # 스쿼트 포즈 판별 (양쪽 무릎 각도 60~110도 사이)
    is_squat_pose = (60 <= left_knee_angle <= 110) and (60 <= right_knee_angle <= 110)

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "is_squat_pose": is_squat_pose
    }

def squat_video(video_bytes: bytes) -> str:
    person = Person()
    bending_total = 0
    bending_correct = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_tmp.name
    output_tmp.close()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 첫 프레임에서 width, height 결정
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("영상 읽기 실패")
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while ret:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result1 = pose.process(image_rgb)

            if result1.pose_landmarks:
                landmarks = result1.pose_landmarks.landmark
                result = analyze_squat(landmarks, width, height)

                lm_dict = {i: l for i, l in enumerate(landmarks)}
                person.update_from_landmarks(lm_dict)

                # 정확도 계산: bending일 때만 체크
                for leg, angle in [
                    (person.left_leg, result["left_knee_angle"]),
                    (person.right_leg, result["right_knee_angle"])
                ]:
                    if leg.movement == "Bending":
                        bending_total += 1
                        if angle >= 60:
                            bending_correct += 1

                # 정확도 계산 및 우측 상단 출력
                accuracy = (bending_correct / bending_total * 100) if bending_total > 0 else 100.0
                cv2.putText(
                    frame,
                    f"Accuracy: {accuracy:.1f}%",
                    (width - 250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 0, 0), 3
                )

                # 좌측 상단에 movement 표시
                cv2.putText(
                    frame,
                    f"Left Leg: {person.left_leg.movement}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Right Leg: {person.right_leg.movement}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2
                )

                # 각도 표시
                cv2.putText(
                    frame,
                    f"{int(result['left_knee_angle'])}",
                    (int(landmarks[25].x * width), int(landmarks[25].y * height) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"{int(result['right_knee_angle'])}",
                    (int(landmarks[26].x * width), int(landmarks[26].y * height) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2
                )

                mp_drawing.draw_landmarks(frame, result1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(frame)
            ret, frame = cap.read()

    cap.release()
    out.release()
    return output_path