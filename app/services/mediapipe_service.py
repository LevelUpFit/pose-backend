import cv2
import numpy as np
import tempfile
import mediapipe as mp
import os

from app.utils.angle_utils import calculate_angle


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def analyze_and_render_video(video_bytes: bytes) -> str:
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
            result = pose.process(image_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                #왼쪽 다리 각도
                left_shoulder = (landmarks[11].x * width, landmarks[11].y * height)
                left_hip = (landmarks[23].x * width, landmarks[23].y * height)
                left_knee = (landmarks[25].x * width, landmarks[25].y * height)
                left_ankle = (landmarks[27].x * width, landmarks[27].y * height)

                left_leg_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                cv2.putText(
                    frame,
                    f"L-Angle: {int(left_leg_angle)} deg",
                    (int(left_hip[0]), int(left_hip[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"R-Angle: {int(left_knee_angle)} deg",
                    (int(left_knee[0]), int(left_knee[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2
                )

                #오른쪽 다리 각도
                right_shoulder = (landmarks[12].x * width, landmarks[12].y * height)
                right_hip = (landmarks[24].x * width, landmarks[24].y * height)
                right_knee = (landmarks[26].x * width, landmarks[26].y * height)
                right_ankle = (landmarks[28].x * width, landmarks[28].y * height)

                right_leg_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                color = 0
                if right_leg_angle > 90:
                    color = (0,0,255)
                else:
                    color = (0,255,0)
                cv2.putText(
                    frame,
                    f"R-Angle: {int(right_leg_angle)} deg",
                    (int(right_hip[0]), int(right_hip[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,                    
                    0.9, color, 2
                )

                cv2.putText(
                    frame,
                    f"R-Angle: {int(right_knee_angle)} deg",
                    (int(right_knee[0]), int(right_knee[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2
                )

                # 랜드마크 표시
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(frame)

    cap.release()
    out.release()

    print("Saved video to:", output_path)
    return output_path