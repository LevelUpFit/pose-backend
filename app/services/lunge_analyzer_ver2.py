import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import sys
import math
import numpy as np
import tempfile
import uuid

from app.services.video_utils_ver2 import rotated_frame_generator

def extract_front_knee_foot_xs_lunge_style(frame_gen, show_video=False):
    from app.utils.angle_utils import calculate_angle
    mp_pose = mp.solutions.pose
    knee_xs = []
    foot_xs = []
    knee_angles = []
    width, height = None, None

    with mp_pose.Pose(static_image_mode=False) as pose:
        for frame in frame_gen:
            if width is None or height is None:
                height, width = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                nose_x = landmarks[0].x * width
                left_shoulder_x = landmarks[11].x * width
                right_shoulder_x = landmarks[12].x * width
                left_knee_x = landmarks[25].x * width
                right_knee_x = landmarks[26].x * width
                left_foot_x = landmarks[31].x * width
                right_foot_x = landmarks[32].x * width

                # 앞다리 판별 및 인덱스 결정
                if nose_x > right_shoulder_x:
                    look_direction = "right"
                else:
                    look_direction = "left"

                if look_direction == "right":
                    if right_foot_x > left_foot_x:
                        knee_xs.append(right_knee_x)
                        foot_xs.append(right_foot_x)
                        front_foot_x = right_foot_x
                        hip_idx, knee_idx, ankle_idx = 24, 26, 28
                    else:
                        knee_xs.append(left_knee_x)
                        foot_xs.append(left_foot_x)
                        front_foot_x = left_foot_x
                        hip_idx, knee_idx, ankle_idx = 23, 25, 27
                else:
                    if left_foot_x < right_foot_x:
                        knee_xs.append(left_knee_x)
                        foot_xs.append(left_foot_x)
                        front_foot_x = left_foot_x
                        hip_idx, knee_idx, ankle_idx = 23, 25, 27
                    else:
                        knee_xs.append(right_knee_x)
                        foot_xs.append(right_foot_x)
                        front_foot_x = right_foot_x
                        hip_idx, knee_idx, ankle_idx = 24, 26, 28

                # 앞다리 무릎 각도 계산 및 배열 저장
                hip = (landmarks[hip_idx].x * width, landmarks[hip_idx].y * height)
                knee = (landmarks[knee_idx].x * width, landmarks[knee_idx].y * height)
                ankle = (landmarks[ankle_idx].x * width, landmarks[ankle_idx].y * height)
                knee_angle = calculate_angle(hip, knee, ankle)
                knee_angles.append(knee_angle)

                if show_video:
                    cv2.line(frame, (int(front_foot_x), 0), (int(front_foot_x), height), (0, 0, 255), 2)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    cv2.imshow("Lunge Video", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    if show_video:
        cv2.destroyAllWindows()
    return knee_xs, foot_xs, knee_angles

def calc_penalty(over_distance, threshold=10, max_penalty=100):
    x = max(0, over_distance)
    exp_input = min((x / threshold) ** 2, 10)
    penalty = math.exp(exp_input) - 1
    return min(penalty, max_penalty)

def plot_knee_foot_distance(knee_xs, foot_xs):
    distances = [knee - foot for knee, foot in zip(knee_xs, foot_xs)]
    penalties = []
    penalty_frames = []
    for idx, d in enumerate(distances):
        # 음수(=무릎이 발끝보다 앞으로 나감)일 때만 패널티 부여
        penalty = calc_penalty(-d) if d < 0 else 0
        penalties.append(penalty)
        if penalty > 0:
            penalty_frames.append(idx)
    avg_penalty = sum(penalties) / len(penalties) if penalties else 0
    accuracy = max(0, 100 - avg_penalty)

    plt.figure(figsize=(10, 5))
    plt.plot(distances, label="Knee - Foot X Distance")
    plt.axhline(0, color='red', linestyle='--', label='Foot X')
    # 패널티가 발생한 프레임에 마커 표시
    if penalty_frames:
        plt.scatter(np.array(penalty_frames), np.array([distances[i] for i in penalty_frames]), 
                    color='red', label='Penalty (Accuracy↓)', zorder=5)
    plt.xlabel("Frame")
    plt.ylabel("Knee X - Foot X (pixels)")
    plt.title(f"Distance from Foot to Knee (X axis) Over Frames\nAccuracy: {accuracy:.1f}%")
    plt.legend()
    plt.tight_layout()
    plt.show()

def lunge_video_ver2(video_bytes: bytes, feedback_id: int) -> dict:
    import tempfile
    import uuid
    from app.utils.minio_client import client as minio_client, bucket_name
    import app.utils.minio_client as minio_client_module

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    frame_gen = rotated_frame_generator(input_path)
    knee_xs, foot_xs, knee_angles = extract_front_knee_foot_xs_lunge_style(frame_gen, show_video=False)

    distances = [knee - foot for knee, foot in zip(knee_xs, foot_xs)]
    penalties = []
    penalty_frames = []
    for idx, d in enumerate(distances):
        penalty = calc_penalty(-d) if d < 0 else 0
        penalties.append(penalty)
        if penalty > 0:
            penalty_frames.append(idx)
    avg_penalty = sum(penalties) / len(penalties) if penalties else 0
    accuracy = max(0, 100 - avg_penalty)


    bucket_name = "levelupfit-videos"

    # MinIO에 원본 영상 저장
    object_name = f"{uuid.uuid4()}.mp4"
    minio_client.fput_object(bucket_name, object_name, input_path, content_type="video/mp4")
    video_url = f"https://{minio_client_module.MINIO_URL}/{bucket_name}/{object_name}"

    return {
        "feedback_id": feedback_id,
        "video_url": video_url,
        "feedback_text": "sample",
        "accuracy": accuracy
        # "knee_xs": knee_xs,
        # "foot_xs": foot_xs,
        # "distances": distances,
        # "penalties": penalties,
        # "penalty_frames": penalty_frames,
        # "knee_angles": knee_angles
    }

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python lunge_analyzer_ver2.py <video_path>")
#         exit(1)
#     video_path = sys.argv[1]
#     with open(video_path, "rb") as f:
#         video_bytes = f.read()
#     result = lunge_video_ver2(video_bytes, feedback_id=1)