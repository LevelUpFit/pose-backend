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
    hip_y_list = []
    knee_y_list = []
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
                hip_y_list.append(int(hip[1]))
                knee_y_list.append(int(knee[1]))

                if show_video:
                    cv2.line(frame, (int(front_foot_x), 0), (int(front_foot_x), height), (0, 0, 255), 2)
                    y = int(knee[1])
                    cv2.line(frame, (0, y), (width, y), (0, 255, 255), 2)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    cv2.imshow("Lunge Video", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    if show_video:
        cv2.destroyAllWindows()
    return knee_xs, foot_xs, knee_angles, hip_y_list, knee_y_list

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

def calc_three_point_angle(a, b, c):
    # 각 b를 기준으로 a-b-c의 각도를 구함
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_vertical_alignment(frame_gen):
    mp_pose = mp.solutions.pose
    width, height = None, None
    vertical_angles = []

    with mp_pose.Pose(static_image_mode=False) as pose:
        for frame in frame_gen:
            if width is None or height is None:
                height, width = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            if not result.pose_landmarks:
                continue
            landmarks = result.pose_landmarks.landmark
            nose_x = landmarks[0].x * width
            right_shoulder_x = landmarks[12].x * width
            left_foot_x = landmarks[31].x * width
            right_foot_x = landmarks[32].x * width

            # 앞다리 판별
            if nose_x > right_shoulder_x:
                look_direction = "right"
            else:
                look_direction = "left"

            if look_direction == "right":
                if right_foot_x > left_foot_x:
                    hip_idx, shoulder_idx, opp_knee_idx = 24, 12, 25
                else:
                    hip_idx, shoulder_idx, opp_knee_idx = 23, 11, 26
            else:
                if left_foot_x < right_foot_x:
                    hip_idx, shoulder_idx, opp_knee_idx = 23, 11, 26
                else:
                    hip_idx, shoulder_idx, opp_knee_idx = 24, 12, 25

            hip = (landmarks[hip_idx].x * width, landmarks[hip_idx].y * height)
            shoulder = (landmarks[shoulder_idx].x * width, landmarks[shoulder_idx].y * height)
            opp_knee = (landmarks[opp_knee_idx].x * width, landmarks[opp_knee_idx].y * height)

            angle = calc_three_point_angle(shoulder, hip, opp_knee)
            vertical_angles.append(angle)

    return vertical_angles

def lunge_video_level2(video_bytes: bytes, feedback_id: int) -> dict:
    import tempfile
    import uuid
    from app.utils.minio_client import client as minio_client, bucket_name
    import app.utils.minio_client as minio_client_module

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    frame_gen = rotated_frame_generator(input_path)
    knee_xs, foot_xs, knee_angles, hip_y_list, knee_y_list = extract_front_knee_foot_xs_lunge_style(frame_gen, show_video=False)

    # --- 수직선 정렬 분석 ---
    vertical_angles = analyze_vertical_alignment(rotated_frame_generator(input_path))
    # 180도에 가까운 값이 수직, 10도 이내면 OK
    vertical_deviation = [abs(180 - angle) for angle in vertical_angles]
    n = len(vertical_deviation)
    k = max(1, int(n * 0.1))
    best_vertical = np.mean(sorted(vertical_deviation)[:k]) if k > 0 else 180

    vertical_tolerance = 10  # 도
    if best_vertical <= vertical_tolerance:
        vertical_score = 100
        vertical_level = "좋음"
    elif best_vertical <= vertical_tolerance * 2:
        vertical_score = max(0, 100 - ((best_vertical - vertical_tolerance) / vertical_tolerance) * 100)
        vertical_level = "중간"
    else:
        vertical_score = 0
        vertical_level = "나쁨"
    vertical_score = min(vertical_score, 100)

    distances = [knee - foot for knee, foot in zip(knee_xs, foot_xs)]
    penalties = []
    penalty_frames = []
    for idx, d in enumerate(distances):
        penalty = calc_penalty(-d) if d < 0 else 0
        penalties.append(penalty)
        if penalty > 0:
            penalty_frames.append(idx)
    avg_penalty = sum(penalties) / len(penalties) if penalties else 0
    knee_accuracy = max(0, 100 - avg_penalty)

    # 기존 movement_range 계산 (기존 기능 유지)
    sorted_diffs = sorted([h - k for h, k in zip(hip_y_list, knee_y_list)])
    n = len(sorted_diffs)
    k = max(1, int(n * 0.1))
    movement_range = round(np.mean(sorted_diffs[-k:]), 2)

    # test.py와 동일한 가동범위 평가 공식
    diff_y_list = [knee - hip for knee, hip in zip(knee_y_list, hip_y_list)]
    sorted_diffs = sorted(diff_y_list, key=lambda x: abs(x))
    n = len(sorted_diffs)
    k = max(1, int(n * 0.1))
    best_range_avg = np.mean(sorted_diffs[:k]) if k > 0 else 0

    tolerance = 25  # 픽셀
    if abs(best_range_avg) <= tolerance:
        score = 100
        level = "좋음"
    elif abs(best_range_avg) <= tolerance * 2:
        score = max(0, 100 - ((abs(best_range_avg) - tolerance) / tolerance) * 100)
        level = "중간"
    else:
        score = 0
        level = "나쁨"
    score = min(score, 100)

    bucket_name = "levelupfit-videos"
    object_name = f"{uuid.uuid4()}.mp4"
    minio_client.fput_object(bucket_name, object_name, input_path, content_type="video/mp4")
    video_url = f"https://{minio_client_module.MINIO_URL}/{bucket_name}/{object_name}"
    feedback_text = make_feedback_intermediate(vertical_score, knee_accuracy, movement_range)
    print(round(score, 1), level, round(best_range_avg, 2), round(vertical_score, 1), vertical_level, round(best_vertical, 2))

    # 허리 수직 각도 점수는 vertical_score (0~100)
    # 두 점수의 평균을 최종 accuracy로 사용
    accuracy = round((knee_accuracy + vertical_score) / 2, 1)

    return {
        "feedback_id": feedback_id,
        "video_url": video_url,
        "feedback_text": feedback_text,
        "accuracy": accuracy,
        "movementRange": round(score, 1)
    }

def make_feedback_intermediate(vertical_score, knee_accuracy, movement_range):
    feedback = []
    if knee_accuracy >= 90:
        feedback.append("무릎이 발끝 앞으로 나가지 않았어요. 좋아요!")
    else:
        feedback.append("무릎이 발끝 앞으로 나갔습니다. 주의하세요!")
    if vertical_score >= 90:
        feedback.append("어깨, 엉덩이, 반대쪽 발이 잘 수직을 이루고 있습니다.")
    elif vertical_score >= 60:
        feedback.append("수직 정렬이 약간 부족합니다.\n엉덩이와 어깨, 반대쪽 발이 일직선이 되도록 신경써보세요.")
    else:
        feedback.append("수직 정렬이 많이 부족합니다.\n자세를 더 곧게 유지하세요.")
    if movement_range >= 80:
        feedback.append("가동범위가 충분합니다.")
    else:
        feedback.append("가동범위가 부족합니다. 더 깊게 내려가보세요.")
    return "\n".join(feedback)
