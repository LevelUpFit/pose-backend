import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np

from app.core.config import settings
from app.services.video_utils_ver2 import rotated_frame_generator
from app.utils.angle_utils import calc_penalty, calc_three_point_angle, calculate_angle
from app.utils.draw_utils import put_korean_text

def extract_front_knee_foot_xs_lunge_style(frame_gen, show_video=False, save_video_path=None):
    mp_pose = mp.solutions.pose
    knee_xs = []
    foot_xs = []
    knee_angles = []
    hip_y_list = []
    knee_y_list = []
    width, height = None, None
    video_writer = None
    frame_penalties = []  # 프레임별 패널티 저장
    total_penalty = 0
    frame_count = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        for frame in frame_gen:
            if width is None or height is None:
                height, width = frame.shape[:2]
                # VideoWriter 초기화 (save_video_path가 주어진 경우)
                if save_video_path is not None:
                    # H.264 코덱 시도 (브라우저 호환성 최고)
                    for codec in ['avc1', 'h264', 'H264', 'x264', 'X264']:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video_writer = cv2.VideoWriter(save_video_path, fourcc, 30, (width, height))
                        if video_writer.isOpened():
                            print(f"Using codec: {codec}")
                            break
                    
                    if video_writer is None or not video_writer.isOpened():
                        # 모든 H.264 코덱 실패시 mp4v로 폴백
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(save_video_path, fourcc, 30, (width, height))
                        print("Fallback to mp4v codec")

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
                        knee_x = right_knee_x
                        foot_x = right_foot_x
                        front_foot_x = right_foot_x
                        hip_idx, knee_idx, ankle_idx = 24, 26, 28
                        # 수직선 정렬용 (반대쪽 = 왼쪽)
                        opp_shoulder_idx, opp_hip_idx, opp_knee_idx = 11, 23, 25
                    else:
                        knee_x = left_knee_x
                        foot_x = left_foot_x
                        front_foot_x = left_foot_x
                        hip_idx, knee_idx, ankle_idx = 23, 25, 27
                        # 수직선 정렬용 (반대쪽 = 오른쪽)
                        opp_shoulder_idx, opp_hip_idx, opp_knee_idx = 12, 24, 26
                else:
                    if left_foot_x < right_foot_x:
                        knee_x = left_knee_x
                        foot_x = left_foot_x
                        front_foot_x = left_foot_x
                        hip_idx, knee_idx, ankle_idx = 23, 25, 27
                        # 수직선 정렬용 (반대쪽 = 오른쪽)
                        opp_shoulder_idx, opp_hip_idx, opp_knee_idx = 12, 24, 26
                    else:
                        knee_x = right_knee_x
                        foot_x = right_foot_x
                        front_foot_x = right_foot_x
                        hip_idx, knee_idx, ankle_idx = 24, 26, 28
                        # 수직선 정렬용 (반대쪽 = 왼쪽)
                        opp_shoulder_idx, opp_hip_idx, opp_knee_idx = 11, 23, 25

                knee_xs.append(knee_x)
                foot_xs.append(foot_x)
                
                # 수직선 정렬 계산 (어깨-엉덩이-반대쪽 무릎)
                opp_shoulder = (landmarks[opp_shoulder_idx].x * width, landmarks[opp_shoulder_idx].y * height)
                opp_hip = (landmarks[opp_hip_idx].x * width, landmarks[opp_hip_idx].y * height)
                opp_knee_pt = (landmarks[opp_knee_idx].x * width, landmarks[opp_knee_idx].y * height)
                vertical_angle = calc_three_point_angle(opp_shoulder, opp_hip, opp_knee_pt)
                vertical_deviation = abs(180 - vertical_angle)

                # 무릎이 발끝보다 앞으로 나갔는지 판별 및 패널티 계산
                if look_direction == "right":
                    over_distance = max(0, knee_x - foot_x)
                else:
                    over_distance = max(0, foot_x - knee_x)
                
                penalty = calc_penalty(over_distance, threshold=10, max_penalty=100) if over_distance > 0 else 0
                total_penalty += penalty
                frame_count += 1

                # 앞다리 무릎 각도 계산 및 배열 저장
                hip = (landmarks[hip_idx].x * width, landmarks[hip_idx].y * height)
                knee = (landmarks[knee_idx].x * width, landmarks[knee_idx].y * height)
                ankle = (landmarks[ankle_idx].x * width, landmarks[ankle_idx].y * height)
                knee_angle = calculate_angle(hip, knee, ankle)
                knee_angles.append(knee_angle)
                hip_y_list.append(int(hip[1]))
                knee_y_list.append(int(knee[1]))

                # 비디오에 시각적 피드백 추가 (save_video_path가 있을 때만)
                if save_video_path is not None:
                    # 1️⃣ 발끝 기준 수직선 (무릎이 넘으면 빨간색, 안넘으면 초록색)
                    line_color = (0, 0, 255) if over_distance > 0 else (0, 255, 0)
                    cv2.line(frame, (int(front_foot_x), 0), (int(front_foot_x), height), line_color, 2)
                    
                    # 2️⃣ 수직선 정렬 시각화 (어깨-엉덩이-반대쪽 무릎)
                    # 편차에 따라 색상 결정: ≤10도 초록, 10~20도 노랑, >20도 빨강
                    if vertical_deviation <= 10:
                        vertical_color = (0, 255, 0)  # 초록색 (좋음)
                    elif vertical_deviation <= 20:
                        vertical_color = (0, 255, 255)  # 노란색 (중간)
                    else:
                        vertical_color = (0, 0, 255)  # 빨간색 (나쁨)
                    
                    # 어깨-엉덩이-반대쪽 무릎을 연결하는 선
                    cv2.line(frame, (int(opp_shoulder[0]), int(opp_shoulder[1])), 
                             (int(opp_hip[0]), int(opp_hip[1])), vertical_color, 3)
                    cv2.line(frame, (int(opp_hip[0]), int(opp_hip[1])), 
                             (int(opp_knee_pt[0]), int(opp_knee_pt[1])), vertical_color, 3)
                    
                    # 세 점에 원 표시
                    cv2.circle(frame, (int(opp_shoulder[0]), int(opp_shoulder[1])), 8, vertical_color, -1)
                    cv2.circle(frame, (int(opp_hip[0]), int(opp_hip[1])), 8, vertical_color, -1)
                    cv2.circle(frame, (int(opp_knee_pt[0]), int(opp_knee_pt[1])), 8, vertical_color, -1)
                    
                    # 3️⃣ 가동범위 시각화 (엉덩이-무릎 높이 차이)
                    hip_y_val = int(hip[1])
                    knee_y_val = int(knee[1])
                    range_diff = knee_y_val - hip_y_val  # 양수: 무릎이 엉덩이보다 아래
                    
                    # 가동범위 색상 결정: ≤25픽셀 초록(좋음), 25~50픽셀 노랑(중간), >50픽셀 빨강(나쁨)
                    if abs(range_diff) <= 25:
                        range_color = (0, 255, 0)  # 초록색 (좋음 - 무릎이 엉덩이 높이에 가까움)
                    elif abs(range_diff) <= 50:
                        range_color = (0, 255, 255)  # 노란색 (중간)
                    else:
                        range_color = (0, 0, 255)  # 빨간색 (나쁨 - 더 깊게 내려가야 함)
                    
                    # 엉덩이 높이 가로선
                    cv2.line(frame, (0, hip_y_val), (width, hip_y_val), (255, 165, 0), 2)  # 주황색 (엉덩이)
                    
                    # 무릎 높이 가로선
                    cv2.line(frame, (0, knee_y_val), (width, knee_y_val), range_color, 2)
                    
                    # 엉덩이-무릎 높이 차이를 세로선으로 표시
                    mid_x = int((hip[0] + knee[0]) / 2)
                    cv2.line(frame, (mid_x, hip_y_val), (mid_x, knee_y_val), range_color, 3)
                    
                    # 랜드마크 그리기
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    
                    # ========== 통합 정보 박스 (한글 지원) ==========
                    current_accuracy = max(0, 100 - (total_penalty / frame_count if frame_count > 0 else 0))
                    
                    # 통합 정보 박스 배경
                    cv2.rectangle(frame, (10, 10), (280, 130), (40, 40, 40), -1)
                    cv2.rectangle(frame, (10, 10), (280, 130), (100, 100, 100), 2)
                    
                    # 한글 텍스트 렌더링
                    # 무릎 정확도 색상
                    acc_color = (0, 255, 0) if current_accuracy >= 90 else ((0, 255, 255) if current_accuracy >= 70 else (0, 0, 255))
                    
                    frame = put_korean_text(frame, f"정확도: {current_accuracy:.1f}%", (20, 15), font_size=22, color=acc_color)
                    frame = put_korean_text(frame, f"수직 정렬: {vertical_deviation:.1f}°", (20, 50), font_size=20, color=vertical_color)
                    frame = put_korean_text(frame, f"가동범위: {abs(range_diff)}px", (20, 85), font_size=20, color=range_color)

                if show_video:
                    cv2.imshow("Lunge Video", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # 프레임 저장 (랜드마크가 있든 없든)
            if video_writer is not None:
                video_writer.write(frame)

    if show_video:
        cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
    return knee_xs, foot_xs, knee_angles, hip_y_list, knee_y_list

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
    print("="*50)
    print("🎯 LEVEL 2 분석 시작 (중급)")
    print("분석 항목: 무릎-발끝 정렬, 수직선 정렬, 가동범위")
    print("="*50)
    
    import tempfile
    import uuid
    from app.utils.minio_client import client as minio_client, bucket_name
    import app.utils.minio_client as minio_client_module

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    # 1단계: 분석값 추출 (비디오 생성 없이)
    frame_gen = rotated_frame_generator(input_path)
    knee_xs, foot_xs, knee_angles, hip_y_list, knee_y_list = extract_front_knee_foot_xs_lunge_style(frame_gen, show_video=False)

    # 2단계: 시각적 피드백이 포함된 비디오 생성
    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_tmp.name
    output_tmp.close()
    
    frame_gen_visual = rotated_frame_generator(input_path)
    extract_front_knee_foot_xs_lunge_style(frame_gen_visual, show_video=False, save_video_path=output_path)

    # 3단계: FFmpeg로 브라우저 스트리밍 최적화 (faststart)
    import subprocess
    optimized_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    optimized_path = optimized_tmp.name
    optimized_tmp.close()
    
    try:
        # FFmpeg로 H.264 재인코딩 + faststart (moov atom을 파일 앞으로)
        subprocess.run([
            'ffmpeg', '-i', output_path,
            '-c:v', 'libx264',  # H.264 코덱
            '-preset', 'fast',  # 빠른 인코딩
            '-movflags', '+faststart',  # 스트리밍 최적화
            '-y',  # 덮어쓰기
            optimized_path
        ], check=True, capture_output=True)
        
        # 최적화된 파일로 교체
        final_output = optimized_path
        print(f"Optimized video with FFmpeg: {final_output}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg optimization failed: {e.stderr.decode()}")
        # FFmpeg 실패시 원본 사용
        final_output = output_path
    except FileNotFoundError:
        print("FFmpeg not found, using original video")
        final_output = output_path

    # --- 수직선 정렬 분석 ---
    vertical_angles = analyze_vertical_alignment(rotated_frame_generator(input_path))
    # 180도에 가까운 값이 수직, 10도 이내면 OK
    vertical_deviation = [abs(180 - angle) for angle in vertical_angles]
    n = len(vertical_deviation)
    k = max(1, int(n * 0.1))
    best_vertical = np.mean(sorted(vertical_deviation)[:k]) if k > 0 else 180

    vertical_tolerance = settings.LUNGE_VERTICAL_TOLERANCE_DEG
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

    tolerance = settings.LUNGE_RANGE_TOLERANCE_PX
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

    bucket_name = settings.MINIO_BUCKET_NAME
    object_name = f"{uuid.uuid4()}.mp4"
    try:
        import os
        file_size = os.path.getsize(final_output)
        with open(final_output, 'rb') as file_data:
            minio_client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=file_data,
                length=file_size,
                content_type="video/mp4"
            )
    finally:
        # 임시 파일 정리
        import os
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        if 'optimized_path' in locals() and os.path.exists(optimized_path) and optimized_path != final_output:
            os.remove(optimized_path)
    video_url = f"https://{minio_client_module.MINIO_URL}/{bucket_name}/{object_name}"
    
    # 허리 수직 각도 점수는 vertical_score (0~100)
    # 두 점수의 평균을 최종 accuracy로 사용
    accuracy = round((knee_accuracy + vertical_score) / 2, 1)
    
    # LLM 피드백 생성 (실패시 기존 방식으로 fallback)
    from app.services.llm_feedback import generate_feedback_level2
    feedback_text = generate_feedback_level2(
        accuracy=accuracy,
        movement_range=round(score, 1),
        knee_accuracy=knee_accuracy,
        vertical_score=vertical_score
    )
    print(round(score, 1), level, round(best_range_avg, 2), round(vertical_score, 1), vertical_level, round(best_vertical, 2))

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
