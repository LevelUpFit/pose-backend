import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np

from app.core.config import settings
from app.services.video_utils_ver2 import rotated_frame_generator
from app.utils.angle_utils import calc_penalty, calculate_angle
from app.utils.draw_utils import put_korean_text

def process_lunge_frame(frame, pose, width, height, frame_state, calculate_angle=None, tolerance=None):
    """
    1. Mediapipe로 랜드마크 추출
    2. 앞다리 판별 및 좌표/각도 계산
    3. 기준 발끝에 y축 선(초록/빨강) 그리기
    4. 랜드마크 그리기
    5. 통합 정보 박스 표시 (한글 지원)
    6. 분석값 반환
    """
    mp_pose = mp.solutions.pose
    tolerance = tolerance or settings.LUNGE_RANGE_TOLERANCE_PX
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return frame, None, None, None, None, None

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
            hip_idx, knee_idx, ankle_idx = 24, 26, 28
        else:
            knee_x = left_knee_x
            foot_x = left_foot_x
            hip_idx, knee_idx, ankle_idx = 23, 25, 27
    else:
        if left_foot_x < right_foot_x:
            knee_x = left_knee_x
            foot_x = left_foot_x
            hip_idx, knee_idx, ankle_idx = 23, 25, 27
        else:
            knee_x = right_knee_x
            foot_x = right_foot_x
            hip_idx, knee_idx, ankle_idx = 24, 26, 28

    # 무릎 각도 계산 및 y좌표
    hip = None
    knee = None
    ankle = None
    if calculate_angle:
        hip = (landmarks[hip_idx].x * width, landmarks[hip_idx].y * height)
        knee = (landmarks[knee_idx].x * width, landmarks[knee_idx].y * height)
        ankle = (landmarks[ankle_idx].x * width, landmarks[ankle_idx].y * height)
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_y = int(hip[1])
        knee_y = int(knee[1])
    else:
        knee_angle = None
        hip_y = None
        knee_y = None

    # 무릎이 발끝보다 앞으로 나갔는지 판별
    if look_direction == "right":
        over_distance = max(0, knee_x - foot_x)
    else:
        over_distance = max(0, foot_x - knee_x)
    
    # 패널티 계산 및 누적
    penalty = calc_penalty(over_distance, threshold=10, max_penalty=100) if over_distance > 0 else 0
    frame_state['total_penalty'] += penalty
    frame_state['frame_count'] += 1

    # 1️⃣ 기준 발끝에 y축 선 (무릎이 발끝 넘으면 빨간색)
    line_color = (0, 0, 255) if over_distance > 0 else (0, 255, 0)
    cv2.line(frame, (int(foot_x), 0), (int(foot_x), height), line_color, 2)

    # 2️⃣ 가동범위 시각화 (엉덩이-무릎 높이 차이)
    range_color = (255, 255, 255)
    range_diff = 0
    if knee_y is not None and hip_y is not None:
        range_diff = knee_y - hip_y
        
        if abs(range_diff) <= tolerance:
            range_color = (0, 255, 0)  # 초록색 (좋음)
        elif abs(range_diff) <= tolerance * 2:
            range_color = (0, 255, 255)  # 노란색 (중간)
        else:
            range_color = (0, 0, 255)  # 빨간색 (나쁨)
        
        # 엉덩이 높이 가로선
        cv2.line(frame, (0, hip_y), (width, hip_y), (255, 165, 0), 2)
        # 무릎 높이 가로선
        cv2.line(frame, (0, knee_y), (width, knee_y), range_color, 2)
        # 엉덩이-무릎 높이 차이를 세로선으로 표시
        if hip is not None and knee is not None:
            mid_x = int((hip[0] + knee[0]) / 2)
            cv2.line(frame, (mid_x, hip_y), (mid_x, knee_y), range_color, 3)

    # 랜드마크 그리기
    mp.solutions.drawing_utils.draw_landmarks(
        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )

    # ========== 통합 정보 박스 (한글 지원) ==========
    current_accuracy = max(0, 100 - (frame_state['total_penalty'] / frame_state['frame_count'] if frame_state['frame_count'] > 0 else 0))
    
    # 통합 정보 박스 배경
    cv2.rectangle(frame, (10, 10), (280, 100), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 10), (280, 100), (100, 100, 100), 2)
    
    # 한글 텍스트 렌더링
    acc_color = (0, 255, 0) if current_accuracy >= 90 else ((0, 255, 255) if current_accuracy >= 70 else (0, 0, 255))
    
    frame = put_korean_text(frame, f"정확도: {current_accuracy:.1f}%", (20, 15), font_size=22, color=acc_color)
    frame = put_korean_text(frame, f"가동범위: {abs(range_diff)}px", (20, 50), font_size=20, color=range_color)

    return frame, knee_x, foot_x, knee_angle, hip_y, knee_y

def save_landmark_video(input_path, output_path):
    """
    랜드마크와 기준선이 그려진 영상을 저장만 함 (분석값은 반환하지 않음)
    """
    import mediapipe as mp
    import cv2

    frame_state = {'total_penalty': 0, 'frame_count': 0}

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # H.264 코덱 시도 (브라우저 호환성 최고)
    out = None
    for codec in ['avc1', 'h264', 'H264', 'x264', 'X264']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Using codec: {codec}")
            break
    
    if out is None or not out.isOpened():
        # 모든 H.264 코덱 실패시 mp4v로 폴백
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print("Fallback to mp4v codec")
    
    if not out.isOpened():
        cap.release()
        raise Exception("VideoWriter 초기화 실패")

    mp_pose = mp.solutions.pose
    try:
        with mp_pose.Pose(static_image_mode=False) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame, *_ = process_lunge_frame(
                    frame,
                    pose,
                    width,
                    height,
                    frame_state,
                    calculate_angle=calculate_angle,
                    tolerance=settings.LUNGE_RANGE_TOLERANCE_PX,
                )
                out.write(frame)
    finally:
        cap.release()
        out.release()

def extract_front_knee_foot_xs_lunge_style(frame_gen, show_video=False):
    """
    분석값만 추출 (영상 저장 X)
    """
    mp_pose = mp.solutions.pose
    knee_xs = []
    foot_xs = []
    knee_angles = []
    hip_y_list = []
    knee_y_list = []
    width, height = None, None

    frame_state = {'total_penalty': 0, 'frame_count': 0}

    with mp_pose.Pose(static_image_mode=False) as pose:
        for frame in frame_gen:
            if width is None or height is None:
                height, width = frame.shape[:2]
            _, knee_x, foot_x, knee_angle, hip_y, knee_y = process_lunge_frame(
                frame,
                pose,
                width,
                height,
                frame_state,
                calculate_angle=calculate_angle,
                tolerance=settings.LUNGE_RANGE_TOLERANCE_PX,
            )
            if knee_x is not None:
                knee_xs.append(knee_x)
                foot_xs.append(foot_x)
                knee_angles.append(knee_angle)
                hip_y_list.append(hip_y)
                knee_y_list.append(knee_y)
            if show_video:
                cv2.imshow("Lunge Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    if show_video:
        cv2.destroyAllWindows()
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

def lunge_video_ver2(video_bytes: bytes, feedback_id: int) -> dict:
    print("="*50)
    print("🎯 LEVEL 1 분석 시작 (기본)")
    print("분석 항목: 무릎-발끝 정렬, 가동범위")
    print("="*50)
    
    import tempfile
    import uuid
    from app.utils.minio_client import client as minio_client, bucket_name
    import app.utils.minio_client as minio_client_module

    # 1. 원본 영상 임시파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    # 2. 랜드마크 영상 임시파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_tmp:
        output_path = output_tmp.name

    # 3. 랜드마크 영상 생성
    save_landmark_video(input_path, output_path)

    # 3-1. FFmpeg로 브라우저 스트리밍 최적화 (faststart)
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

    # 4. 분석 (기존대로 input_path 사용)
    frame_gen = rotated_frame_generator(input_path)
    knee_xs, foot_xs, knee_angles, hip_y_list, knee_y_list = extract_front_knee_foot_xs_lunge_style(frame_gen, show_video=False)

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

    # 5. MinIO에 랜드마크 영상 업로드
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
        video_url = f"https://{minio_client_module.MINIO_URL}/{bucket_name}/{object_name}"
    finally:
        # 임시 파일 정리
        import os
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        if 'optimized_path' in locals() and os.path.exists(optimized_path) and optimized_path != final_output:
            os.remove(optimized_path)
    
    # LLM 피드백 생성 (실패시 기존 방식으로 fallback)
    from app.services.llm_feedback import generate_feedback_level1
    feedback_text = generate_feedback_level1(accuracy, round(score, 1))
    print(round(score, 1), level, round(best_range_avg, 2))

    return {
        "feedback_id": feedback_id,
        "video_url": video_url,
        "feedback_text": feedback_text,
        "accuracy": accuracy,
        "movementRange": round(score, 1)
    }

def make_feedback_basic(accuracy, movement_range):
    feedback = []
    if accuracy >= 90:
        feedback.append("무릎이 발끝 앞으로 나가지 않았어요. 좋아요!")
    else:
        feedback.append("무릎이 발끝 앞으로 나갔습니다. 주의하세요!")
    if movement_range >= 80:
        feedback.append("가동범위가 충분합니다.")
    else:
        feedback.append("가동범위가 부족합니다. 더 깊게 내려가보세요.")
    return "\n".join(feedback)
