import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import sys
import math
import numpy as np
import tempfile
import uuid
from PIL import Image, ImageDraw, ImageFont

from app.services.video_utils_ver2 import rotated_frame_generator

# 폰트 캐싱 (매 프레임마다 로드하지 않도록)
_font_cache = {}

def get_korean_font(font_size=20):
    """한글 폰트를 캐싱하여 반환합니다."""
    if font_size in _font_cache:
        return _font_cache[font_size]
    
    font = None
    # Windows, Linux, macOS 폰트 경로들
    font_paths = [
        # Windows
        "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕
        "C:/Windows/Fonts/malgunbd.ttf",    # 맑은 고딕 Bold
        "C:/Windows/Fonts/gulim.ttc",       # 굴림
        "C:/Windows/Fonts/batang.ttc",      # 바탕
        "C:/Windows/Fonts/ngulim.ttf",      # 새굴림
        # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"[Font] Loaded: {font_path}")
            break
        except Exception:
            continue
    
    if font is None:
        print("[Font] Warning: No Korean font found, using default")
        font = ImageFont.load_default()
    
    _font_cache[font_size] = font
    return font

def put_korean_text(frame, text, position, font_size=20, color=(255, 255, 255)):
    """OpenCV 프레임에 한글 텍스트를 렌더링하는 함수"""
    try:
        # BGR -> RGB 변환
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 캐싱된 폰트 사용
        font = get_korean_font(font_size)
        
        # BGR -> RGB 색상 변환
        color_rgb = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=color_rgb)
        
        # RGB -> BGR 변환하여 반환
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[put_korean_text] Error: {e}")
        # 에러 시 원본 프레임 반환
        return frame

def annotate_lunge_video(input_path: str, output_path: str):
    # 1) 원본 비디오에서 FPS, 가로×세로 가져오기
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 2) VideoWriter 준비 (MP4)
    # H.264 코덱 시도 (브라우저 호환성 최고)
    writer = None
    for codec in ['avc1', 'h264', 'H264', 'x264', 'X264']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Using codec: {codec}")
            break
    
    if writer is None or not writer.isOpened():
        # 모든 H.264 코덱 실패시 mp4v로 폴백
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print("Fallback to mp4v codec")
    
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    # 3) Mediapipe 포즈 모델로 프레임마다 랜드마크 그리기
    with mp.solutions.pose.Pose(static_image_mode=False) as pose:
        for frame in rotated_frame_generator(input_path):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
            writer.write(frame)

    writer.release()


def extract_front_knee_foot_xs_lunge_style(frame_gen, show_video=False, save_video_path=None, fps=30):
    from app.utils.angle_utils import calculate_angle
    mp_pose = mp.solutions.pose
    knee_xs = []
    foot_xs = []
    knee_angles = []
    hip_y_list = []
    knee_y_list = []
    width, height = None, None
    video_writer = None
    total_penalty = 0
    frame_count = 0
    
    # 수축/이완 분석용 변수
    diff_y_history = []  # 프레임별 엉덩이-무릎 높이 차이
    current_phase = None  # 'contraction' or 'relaxation'
    phase_start_frame = 0
    smoothing_window = 7

    with mp_pose.Pose(static_image_mode=False) as pose:
        for frame in frame_gen:
            if width is None or height is None:
                height, width = frame.shape[:2]
                if save_video_path is not None:
                    # H.264 코덱 시도 (브라우저 호환성 최고)
                    for codec in ['avc1', 'h264', 'H264', 'x264', 'X264']:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
                        if video_writer.isOpened():
                            print(f"Using codec: {codec}")
                            break
                    
                    if video_writer is None or not video_writer.isOpened():
                        # 모든 H.264 코덱 실패시 mp4v로 폴백
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
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

                # 수축/이완 분석용 데이터 저장
                current_diff = abs(int(knee[1]) - int(hip[1]))
                diff_y_history.append(current_diff)
                
                # 수축/이완 상태 판별 (스무딩 적용)
                if len(diff_y_history) >= smoothing_window:
                    smoothed_recent = np.mean(diff_y_history[-smoothing_window:])
                    smoothed_prev = np.mean(diff_y_history[-smoothing_window-1:-1]) if len(diff_y_history) > smoothing_window else smoothed_recent
                    
                    if smoothed_recent > smoothed_prev + 3:  # 차이가 커지면 수축 (내려가는 중)
                        new_phase = 'contraction'
                    elif smoothed_recent < smoothed_prev - 3:  # 차이가 작아지면 이완 (올라가는 중)
                        new_phase = 'relaxation'
                    else:
                        new_phase = current_phase  # 유지
                    
                    if new_phase != current_phase and new_phase is not None:
                        phase_start_frame = frame_count
                        current_phase = new_phase

                # 비디오에 시각적 피드백 추가
                if save_video_path is not None:
                    # 1️⃣ 발끝 기준 수직선 (무릎이 넘으면 빨간색, 안넘으면 초록색)
                    line_color = (0, 0, 255) if over_distance > 0 else (0, 255, 0)
                    cv2.line(frame, (int(front_foot_x), 0), (int(front_foot_x), height), line_color, 2)
                    
                    # 2️⃣ 수직선 정렬 시각화 (어깨-엉덩이-반대쪽 무릎)
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
                    range_diff = knee_y_val - hip_y_val
                    
                    if abs(range_diff) <= 25:
                        range_color = (0, 255, 0)  # 초록색 (좋음)
                    elif abs(range_diff) <= 50:
                        range_color = (0, 255, 255)  # 노란색 (중간)
                    else:
                        range_color = (0, 0, 255)  # 빨간색 (나쁨)
                    
                    # 엉덩이 높이 가로선
                    cv2.line(frame, (0, hip_y_val), (width, hip_y_val), (255, 165, 0), 2)
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
                    
                    # 4️⃣ 수축/이완 상태 계산
                    phase_duration = 0
                    phase_color = (255, 255, 255)
                    phase_korean = "-"
                    if current_phase is not None:
                        phase_duration = (frame_count - phase_start_frame) / fps
                        if 2.0 <= phase_duration <= 3.0:
                            phase_color = (0, 255, 0)
                        elif phase_duration < 2.0:
                            phase_color = (0, 255, 255)
                        else:
                            phase_color = (0, 0, 255)
                        phase_korean = "수축 ↓" if current_phase == 'contraction' else "이완 ↑"
                    
                    # 통합 정보 박스 배경
                    box_height = 160 if current_phase else 120
                    cv2.rectangle(frame, (10, 10), (280, 10 + box_height), (40, 40, 40), -1)
                    cv2.rectangle(frame, (10, 10), (280, 10 + box_height), (100, 100, 100), 2)
                    
                    # 한글 텍스트 렌더링
                    # 무릎 정확도 색상
                    acc_color = (0, 255, 0) if current_accuracy >= 90 else ((0, 255, 255) if current_accuracy >= 70 else (0, 0, 255))
                    
                    frame = put_korean_text(frame, f"정확도: {current_accuracy:.1f}%", (20, 15), font_size=22, color=acc_color)
                    frame = put_korean_text(frame, f"수직 정렬: {vertical_deviation:.1f}°", (20, 45), font_size=20, color=vertical_color)
                    frame = put_korean_text(frame, f"가동범위: {abs(range_diff)}px", (20, 75), font_size=20, color=range_color)
                    
                    if current_phase is not None:
                        frame = put_korean_text(frame, f"동작: {phase_korean}", (20, 105), font_size=20, color=phase_color)
                        frame = put_korean_text(frame, f"시간: {phase_duration:.1f}초", (20, 135), font_size=20, color=phase_color)

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
            # 랜드마크가 있든 없든 프레임 저장 (없으면 원본 프레임 저장)
            if video_writer is not None:
                video_writer.write(frame)

    if show_video:
        cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
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

def make_feedback_advanced(vertical_score, movement_speed, knee_accuracy, movement_range):
    feedback = []
    # 무릎 정확도
    if knee_accuracy >= 90:
        feedback.append("무릎이 발끝 앞으로 나가지 않았어요. 좋아요!")
    else:
        feedback.append("무릎이 발끝 앞으로 나갔습니다. 주의하세요!")
    # 수직 정렬
    if vertical_score >= 90:
        feedback.append("수직 정렬이 매우 좋습니다.")
    elif vertical_score >= 60:
        feedback.append("수직 정렬이 약간 부족합니다.")
    else:
        feedback.append("수직 정렬이 많이 부족합니다.")
    # 가동범위
    if movement_range >= 80:
        feedback.append("가동범위가 충분합니다.")
    else:
        feedback.append("가동범위가 부족합니다. 더 깊게 내려가보세요.")
    # 수축/이완 속도
    if movement_speed["contractionPercent"] >= 80 and movement_speed["relaxationPercent"] >= 80:
        feedback.append("수축과 이완 속도가 적절합니다.")
    else:
        if movement_speed["contractionPercent"] < 80:
            feedback.append(f"수축 속도가 적절하지 않습니다. 평균 수축 시간: {movement_speed['avgContractionTime']:.2f}초")
        if movement_speed["relaxationPercent"] < 80:
            feedback.append(f"이완 속도가 적절하지 않습니다. 평균 이완 시간: {movement_speed['avgRelaxationTime']:.2f}초")
    return "\n".join(feedback)

def find_contraction_relaxation(diff_y_list, fps, threshold=3):
    def smooth_signal(signal, window=7):
        return np.convolve(signal, np.ones(window)/window, mode='same')
    smoothed = smooth_signal(diff_y_list)
    direction = np.sign(np.diff(smoothed))
    transitions = []
    prev_dir = direction[0]
    for i in range(1, len(direction)):
        if direction[i] != prev_dir and abs(smoothed[i+1] - smoothed[i]) > threshold:
            transitions.append(i+1)
            prev_dir = direction[i]
    segments = []
    prev = 0
    for idx in transitions:
        # diff가 커질 때 contraction, 작아질 때 relaxation (영상과 일치)
        phase = 'contraction' if smoothed[idx] > smoothed[idx-1] else 'relaxation'
        segments.append((phase, prev, idx))
        prev = idx
    if prev < len(smoothed)-1:
        phase = 'contraction' if smoothed[-1] > smoothed[-2] else 'relaxation'
        if len(segments) == 0 or segments[-1][0] != phase:
            segments.append((phase, prev, len(smoothed)-1))
    # 평균 시간 계산
    contraction_times = []
    relaxation_times = []
    for phase, start, end in segments:
        duration_sec = (end - start) / fps if fps > 0 else 0
        if phase == 'contraction':
            contraction_times.append(duration_sec)
        elif phase == 'relaxation':
            relaxation_times.append(duration_sec)
    avg_contraction = np.mean(contraction_times) if contraction_times else 0
    avg_relaxation = np.mean(relaxation_times) if relaxation_times else 0
    # 적절성 퍼센트
    def calc_percent(avg_time):
        if 2.0 <= avg_time <= 3.0:
            return 100
        elif avg_time < 2.0:
            return int((avg_time / 2.0) * 100)
        else:
            return int((3.0 / avg_time) * 100)
    contraction_percent = calc_percent(avg_contraction)
    relaxation_percent = calc_percent(avg_relaxation)
    return avg_contraction, avg_relaxation, contraction_percent, relaxation_percent

def lunge_video_level3(video_bytes: bytes, feedback_id: int) -> dict:
    print("="*50)
    print("🎯 LEVEL 3 분석 시작 (고급)")
    print("분석 항목: 무릎-발끝 정렬, 수직선 정렬, 가동범위, 수축/이완 속도")
    print("="*50)
    
    import tempfile
    import uuid
    from app.utils.minio_client import client as minio_client, bucket_name
    import app.utils.minio_client as minio_client_module

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
    input_path = input_tmp.name

    # FPS 추출
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # 1단계: 분석값 추출 (비디오 생성 없이)
    frame_gen = rotated_frame_generator(input_path)
    knee_xs, foot_xs, knee_angles, hip_y_list, knee_y_list = extract_front_knee_foot_xs_lunge_style(frame_gen, show_video=False)

    # 2단계: 시각적 피드백이 포함된 비디오 생성
    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_tmp.name
    output_tmp.close()
    
    frame_gen_visual = rotated_frame_generator(input_path)
    extract_front_knee_foot_xs_lunge_style(frame_gen_visual, show_video=False, save_video_path=output_path, fps=fps)

    # --- 수직선 정렬 분석 ---
    vertical_angles = analyze_vertical_alignment(rotated_frame_generator(input_path))
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
    diff_y_list = [abs(knee - hip) for knee, hip in zip(knee_y_list, hip_y_list)]
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

    # --- 수축/이완 속도 분석 추가 ---
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        frame_count = len(knee_y_list)
        fps = frame_count / (len(knee_y_list) / 30) if len(knee_y_list) > 0 else 30

    avg_contraction, avg_relaxation, contraction_percent, relaxation_percent = find_contraction_relaxation(diff_y_list, fps)

    movementSpeed = {
        "avgContractionTime": round(avg_contraction, 2),
        "avgRelaxationTime": round(avg_relaxation, 2),
        "contractionPercent": contraction_percent,
        "relaxationPercent": relaxation_percent
    }

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

    # (4) MinIO에 업로드
    bucket_name = "levelupfit-videos"
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
    accuracy = (knee_accuracy + vertical_score) / 2
    
    # LLM 피드백 생성 (실패시 기존 방식으로 fallback)
    from app.services.llm_feedback import generate_feedback_level3
    feedback_text = generate_feedback_level3(
        accuracy=accuracy,
        movement_range=round(score, 1),
        knee_accuracy=knee_accuracy,
        vertical_score=vertical_score,
        movement_speed=movementSpeed
    )
    print(round(score, 1), level, round(best_range_avg, 2), round(vertical_score, 1), vertical_level, round(best_vertical, 2))

    return {
        "feedback_id": feedback_id,
        "video_url": video_url,
        "feedback_text": feedback_text,
        "accuracy": accuracy,
        "movementRange": round(score, 1),
        "movementSpeed": {
            "avgContractionTime": round(avg_contraction, 2),
            "avgRelaxationTime": round(avg_relaxation, 2),
            "contractionPercent": contraction_percent,
            "relaxationPercent": relaxation_percent
        }
        # "rangeScore": round(score, 1),
        # "rangeLevel": level,
        # "rangeDiffAvg": round(best_range_avg, 2),
        # "verticalScore": round(vertical_score, 1),
        # "verticalLevel": vertical_level,
        # "verticalDeviationAvg": round(best_vertical, 2)
    }

