import sys
import subprocess
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from app.services.video_utils_ver2 import rotated_frame_generator

def get_duration_with_ffprobe(video_path):
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def get_frame_count_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count

def extract_hip_knee_y_lists_and_show_landmarks(frame_gen, show_landmarks=True):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    hip_y_list = []
    knee_y_list = []
    width, height = None, None

    prev_diff = None
    phase_text = ""
    phase_color = (0, 0, 255)

    with mp_pose.Pose(static_image_mode=False) as pose:
        for frame_idx, frame in enumerate(frame_gen):
            if width is None or height is None:
                height, width = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                nose_x = landmarks[0].x * width
                right_shoulder_x = landmarks[12].x * width
                left_foot_x = landmarks[31].x * width
                right_foot_x = landmarks[32].x * width
                if nose_x > right_shoulder_x:
                    look_direction = "right"
                else:
                    look_direction = "left"
                if look_direction == "right":
                    if right_foot_x > left_foot_x:
                        hip_idx, knee_idx = 24, 26
                    else:
                        hip_idx, knee_idx = 23, 25
                else:
                    if left_foot_x < right_foot_x:
                        hip_idx, knee_idx = 23, 25
                    else:
                        hip_idx, knee_idx = 24, 26
                hip_x = int(landmarks[hip_idx].x * width)
                hip_y = int(landmarks[hip_idx].y * height)
                knee_x = int(landmarks[knee_idx].x * width)
                knee_y = int(landmarks[knee_idx].y * height)
                hip_y_list.append(hip_y)
                knee_y_list.append(knee_y)

                diff = abs(knee_y - hip_y)
                if prev_diff is not None:
                    # diff가 작아지면 수축(contraction), diff가 커지면 이완(relaxation)
                    if diff < prev_diff:
                        phase_text = "contraction"
                        phase_color = (0, 0, 255)
                    elif diff > prev_diff:
                        phase_text = "relaxation"
                        phase_color = (255, 0, 0)
                prev_diff = diff

                if show_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    cv2.circle(frame, (hip_x, hip_y), 8, (0, 165, 255), -1)
                    cv2.putText(frame, f"Hip y={hip_y}", (hip_x+10, hip_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.circle(frame, (knee_x, knee_y), 8, (255, 0, 0), -1)
                    cv2.putText(frame, f"Knee y={knee_y}", (knee_x+10, knee_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"{phase_text}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, phase_color, 3)
                    scale = 0.5
                    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    cv2.imshow("Mediapipe Landmarks", small_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    if show_landmarks:
        cv2.destroyAllWindows()
    return hip_y_list, knee_y_list

def smooth_signal(signal, window=7):
    return np.convolve(signal, np.ones(window)/window, mode='same')

def find_contraction_relaxation(diff_y_list, threshold=3):
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
        # diff가 커질 때 contraction, 작아질 때 relaxation (그래프와 영상 일치)
        phase = 'contraction' if smoothed[idx] > smoothed[idx-1] else 'relaxation'
        segments.append((phase, prev, idx))
        prev = idx
    if prev < len(smoothed)-1:
        phase = 'contraction' if smoothed[-1] > smoothed[-2] else 'relaxation'
        if len(segments) == 0 or segments[-1][0] != phase:
            segments.append((phase, prev, len(smoothed)-1))
    return segments, smoothed

# 메인 부분에서 diff_y_list도 abs로!
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python time_test.py <video_path>")
        exit(1)
    video_path = sys.argv[1]

    duration = get_duration_with_ffprobe(video_path)
    frame_count = get_frame_count_opencv(video_path)
    fps = frame_count / duration if duration > 0 else 0

    print(f"실제 프레임 수: {frame_count}")
    print(f"영상 길이(초): {duration:.3f}")
    print(f"평균 FPS: {fps:.3f}")

    hip_y_list, knee_y_list = extract_hip_knee_y_lists_and_show_landmarks(
        rotated_frame_generator(video_path), show_landmarks=True
    )
    diff_y_list = [abs(knee - hip) for knee, hip in zip(knee_y_list, hip_y_list)]
    segments, smoothed = find_contraction_relaxation(diff_y_list)

    for i, (phase, start, end) in enumerate(segments):
        duration_sec = (end - start) / fps if fps > 0 else 0
        print(f"{i+1}. {phase}: {start}~{end}프레임, {duration_sec:.2f}초")

    plt.figure(figsize=(12, 6))
    plt.plot(diff_y_list, label='Original diff_y')
    plt.plot(smoothed, label='Smoothed diff_y', linewidth=2)
    for phase, start, end in segments:
        plt.axvspan(start, end, alpha=0.2,
                    color='red' if phase == 'contraction' else 'blue',
                    label=phase)
    plt.xlabel('Frame')
    plt.ylabel('Knee Y - Hip Y')
    plt.title('Contraction/Relaxation Segments')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()