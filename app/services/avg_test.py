import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import subprocess
import json
import sys

def get_rotation_from_ffprobe(path):
    try:
        result = subprocess.run([
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        info = json.loads(result.stdout.decode())
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                side_data_list = stream.get('side_data_list', [])
                for side_data in side_data_list:
                    if side_data.get('side_data_type') == 'Display Matrix':
                        rotation = side_data.get('rotation', 0)
                        return int(rotation)
        return 0  # 기본값: 회전 없음
    except Exception as e:
        print("ffprobe failed:", e)
        return 0

def rotate_frame(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame

def correct_landmark_rotation(landmark, rotation, image_width, image_height):
    """
    landmark: MediaPipe landmark 객체
    rotation: 90, -90, 180 (정수)
    image_width, image_height: 프레임 크기
    return: (corrected_x, corrected_y)
    """
    x, y = landmark.x, landmark.y

    if rotation == -90:  # 시계 방향 90도
        corrected_x = 1 - y
        corrected_y = x
    elif rotation == 90:  # 반시계 방향 90도
        corrected_x = y
        corrected_y = 1 - x
    elif abs(rotation) == 180:
        corrected_x = 1 - x
        corrected_y = 1 - y
    else:  # 회전 없음
        corrected_x = x
        corrected_y = y

    return corrected_x * image_width, corrected_y * image_height

def extract_front_knee_foot_xs_lunge_style(video_path):
    mp_pose = mp.solutions.pose
    knee_xs = []
    foot_xs = []

    rotation = get_rotation_from_ffprobe(video_path)
    print(f"회전 정보: {rotation}도")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = rotate_frame(frame, rotation)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                # 회전 보정 적용
                nose_x, _ = correct_landmark_rotation(landmarks[0], rotation, width, height)
                left_shoulder_x, _ = correct_landmark_rotation(landmarks[11], rotation, width, height)
                right_shoulder_x, _ = correct_landmark_rotation(landmarks[12], rotation, width, height)
                left_knee_x, _ = correct_landmark_rotation(landmarks[25], rotation, width, height)
                right_knee_x, _ = correct_landmark_rotation(landmarks[26], rotation, width, height)
                left_foot_x, _ = correct_landmark_rotation(landmarks[31], rotation, width, height)
                right_foot_x, _ = correct_landmark_rotation(landmarks[32], rotation, width, height)

                if nose_x > right_shoulder_x:
                    look_direction = "right"
                else:
                    look_direction = "left"

                if look_direction == "right":
                    if right_foot_x > left_foot_x:
                        knee_xs.append(right_knee_x)
                        foot_xs.append(right_foot_x)
                    else:
                        knee_xs.append(left_knee_x)
                        foot_xs.append(left_foot_x)
                else:
                    if left_foot_x < right_foot_x:
                        knee_xs.append(left_knee_x)
                        foot_xs.append(left_foot_x)
                    else:
                        knee_xs.append(right_knee_x)
                        foot_xs.append(right_foot_x)
    cap.release()
    return knee_xs, foot_xs

def plot_knee_foot_distance(knee_xs, foot_xs):
    distances = [knee - foot for knee, foot in zip(knee_xs, foot_xs)]
    plt.figure(figsize=(10, 5))
    plt.plot(distances, label="Knee - Foot X Distance")
    plt.axhline(0, color='red', linestyle='--', label='Foot X')
    plt.xlabel("Frame")
    plt.ylabel("Knee X - Foot X (pixels)")
    plt.title("Distance from Foot to Knee (X axis) Over Frames")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python avg_test.py <video_path>")
        exit(1)
    video_path = sys.argv[1]
    knee_xs, foot_xs = extract_front_knee_foot_xs_lunge_style(video_path)
    plot_knee_foot_distance(knee_xs, foot_xs)