import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

def plot_knee_foot_x(knee_xs, foot_xs):
    plt.figure(figsize=(10, 5))
    plt.plot(knee_xs, label="Knee X")
    plt.plot(foot_xs, label="Foot X")
    plt.xlabel("Frame")
    plt.ylabel("X Coordinate (pixels)")
    plt.title("Knee and Foot X Coordinates Over Frames")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_both_knee_foot_x(left_knee_xs, right_knee_xs, left_foot_xs, right_foot_xs):
    plt.figure(figsize=(10, 5))
    plt.plot(left_knee_xs, label="Left Knee X")
    plt.plot(right_knee_xs, label="Right Knee X")
    plt.plot(left_foot_xs, label="Left Foot X")
    plt.plot(right_foot_xs, label="Right Foot X")
    plt.xlabel("Frame")
    plt.ylabel("X Coordinate (pixels)")
    plt.title("Both Knees and Feet X Coordinates Over Frames")
    plt.legend()
    plt.tight_layout()
    plt.show()

def extract_knee_foot_xs(video_path):
    mp_pose = mp.solutions.pose
    knee_xs = []
    foot_xs = []

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                # 왼쪽 무릎(25), 오른쪽 무릎(26), 왼발끝(31), 오른발끝(32)
                left_knee_x = landmarks[25].x * width
                right_knee_x = landmarks[26].x * width
                left_foot_x = landmarks[31].x * width
                right_foot_x = landmarks[32].x * width
                # 예시: 오른쪽 기준
                knee_xs.append(right_knee_x)
                foot_xs.append(right_foot_x)
    cap.release()
    return knee_xs, foot_xs

def extract_front_knee_foot_xs(video_path):
    mp_pose = mp.solutions.pose
    knee_xs = []
    foot_xs = []

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                # 무릎과 발끝 좌표
                left_knee_x = landmarks[25].x * width
                right_knee_x = landmarks[26].x * width
                left_foot_x = landmarks[31].x * width
                right_foot_x = landmarks[32].x * width

                # 앞에 나간 발(더 x값이 큰 쪽이 오른발, 작은 쪽이 왼발)
                if abs(right_foot_x - left_foot_x) > 10:  # 10픽셀 이상 차이날 때만
                    if right_foot_x > left_foot_x:
                        knee_xs.append(right_knee_x)
                        foot_xs.append(right_foot_x)
                    else:
                        knee_xs.append(left_knee_x)
                        foot_xs.append(left_foot_x)
    cap.release()
    return knee_xs, foot_xs

def extract_front_knee_foot_xs_lunge_style(video_path):
    mp_pose = mp.solutions.pose
    knee_xs = []
    foot_xs = []

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                # 코와 어깨 좌표
                nose_x = landmarks[0].x * width
                left_shoulder_x = landmarks[11].x * width
                right_shoulder_x = landmarks[12].x * width
                # 무릎, 발끝 좌표
                left_knee_x = landmarks[25].x * width
                right_knee_x = landmarks[26].x * width
                left_foot_x = landmarks[31].x * width
                right_foot_x = landmarks[32].x * width

                # 바라보는 방향
                if nose_x > right_shoulder_x:
                    look_direction = "right"
                else:
                    look_direction = "left"

                # 앞다리 판단
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
    import sys
    if len(sys.argv) < 3:
        print("Usage: python avg_test.py <video_path1> <video_path2>")
        exit(1)
    video_path1 = sys.argv[1]
    video_path2 = sys.argv[2]
    knee_xs1, foot_xs1 = extract_front_knee_foot_xs_lunge_style(video_path1)
    knee_xs2, foot_xs2 = extract_front_knee_foot_xs_lunge_style(video_path2)

    distances1 = [k - f for k, f in zip(knee_xs1, foot_xs1)]
    distances2 = [k - f for k, f in zip(knee_xs2, foot_xs2)]

    plt.figure(figsize=(10, 5))
    plt.plot(distances1, label=f"Video 1: {video_path1}")
    plt.plot(distances2, label=f"Video 2: {video_path2}")
    plt.axhline(0, color='red', linestyle='--', label='Foot X')
    plt.xlabel("Frame")
    plt.ylabel("Knee X - Foot X (pixels)")
    plt.title("Distance from Foot to Knee (X axis) Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()