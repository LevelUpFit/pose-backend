import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

def show_knee_foot(image_path):
    mp_pose = mp.solutions.pose
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 열 수 없습니다: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    with mp_pose.Pose(static_image_mode=True) as pose:
        result = pose.process(image_rgb)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            # 왼쪽 무릎(25), 오른쪽 무릎(26), 왼발끝(31), 오른발끝(32)
            points = {
                "left_knee": (int(landmarks[25].x * width), int(landmarks[25].y * height)),
                "right_knee": (int(landmarks[26].x * width), int(landmarks[26].y * height)),
                "left_foot": (int(landmarks[31].x * width), int(landmarks[31].y * height)),
                "right_foot": (int(landmarks[32].x * width), int(landmarks[32].y * height)),
            }
            for name, (x, y) in points.items():
                cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(image, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("좌표:", points)
        else:
            print("관절을 찾지 못했습니다.")

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python show_knee_foot.py <image_path>")
        exit(1)
    show_knee_foot(sys.argv[1])