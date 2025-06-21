import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
from app.services.video_utils_ver2 import rotated_frame_generator


# 명령행 인자로 영상 경로 받기
if len(sys.argv) < 2:
    print("사용법: python test.py <video_path>")
    exit(1)
video_path = sys.argv[1]

def extract_hip_knee_y_lists_and_show_landmarks(frame_gen, show_landmarks=True):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
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

                if show_landmarks:
                    # 관절 랜드마크
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    # 엉덩이와 무릎에 마커 및 y좌표 표시
                    cv2.circle(frame, (hip_x, hip_y), 8, (0, 165, 255), -1)  # 주황색
                    cv2.putText(frame, f"Hip y={hip_y}", (hip_x+10, hip_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.circle(frame, (knee_x, knee_y), 8, (255, 0, 0), -1)  # 파란색
                    cv2.putText(frame, f"Knee y={knee_y}", (knee_x+10, knee_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.imshow("Mediapipe Landmarks", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    if show_landmarks:
        cv2.destroyAllWindows()
    return hip_y_list, knee_y_list

hip_y_list, knee_y_list = extract_hip_knee_y_lists_and_show_landmarks(rotated_frame_generator(video_path), show_landmarks=True)

# 차이값 계산 (무릎 y - 엉덩이 y)
diff_y_list = [knee - hip for hip, knee in zip(hip_y_list, knee_y_list)]

# 상위 10% (가장 hip이 knee에 가까운 프레임들) 평균 구하기
sorted_diffs = sorted(diff_y_list, key=lambda x: abs(x))  # 0에 가까운 값이 상위
n = len(sorted_diffs)
k = max(1, int(n * 0.1))
best_range_avg = np.mean(sorted_diffs[:k])

print(f"상위 10% 평균(hip과 knee가 가장 가까운 구간): {best_range_avg:.2f} 픽셀")

# 오차범위 판별
tolerance = 25  # 픽셀

if abs(best_range_avg) <= tolerance:
    score = 100
else:
    # tolerance를 초과한 부분만큼만 점수 차감
    score = max(0, 100 - ((abs(best_range_avg) - tolerance) / tolerance) * 100)
score = min(score, 100)
if abs(best_range_avg) <= tolerance:
    level = "좋음"
elif abs(best_range_avg) <= tolerance * 2:
    level = "중간"
else:
    level = "나쁨"

print(f"가동범위 점수: {score:.1f}%")
print(f"가동범위 평가: {level}")

# 차이값 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(diff_y_list, marker='o', linestyle='-', color='green', label='Hip Y - Knee Y')
plt.axhline(y=0, color='gray', linestyle='--', label='y=0')
plt.axhline(y=tolerance, color='red', linestyle='--', label=f'+{tolerance}')
plt.axhline(y=-tolerance, color='red', linestyle='--', label=f'-{tolerance}')
plt.title('Hip Y - Knee Y Difference Over Frames')
plt.xlabel('Frame')
plt.ylabel('Difference (pixels)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()