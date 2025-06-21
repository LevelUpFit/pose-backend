import numpy as np
import math

#무릎 각도도  hip - knee - ankle 각도도
def calculate_angle(a, b, c):
    """
    a, b, c: 각각 (x, y) 좌표
    각도는 b를 기준으로 a-b-c 사이 각도
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# 각도 계산 함수 (3D 좌표용)
def calculate_angle_person(a, b, c):
    # 각 landmark에서 x, y, z 좌표만 뽑아서 NumPy 벡터로 변환
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


#잘못 구현한 로직
def calculate_slope_angle(a, b):
    """
    벡터 a → b (hip → knee)의 기울기 각도를 계산
    수직 기준 (y축 기준 각도)
    """
    a = np.array(a)
    b = np.array(b)

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # 수직 기준으로 변환 (0 = 수직)
    vertical_angle = abs(90 - abs(angle_deg))
    return vertical_angle


#상위 하위 10%의 평균 각도를 가지고 평균 각도 구하기
def evaluate_range_accuracy_from_angle_list(angle_list, reference_min, reference_max):
    """
    angle_list: 프레임별 관절 각도 리스트
    reference_min, reference_max: 기준 가동범위 설정
    """
    sorted_angles = sorted(angle_list)
    n = len(sorted_angles)
    k = max(1, int(n * 0.1))  # 상하위 10% 최소 1개 보장

    # 상위/하위 10% 평균
    lower_avg = np.mean(sorted_angles[:k])
    upper_avg = np.mean(sorted_angles[-k:])

    measured_range = upper_avg - lower_avg
    reference_range = reference_max - reference_min

    # 커버율 기반 정확도
    coverage_ratio = measured_range / reference_range
    accuracy = min(coverage_ratio * 100, 100)

    return round(accuracy, 2), round(lower_avg, 2), round(upper_avg, 2)