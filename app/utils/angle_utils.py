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