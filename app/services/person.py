from app.utils.angle_utils import calculate_angle
from app.utils.angle_utils import calculate_angle_person
from app.utils.landmark_utils import to_named_landmarks

class Limb:
    def __init__(self, name):
        self.name = name
        self.prev_angle = None
        self.curr_angle = None
        self.movement = "정지"

    def update(self, angle):
        if self.curr_angle is not None:
            self.prev_angle = self.curr_angle
        self.curr_angle = angle

        if self.prev_angle is not None:
            delta = self.curr_angle - self.prev_angle
            if delta > 1:
                self.movement = "Extending"
            elif delta < -1:
                self.movement = "Bending"
            else:
                self.movement = "Stopping"

class Person:
    def __init__(self):
        self.left_leg = Limb("left_leg")
        self.right_leg = Limb("right_leg")
        # 팔, 허리 등 필요 시 추가

    def update_from_landmarks(self, landmarks: dict):
        # calculate_angle 함수는 utils에서 가져온다고 가정
        named = to_named_landmarks(landmarks)
        left_leg_angle = calculate_angle_person(named["left_hip"], named["left_knee"], named["left_ankle"])
        right_leg_angle = calculate_angle_person(named["right_hip"], named["right_knee"], named["right_ankle"])

        self.left_leg.update(left_leg_angle)
        self.right_leg.update(right_leg_angle)
