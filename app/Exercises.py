import numpy as np
import math
from collections import deque
import time


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product == 0:
        return None

    cos_angle = np.dot(ba, bc) / norm_product
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


class Exercise:
    exercise_id = ""
    exercise_name = ""
    tracker_ready = False

    def extract_joints(self, keypoints, frame_shape):
        return {}

    def analyze(self, joints):
        return {}

    def analyse(self, joints):
        return self.analyze(joints)

    def compute_score(self, data):
        return 0

    def feedback(self, analysis, score):
        return "No feedback"


class PendulumExercise(Exercise):
    exercise_id = "pendulum"
    exercise_name = "Pendulum Exercise"
    tracker_ready = True

    def __init__(self, side="right"):
        self.side = side.lower()
        self.buffer = deque(maxlen=30)
        self.score_history = []
        self.baseline_shoulder = None

    def _joint_ids(self):
        if self.side == "left":
            return 5, 7, 9  # shoulder, elbow, wrist
        return 6, 8, 10

    def analyse(self, joints):
        result = {
            "arm_angle": None,
            "elbow_angle": None,
            "shoulder_stability": None,
            "score": 0,
        }

        shoulder_id, elbow_id, wrist_id = self._joint_ids()

        if all(k in joints for k in [shoulder_id, elbow_id, wrist_id]):
            shoulder = joints[shoulder_id]
            elbow = joints[elbow_id]
            wrist = joints[wrist_id]

            result["elbow_angle"] = calculate_angle(shoulder, elbow, wrist)

            dx = wrist[0] - shoulder[0]
            dy = wrist[1] - shoulder[1]
            result["arm_angle"] = math.atan2(dx, dy) * 180 / math.pi

        if self.baseline_shoulder is None and shoulder_id in joints:
            self.baseline_shoulder = joints[shoulder_id]

        if self.baseline_shoulder is not None and shoulder_id in joints:
            result["shoulder_stability"] = np.linalg.norm(
                np.array(joints[shoulder_id]) - np.array(self.baseline_shoulder)
            )

        return result

    def analyze(self, joints):
        return self.analyse(joints)

    def compute_score(self, data):
        score = 100

        if data["elbow_angle"] is not None:
            score -= abs(180 - data["elbow_angle"]) * 0.5

        if data["shoulder_stability"] is not None:
            score -= data["shoulder_stability"] * 0.2

        score = max(0, min(100, score))

        self.score_history.append({
            "timestamp": time.time(),
            "score": score,
            "analysis": data,
        })

        return score

    def feedback(self, analysis, score):
        if score > 80:
            return "Good form"
        if score > 50:
            return "Adjust movement slightly and keep the shoulder relaxed"
        return "Poor control; reduce the movement and restart with a relaxed arm"

    def extract_joints(self, keypoints, frame_shape):
        h, w, _ = frame_shape
        shaped = np.squeeze(np.multiply(keypoints, [h, w, 1]))

        joints = {}
        for idx in [5, 6, 7, 8, 9, 10, 11, 12]:
            y, x, confidence = shaped[idx]
            if confidence > 0.4:
                joints[idx] = (int(x), int(y))

        return joints


class BicepCurlExercise(Exercise):
    exercise_id = "bicep_curl"
    exercise_name = "Bicep Curl"
    tracker_ready = True

    def __init__(self, side="right"):
        self.side = side.lower()
        self.score_history = []

        self.active_set = 1
        self.active_rep = 0
        self.target_sets = 2
        self.target_reps = 5

        self.curl_state = "down"
        self.last_rep_time = 0

        self.baseline_shoulder = None
        self.baseline_elbow = None

    def _joint_ids(self):
        if self.side == "left":
            return 5, 7, 9  # left shoulder, left elbow, left wrist
        return 6, 8, 10  # right shoulder, right elbow, right wrist

    def extract_joints(self, keypoints, frame_shape):
        h, w, _ = frame_shape
        shaped = np.squeeze(np.multiply(keypoints, [h, w, 1]))

        joints = {}
        for idx in [5, 6, 7, 8, 9, 10, 11, 12]:
            y, x, confidence = shaped[idx]
            if confidence > 0.35:
                joints[idx] = (int(x), int(y))

        return joints

    def analyse(self, joints):
        result = {
            "elbow_angle": None,
            "shoulder_stability": None,
            "elbow_stability": None,
            "rep_count": self.active_rep,
            "set_count": self.active_set,
            "phase": self.curl_state,
        }

        shoulder_id, elbow_id, wrist_id = self._joint_ids()

        if not all(k in joints for k in [shoulder_id, elbow_id, wrist_id]):
            return result

        shoulder = joints[shoulder_id]
        elbow = joints[elbow_id]
        wrist = joints[wrist_id]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        result["elbow_angle"] = elbow_angle

        if self.baseline_shoulder is None:
            self.baseline_shoulder = shoulder

        if self.baseline_elbow is None:
            self.baseline_elbow = elbow

        result["shoulder_stability"] = np.linalg.norm(
            np.array(shoulder) - np.array(self.baseline_shoulder)
        )

        result["elbow_stability"] = np.linalg.norm(
            np.array(elbow) - np.array(self.baseline_elbow)
        )

        # Simple rep counting:
        # Down position = arm extended, elbow angle high.
        # Up position = elbow flexed, elbow angle low.
        now = time.time()

        if elbow_angle is not None:
            if self.curl_state == "down" and elbow_angle < 70:
                self.curl_state = "up"

            elif self.curl_state == "up" and elbow_angle > 140:
                if now - self.last_rep_time > 0.8:
                    self.active_rep += 1
                    self.last_rep_time = now

                    if self.active_rep >= self.target_reps:
                        self.active_rep = 0
                        self.active_set += 1

                        if self.active_set > self.target_sets:
                            self.active_set = self.target_sets

                self.curl_state = "down"

        result["rep_count"] = self.active_rep
        result["set_count"] = self.active_set
        result["phase"] = self.curl_state

        return result

    def analyze(self, joints):
        return self.analyse(joints)

    def compute_score(self, data):
        score = 100

        elbow_angle = data.get("elbow_angle")
        shoulder_stability = data.get("shoulder_stability")
        elbow_stability = data.get("elbow_stability")

        if elbow_angle is None:
            score = 0
        else:
            # Penalise incomplete curl range.
            if self.curl_state == "up" and elbow_angle > 90:
                score -= 25

            if self.curl_state == "down" and elbow_angle < 130:
                score -= 20

        if shoulder_stability is not None:
            score -= shoulder_stability * 0.12

        if elbow_stability is not None:
            score -= elbow_stability * 0.08

        score = max(0, min(100, score))

        self.score_history.append({
            "timestamp": time.time(),
            "score": score,
            "analysis": data,
        })

        return score

    def feedback(self, analysis, score):
        elbow_angle = analysis.get("elbow_angle")

        if elbow_angle is None:
            return "Move your arm into view."

        if score > 80:
            return "Good form"

        if analysis.get("shoulder_stability") is not None and analysis["shoulder_stability"] > 70:
            return "Keep your shoulder still and avoid swinging."

        if analysis.get("elbow_stability") is not None and analysis["elbow_stability"] > 70:
            return "Keep your elbow tucked close to your side."

        if self.curl_state == "up" and elbow_angle > 90:
            return "Curl higher and bend the elbow more."

        if self.curl_state == "down" and elbow_angle < 130:
            return "Lower the arm fully with control."

        return "Control the movement speed."


SUPPORTED_EXERCISE_FACTORIES = {
    "pendulum": PendulumExercise,
    "pendulum_variation_1": PendulumExercise,
    "bicep_curl": BicepCurlExercise,
    "bicep_curl_demo": BicepCurlExercise,
}


def list_supported_exercise_ids():
    return sorted(SUPPORTED_EXERCISE_FACTORIES.keys())


def get_exercise(exercise_id, reference_summary_path=None, side="right"):
    try:
        factory = SUPPORTED_EXERCISE_FACTORIES[exercise_id]
    except KeyError as exc:
        raise ValueError(f"Unknown exercise_id: {exercise_id}") from exc

    return factory(side=side)