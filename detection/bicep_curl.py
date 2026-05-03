import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import traceback
import os

from .utils import (
    calculate_angle,
    extract_important_keypoints,
    get_drawing_color,
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class BicepPoseAnalysis:
    def __init__(
        self,
        side: str,
        stage_down_threshold: float,
        stage_up_threshold: float,
        peak_contraction_threshold: float,
        loose_upper_arm_angle_threshold: float,
        visibility_threshold: float,
    ):
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {
            "LOOSE_UPPER_ARM": 0,
            "PEAK_CONTRACTION": 0,
        }

        self.loose_upper_arm = False
        self.peak_contraction_angle = 1000

    def get_joints(self, landmarks) -> bool:
        side = self.side.upper()

        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility,
        ]

        is_visible = all([vis > self.visibility_threshold for vis in joints_visibility])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible

        self.shoulder = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y,
        ]
        self.elbow = [
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y,
        ]
        self.wrist = [
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y,
        ]

        return self.is_visible

    def analyze_pose(
        self,
        landmarks,
        frame,
        results,
        timestamp: int,
        lean_back_error: bool = False,
    ):
        has_error = False
        self.get_joints(landmarks)

        if not self.is_visible:
            return (None, None, has_error)

        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        shoulder_projection = [self.shoulder[0], 1]
        ground_upper_arm_angle = int(
            calculate_angle(self.elbow, self.shoulder, shoulder_projection)
        )

        if lean_back_error:
            return (bicep_curl_angle, ground_upper_arm_angle, has_error)

        # Evaluate LOOSE UPPER ARM error
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            has_error = True
            cv2.rectangle(frame, (350, 0), (600, 40), (245, 117, 16), -1)
            cv2.putText(frame, "ARM ERROR", (360, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "LOOSE UPPER ARM", (355, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False

        # Evaluate PEAK CONTRACTION error
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            self.peak_contraction_angle = bicep_curl_angle
        elif self.stage == "down":
            if (
                self.peak_contraction_angle != 1000
                and self.peak_contraction_angle >= self.peak_contraction_threshold
            ):
                cv2.rectangle(frame, (350, 0), (600, 40), (245, 117, 16), -1)
                cv2.putText(frame, "ARM ERROR", (360, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "WEAK PEAK CONTRACTION", (355, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                self.detected_errors["PEAK_CONTRACTION"] += 1
                has_error = True
            self.peak_contraction_angle = 1000

        return (bicep_curl_angle, ground_upper_arm_angle, has_error)

    def get_counter(self) -> int:
        return self.counter

    def reset(self):
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {"LOOSE_UPPER_ARM": 0, "PEAK_CONTRACTION": 0}
        self.loose_upper_arm = False
        self.peak_contraction_angle = 1000


class BicepCurlDetection:
    ML_MODEL_PATH = os.path.join(MODEL_DIR, "bicep_curl_model.pkl")
    INPUT_SCALER = os.path.join(MODEL_DIR, "bicep_curl_input_scaler.pkl")

    VISIBILITY_THRESHOLD = 0.65
    STAGE_UP_THRESHOLD = 100
    STAGE_DOWN_THRESHOLD = 120
    PEAK_CONTRACTION_THRESHOLD = 60
    LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40
    POSTURE_ERROR_THRESHOLD = 0.95

    def __init__(self) -> None:
        self.init_important_landmarks()
        self.load_machine_learning_model()

        self.left_arm_analysis = BicepPoseAnalysis(
            side="left",
            stage_down_threshold=self.STAGE_DOWN_THRESHOLD,
            stage_up_threshold=self.STAGE_UP_THRESHOLD,
            peak_contraction_threshold=self.PEAK_CONTRACTION_THRESHOLD,
            loose_upper_arm_angle_threshold=self.LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
            visibility_threshold=self.VISIBILITY_THRESHOLD,
        )

        self.right_arm_analysis = BicepPoseAnalysis(
            side="right",
            stage_down_threshold=self.STAGE_DOWN_THRESHOLD,
            stage_up_threshold=self.STAGE_UP_THRESHOLD,
            peak_contraction_threshold=self.PEAK_CONTRACTION_THRESHOLD,
            loose_upper_arm_angle_threshold=self.LOOSE_UPPER_ARM_ANGLE_THRESHOLD,
            visibility_threshold=self.VISIBILITY_THRESHOLD,
        )

        self.stand_posture = 0
        self.previous_stand_posture = 0
        self.has_error = False

    def init_important_landmarks(self) -> None:
        self.important_landmarks = [
            "NOSE",
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "RIGHT_ELBOW", "LEFT_ELBOW",
            "RIGHT_WRIST", "LEFT_WRIST",
            "LEFT_HIP", "RIGHT_HIP",
        ]

        self.headers = ["label"]
        for lm in self.important_landmarks:
            self.headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

    def load_machine_learning_model(self) -> None:
        if not os.path.exists(self.ML_MODEL_PATH):
            raise Exception(f"Cannot find bicep curl model at {self.ML_MODEL_PATH}")

        with open(self.ML_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(self.INPUT_SCALER, "rb") as f2:
            self.input_scaler = pickle.load(f2)

    def clear_results(self) -> None:
        self.stand_posture = 0
        self.previous_stand_posture = 0
        self.has_error = False
        self.right_arm_analysis.reset()
        self.left_arm_analysis.reset()

    def detect(self, mp_results, image, timestamp: int) -> dict:
        """Detect bicep curl errors and return feedback dict."""
        self.has_error = False
        feedback = {
            "left_counter": 0,
            "right_counter": 0,
            "has_error": False,
            "feedback": "Get into position",
            "errors": [],
        }

        try:
            video_dimensions = [image.shape[1], image.shape[0]]
            landmarks = mp_results.pose_landmarks.landmark

            # Model prediction for Lean-back error
            row = extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))

            predicted_class = self.model.predict(X)[0]
            prediction_probabilities = self.model.predict_proba(X)[0]
            class_prediction_probability = round(
                prediction_probabilities[np.argmax(prediction_probabilities)], 2
            )

            if class_prediction_probability >= self.POSTURE_ERROR_THRESHOLD:
                self.stand_posture = predicted_class

            if self.stand_posture == "L":
                self.has_error = True
                feedback["errors"].append("lean_back")

            self.previous_stand_posture = self.stand_posture

            # Arms analysis
            (left_bicep_curl_angle, left_ground_upper_arm_angle, left_arm_error) = self.left_arm_analysis.analyze_pose(
                landmarks=landmarks, frame=image, results=[],
                timestamp=timestamp, lean_back_error=(self.stand_posture == "L"),
            )

            (right_bicep_curl_angle, right_ground_upper_arm_angle, right_arm_error) = self.right_arm_analysis.analyze_pose(
                landmarks=landmarks, frame=image, results=[],
                timestamp=timestamp, lean_back_error=(self.stand_posture == "L"),
            )

            self.has_error = True if (right_arm_error or left_arm_error) else self.has_error

            if left_arm_error:
                feedback["errors"].append("left_arm_error")
            if right_arm_error:
                feedback["errors"].append("right_arm_error")

            # Visualization
            landmark_color, connection_color = get_drawing_color(self.has_error)
            mp_drawing.draw_landmarks(
                image, mp_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1),
            )

            # Status box
            cv2.rectangle(image, (0, 0), (350, 40), (245, 117, 16), -1)
            cv2.putText(image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.right_arm_analysis.counter) if self.right_arm_analysis.is_visible else "UNK",
                        (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.left_arm_analysis.counter) if self.left_arm_analysis.is_visible else "UNK",
                        (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Lean-Too-Far-Back", (165, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image,
                        str("ERROR" if self.stand_posture == "L" else "CORRECT") + f", {class_prediction_probability}",
                        (160, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Visualize angles
            if self.left_arm_analysis.is_visible and left_bicep_curl_angle is not None:
                cv2.putText(image, str(left_bicep_curl_angle),
                            tuple(np.multiply(self.left_arm_analysis.elbow, video_dimensions).astype(int)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if self.right_arm_analysis.is_visible and right_bicep_curl_angle is not None:
                cv2.putText(image, str(right_bicep_curl_angle),
                            tuple(np.multiply(self.right_arm_analysis.elbow, video_dimensions).astype(int)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            # Build feedback
            feedback["left_counter"] = self.left_arm_analysis.get_counter()
            feedback["right_counter"] = self.right_arm_analysis.get_counter()
            feedback["has_error"] = self.has_error

            total = feedback["left_counter"] + feedback["right_counter"]
            if self.has_error:
                if "lean_back" in feedback["errors"]:
                    feedback["feedback"] = "Lean back detected! Stand straight"
                elif left_arm_error or right_arm_error:
                    feedback["feedback"] = "Arm error: Keep upper arm steady"
                else:
                    feedback["feedback"] = "Form needs adjustment"
            elif total > 0:
                feedback["feedback"] = f"Good form! L:{feedback['left_counter']} R:{feedback['right_counter']}"
            else:
                feedback["feedback"] = "Start curling - keep elbows close"

        except Exception as e:
            traceback.print_exc()
            feedback["feedback"] = f"Detection error: {str(e)}"

        return feedback
