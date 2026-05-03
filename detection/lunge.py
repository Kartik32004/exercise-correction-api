import cv2
import pickle
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import traceback

from .utils import (
    calculate_angle,
    extract_important_keypoints,
    get_drawing_color,
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def analyze_knee_angle(
    mp_results,
    stage: str,
    angle_thresholds: list,
    knee_over_toe: bool = False,
    draw_to_image: tuple = None,
) -> dict:
    """Calculate angle of each knee while performer at the DOWN position"""
    results = {
        "error": None,
        "right": {"error": None, "angle": None},
        "left": {"error": None, "angle": None},
    }

    landmarks = mp_results.pose_landmarks.landmark

    # Calculate right knee angle
    right_hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
    ]
    right_knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
    ]
    right_ankle = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
    ]
    results["right"]["angle"] = calculate_angle(right_hip, right_knee, right_ankle)

    # Calculate left knee angle
    left_hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
    ]
    left_knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
    ]
    left_ankle = [
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
    ]
    results["left"]["angle"] = calculate_angle(left_hip, left_knee, left_ankle)

    # Draw to image
    if draw_to_image is not None and stage != "down":
        (image, video_dimensions) = draw_to_image
        cv2.putText(image, str(int(results["right"]["angle"])),
                    tuple(np.multiply(right_knee, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(int(results["left"]["angle"])),
                    tuple(np.multiply(left_knee, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if stage != "down":
        return results

    if knee_over_toe:
        return results

    # Evaluation
    results["error"] = False

    if angle_thresholds[0] <= results["right"]["angle"] <= angle_thresholds[1]:
        results["right"]["error"] = False
    else:
        results["right"]["error"] = True
        results["error"] = True

    if angle_thresholds[0] <= results["left"]["angle"] <= angle_thresholds[1]:
        results["left"]["error"] = False
    else:
        results["left"]["error"] = True
        results["error"] = True

    # Draw to image
    if draw_to_image is not None:
        (image, video_dimensions) = draw_to_image
        if results["error"]:
            cv2.rectangle(image, (0, 50), (120, 100), (245, 117, 16), -1)
            cv2.putText(image, "KNEE ANGLE ERROR", (10, 62), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image,
                        "LEFT KNEE" if results["left"]["error"] else "RIGHT KNEE",
                        (10, 82), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        right_color = (255, 255, 255) if not results["right"]["error"] else (0, 0, 255)
        left_color = (255, 255, 255) if not results["left"]["error"] else (0, 0, 255)

        cv2.putText(image, str(int(results["right"]["angle"])),
                    tuple(np.multiply(right_knee, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, right_color, 1, cv2.LINE_AA)
        cv2.putText(image, str(int(results["left"]["angle"])),
                    tuple(np.multiply(left_knee, video_dimensions).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, left_color, 1, cv2.LINE_AA)

    return results


class LungeDetection:
    STAGE_ML_MODEL_PATH = os.path.join(MODEL_DIR, "lunge_stage_model.pkl")
    ERR_ML_MODEL_PATH = os.path.join(MODEL_DIR, "lunge_err_model.pkl")
    INPUT_SCALER_PATH = os.path.join(MODEL_DIR, "lunge_input_scaler.pkl")

    PREDICTION_PROB_THRESHOLD = 0.8
    KNEE_ANGLE_THRESHOLD = [60, 125]

    def __init__(self) -> None:
        self.init_important_landmarks()
        self.load_machine_learning_model()

        self.current_stage = ""
        self.counter = 0
        self.has_error = False
        self.error_counts = {"knee_over_toe": 0, "knee_angle": 0}

    def init_important_landmarks(self) -> None:
        self.important_landmarks = [
            "NOSE",
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_ANKLE", "RIGHT_ANKLE",
            "LEFT_HEEL", "RIGHT_HEEL",
            "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
        ]

        self.headers = ["label"]
        for lm in self.important_landmarks:
            self.headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

    def load_machine_learning_model(self) -> None:
        if (
            not os.path.exists(self.STAGE_ML_MODEL_PATH)
            or not os.path.exists(self.INPUT_SCALER_PATH)
            or not os.path.exists(self.ERR_ML_MODEL_PATH)
        ):
            raise Exception(f"Cannot find lunge model files in {MODEL_DIR}")

        with open(self.ERR_ML_MODEL_PATH, "rb") as f:
            self.err_model = pickle.load(f)
        with open(self.STAGE_ML_MODEL_PATH, "rb") as f:
            self.stage_model = pickle.load(f)
        with open(self.INPUT_SCALER_PATH, "rb") as f2:
            self.input_scaler = pickle.load(f2)

    def clear_results(self) -> None:
        self.counter = 0
        self.current_stage = ""
        self.has_error = False
        self.error_counts = {"knee_over_toe": 0, "knee_angle": 0}

    def detect(self, mp_results, image, timestamp) -> dict:
        """Detect lunge errors and return feedback dict."""
        feedback = {
            "counter": self.counter,
            "stage": self.current_stage,
            "has_error": False,
            "feedback": "Get into lunge position",
            "error_counts": self.error_counts.copy(),
        }

        try:
            video_dimensions = [image.shape[1], image.shape[0]]

            # Extract keypoints from frame for the input
            row = extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))

            # Make prediction for stage
            stage_predicted_class = self.stage_model.predict(X)[0]
            stage_prediction_probabilities = self.stage_model.predict_proba(X)[0]
            stage_prediction_probability = round(
                stage_prediction_probabilities[stage_prediction_probabilities.argmax()], 2,
            )

            # Evaluate stage prediction for counter
            if (
                stage_predicted_class == "I"
                and stage_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
            ):
                self.current_stage = "init"
            elif (
                stage_predicted_class == "M"
                and stage_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
            ):
                self.current_stage = "mid"
            elif (
                stage_predicted_class == "D"
                and stage_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
            ):
                if self.current_stage in ["init", "mid"]:
                    self.counter += 1
                self.current_stage = "down"

            # Analyze lunge pose - Knee over toe
            k_o_t_error = None
            err_predicted_class = None
            err_prediction_probability = None
            if self.current_stage == "down":
                err_predicted_class = self.err_model.predict(X)[0]
                err_prediction_probabilities = self.err_model.predict_proba(X)[0]
                err_prediction_probability = round(
                    err_prediction_probabilities[err_prediction_probabilities.argmax()], 2,
                )

                if (
                    err_predicted_class == "L"
                    and err_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
                ):
                    k_o_t_error = "Incorrect"
                    self.has_error = True
                    self.error_counts["knee_over_toe"] += 1
                elif (
                    err_predicted_class == "C"
                    and err_prediction_probability >= self.PREDICTION_PROB_THRESHOLD
                ):
                    k_o_t_error = "Correct"
                    self.has_error = False
            else:
                self.has_error = False

            # Analyze knee angle
            analyzed_results = analyze_knee_angle(
                mp_results=mp_results,
                stage=self.current_stage,
                angle_thresholds=self.KNEE_ANGLE_THRESHOLD,
                knee_over_toe=(k_o_t_error == "Incorrect"),
                draw_to_image=(image, video_dimensions),
            )

            self.has_error = (
                analyzed_results["error"] if not self.has_error else self.has_error
            )
            if analyzed_results["error"]:
                self.error_counts["knee_angle"] += 1

            # Visualization
            landmark_color, connection_color = get_drawing_color(self.has_error)
            mp_drawing.draw_landmarks(
                image, mp_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1),
            )

            # Status box
            cv2.rectangle(image, (0, 0), (325, 40), (245, 117, 16), -1)
            cv2.putText(image, "COUNT", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image,
                        f'{str(self.counter)}, {stage_predicted_class.split(" ")[0]}, {str(stage_prediction_probability)}',
                        (5, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "KNEE_OVER_TOE", (145, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image,
                        f"{err_predicted_class}, {err_prediction_probability}, {k_o_t_error}",
                        (135, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Build feedback
            feedback["counter"] = self.counter
            feedback["stage"] = self.current_stage
            feedback["has_error"] = self.has_error
            feedback["error_counts"] = self.error_counts.copy()

            if self.has_error:
                if k_o_t_error == "Incorrect":
                    feedback["feedback"] = "Knee over toe! Step wider"
                elif analyzed_results.get("error"):
                    feedback["feedback"] = "Adjust knee angle (aim 60-125°)"
                else:
                    feedback["feedback"] = "Form needs correction"
            elif self.counter > 0:
                feedback["feedback"] = f"Good form! Rep count: {self.counter}"
            else:
                feedback["feedback"] = "Start your lunges"

        except Exception as e:
            traceback.print_exc()
            feedback["feedback"] = f"Detection error: {str(e)}"

        return feedback
