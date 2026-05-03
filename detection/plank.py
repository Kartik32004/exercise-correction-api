import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
import os
import traceback

from .utils import extract_important_keypoints, get_drawing_color

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class PlankDetection:
    ML_MODEL_PATH = os.path.join(MODEL_DIR, "plank_model.pkl")
    INPUT_SCALER_PATH = os.path.join(MODEL_DIR, "plank_input_scaler.pkl")
    PREDICTION_PROBABILITY_THRESHOLD = 0.6

    def __init__(self) -> None:
        self.init_important_landmarks()
        self.load_machine_learning_model()

        self.previous_stage = "unknown"
        self.has_error = False
        self.correct_frames = 0
        self.total_frames = 0
        self.error_counts = {"low_back": 0, "high_back": 0}

    def init_important_landmarks(self) -> None:
        self.important_landmarks = [
            "NOSE",
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST",
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
        if not os.path.exists(self.ML_MODEL_PATH) or not os.path.exists(self.INPUT_SCALER_PATH):
            raise Exception(f"Cannot find plank model files in {MODEL_DIR}")

        with open(self.ML_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(self.INPUT_SCALER_PATH, "rb") as f2:
            self.input_scaler = pickle.load(f2)

    def clear_results(self) -> None:
        self.previous_stage = "unknown"
        self.has_error = False
        self.correct_frames = 0
        self.total_frames = 0
        self.error_counts = {"low_back": 0, "high_back": 0}

    def detect(self, mp_results, image, timestamp) -> dict:
        """Detect plank form errors and return feedback dict."""
        feedback = {
            "stage": "unknown",
            "has_error": False,
            "feedback": "Get into plank position",
            "correct_frames": self.correct_frames,
            "total_frames": self.total_frames,
            "error_counts": self.error_counts.copy(),
        }

        try:
            # Extract keypoints from frame for the input
            row = extract_important_keypoints(mp_results, self.important_landmarks)
            X = pd.DataFrame([row], columns=self.headers[1:])
            X = pd.DataFrame(self.input_scaler.transform(X))

            # Make prediction and its probability
            predicted_class = self.model.predict(X)[0]
            prediction_probability = self.model.predict_proba(X)[0]

            # Evaluate model prediction
            if (
                predicted_class == "C"
                and prediction_probability[prediction_probability.argmax()]
                >= self.PREDICTION_PROBABILITY_THRESHOLD
            ):
                current_stage = "correct"
            elif (
                predicted_class == "L"
                and prediction_probability[prediction_probability.argmax()]
                >= self.PREDICTION_PROBABILITY_THRESHOLD
            ):
                current_stage = "low back"
            elif (
                predicted_class == "H"
                and prediction_probability[prediction_probability.argmax()]
                >= self.PREDICTION_PROBABILITY_THRESHOLD
            ):
                current_stage = "high back"
            else:
                current_stage = "unknown"

            self.total_frames += 1

            if current_stage in ["low back", "high back"]:
                self.has_error = True
                if current_stage == "low back":
                    self.error_counts["low_back"] += 1
                else:
                    self.error_counts["high_back"] += 1
            else:
                self.has_error = False
                if current_stage == "correct":
                    self.correct_frames += 1

            self.previous_stage = current_stage

            # Visualization
            landmark_color, connection_color = get_drawing_color(self.has_error)
            mp_drawing.draw_landmarks(
                image, mp_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=1),
            )

            # Status box
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image,
                        str(round(prediction_probability[np.argmax(prediction_probability)], 2)),
                        (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Build feedback
            feedback["stage"] = current_stage
            feedback["has_error"] = self.has_error
            feedback["correct_frames"] = self.correct_frames
            feedback["total_frames"] = self.total_frames
            feedback["error_counts"] = self.error_counts.copy()

            if current_stage == "correct":
                pct = round((self.correct_frames / max(self.total_frames, 1)) * 100)
                feedback["feedback"] = f"Great plank form! Quality: {pct}%"
            elif current_stage == "low back":
                feedback["feedback"] = "Low back detected — Lift your hips!"
            elif current_stage == "high back":
                feedback["feedback"] = "High back detected — Lower your hips!"
            else:
                feedback["feedback"] = "Get into plank position"

        except Exception as e:
            traceback.print_exc()
            feedback["feedback"] = f"Detection error: {str(e)}"

        return feedback
