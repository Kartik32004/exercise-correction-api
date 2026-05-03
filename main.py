import asyncio
import io
import json
import time
import traceback

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from detection.bicep_curl import BicepCurlDetection
from detection.plank import PlankDetection
from detection.lunge import LungeDetection

# ── App setup ────────────────────────────────────────────────────────────
app = FastAPI(title="Exercise Correction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pre-load ML models at startup ────────────────────────────────────────
print("Loading ML models...")
bicep_curl_detector = BicepCurlDetection()
plank_detector = PlankDetection()
lunge_detector = LungeDetection()
print("All ML models loaded ✅")

mp_pose = mp.solutions.pose


# ── Health endpoint ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "exercise-correction-api"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ── Helper: decode incoming JPEG blob to OpenCV image ────────────────────
def decode_frame(data: bytes) -> np.ndarray:
    """Decode raw JPEG bytes into a BGR OpenCV image."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def encode_frame(image: np.ndarray) -> bytes:
    """Encode an OpenCV BGR image to JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buffer.tobytes()


# ── WebSocket: Bicep Curl ────────────────────────────────────────────────
@app.websocket("/ws/live_bicep_curl")
async def ws_bicep_curl(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Bicep Curl client connected")

    # Each WS connection gets its own detector state
    detector = BicepCurlDetection()
    frame_count = 0

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        try:
            while True:
                data = await websocket.receive_bytes()
                image = decode_frame(data)
                if image is None:
                    continue

                frame_count += 1
                timestamp = frame_count

                # MediaPipe expects RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    feedback = detector.detect(
                        mp_results=results, image=image, timestamp=timestamp
                    )

                    # Send JSON feedback
                    await websocket.send_text(json.dumps({
                        "feedback": feedback["feedback"],
                        "count": feedback["left_counter"] + feedback["right_counter"],
                        "left_counter": feedback["left_counter"],
                        "right_counter": feedback["right_counter"],
                        "has_error": feedback["has_error"],
                        "is_correct": not feedback["has_error"],
                    }))
                else:
                    # Still draw something so client gets a frame back
                    await websocket.send_text(json.dumps({
                        "feedback": "No pose detected - adjust camera",
                    }))

                # Send processed frame back
                encoded = encode_frame(image)
                await websocket.send_bytes(encoded)

        except WebSocketDisconnect:
            print("[WS] Bicep Curl client disconnected")
        except Exception as e:
            print(f"[WS] Bicep Curl error: {e}")
            traceback.print_exc()


# ── WebSocket: Plank ─────────────────────────────────────────────────────
@app.websocket("/ws/live_plank")
async def ws_plank(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Plank client connected")

    detector = PlankDetection()
    frame_count = 0
    start_time = time.time()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        try:
            while True:
                data = await websocket.receive_bytes()
                image = decode_frame(data)
                if image is None:
                    continue

                frame_count += 1
                elapsed = int(time.time() - start_time)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    feedback = detector.detect(
                        mp_results=results, image=image, timestamp=elapsed
                    )

                    quality = 0
                    if feedback["total_frames"] > 0:
                        quality = round(
                            (feedback["correct_frames"] / feedback["total_frames"]) * 100
                        )

                    await websocket.send_text(json.dumps({
                        "feedback": feedback["feedback"],
                        "stage": feedback["stage"],
                        "has_error": feedback["has_error"],
                        "is_correct": not feedback["has_error"],
                        "hold_time": elapsed,
                        "quality": quality,
                        "error_counts": feedback["error_counts"],
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "feedback": "No pose detected - adjust camera",
                    }))

                encoded = encode_frame(image)
                await websocket.send_bytes(encoded)

        except WebSocketDisconnect:
            print("[WS] Plank client disconnected")
        except Exception as e:
            print(f"[WS] Plank error: {e}")
            traceback.print_exc()


# ── WebSocket: Lunge ─────────────────────────────────────────────────────
@app.websocket("/ws/live_lunge")
async def ws_lunge(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Lunge client connected")

    detector = LungeDetection()
    frame_count = 0

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        try:
            while True:
                data = await websocket.receive_bytes()
                image = decode_frame(data)
                if image is None:
                    continue

                frame_count += 1
                timestamp = frame_count

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    feedback = detector.detect(
                        mp_results=results, image=image, timestamp=timestamp
                    )

                    await websocket.send_text(json.dumps({
                        "feedback": feedback["feedback"],
                        "count": feedback["counter"],
                        "stage": feedback["stage"],
                        "has_error": feedback["has_error"],
                        "is_correct": not feedback["has_error"],
                        "error_counts": feedback["error_counts"],
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "feedback": "No pose detected - adjust camera",
                    }))

                encoded = encode_frame(image)
                await websocket.send_bytes(encoded)

        except WebSocketDisconnect:
            print("[WS] Lunge client disconnected")
        except Exception as e:
            print(f"[WS] Lunge error: {e}")
            traceback.print_exc()


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
