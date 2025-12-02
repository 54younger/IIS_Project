import cv2
import numpy as np
import torch.nn.functional as F
import threading
import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import imageio.v3 as iio
from matplotlib import pyplot as plt
from IPython.display import Image, Video

# The OpenFace detectors
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor

from openface.STAR.demo import Alignment
utility = Alignment.__init__.__globals__['utility']
if getattr(utility.set_environment, '__name__', '') != 'patched_set_environment':
    original_set_environment = utility.set_environment
    def patched_set_environment(config):
        result = original_set_environment(config)
        config.log_dir = '.'
        return result
    utility.set_environment = patched_set_environment

import numpy as np
import os
import cv2

def override_preprocess_image(self, image_path, resize: float = 1.0):
    if isinstance(image_path, np.ndarray):
        img_raw = image_path
    else:
        img_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_raw is None:
            raise ValueError(f"Failed to read image: {image_path}")

    if img_raw.ndim == 2:
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    elif img_raw.ndim == 3 and img_raw.shape[2] == 4:
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)

    img = img_raw.astype(np.float32, copy=False)
    if resize != 1.0:
        img = cv2.resize(img, None, fx=resize, fy=resize)

    img -= (104.0, 117.0, 123.0)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(self.device)
    return img, img_raw

FaceDetector.preprocess_image = override_preprocess_image


from furhat_remote_api import FurhatRemoteAPI
from google import genai


# EMOTION AGGREGATION SYSTEM
import time
from collections import deque

class EmotionAggregator:
    """Aggregates emotion predictions over time using weighted smoothing."""

    def __init__(self, window_size=30, decay_rate=0.9):
        self.window_size = window_size   # Max number of predictions to keep
        self.decay_rate = decay_rate     # Exponential decay for older predictions
        self.predictions = deque(maxlen=window_size)
        self.emo_names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
        self.lock = threading.Lock()

    def add_prediction(self, emotion, confidence, probs=None):
        """Add a new emotion prediction with timestamp."""
        with self.lock:
            self.predictions.append({
                "emotion": emotion,
                "confidence": confidence,
                "probs": probs,  # Full probability distribution
                "timestamp": time.time()
            })

    def get_aggregated_emotion(self):
        """Compute aggregated emotion using weighted smoothing."""
        with self.lock:
            if not self.predictions:
                return {"emotion": "Neutral", "confidence": 0.0}

            # Initialize emotion probability accumulator
            emotion_scores = {emo: 0.0 for emo in self.emo_names}
            total_weight = 0.0

            current_time = time.time()

            # Aggregate with exponential weighting (recent = higher weight)
            for i, pred in enumerate(self.predictions):
                age = current_time - pred["timestamp"]
                weight = self.decay_rate ** age  # Exponential decay based on time

                if pred["probs"] is not None:
                    # Use full probability distribution
                    for j, emo in enumerate(self.emo_names):
                        emotion_scores[emo] += pred["probs"][j] * weight
                else:
                    # Use single prediction
                    emotion_scores[pred["emotion"]] += pred["confidence"] * weight

                total_weight += weight

            # Normalize scores
            if total_weight > 0:
                for emo in emotion_scores:
                    emotion_scores[emo] /= total_weight

            # Get most confident emotion
            best_emotion = max(emotion_scores.items(), key=lambda x: x[1])

            return {
                "emotion": best_emotion[0],
                "confidence": best_emotion[1],
                "distribution": emotion_scores
            }

    def clear(self):
        """Clear all predictions (call at start of new user response)."""
        with self.lock:
            self.predictions.clear()

# Global emotion aggregator
emotion_aggregator = EmotionAggregator(window_size=30, decay_rate=0.95)

# CV2 LIVE EMOTION DETECTION

def cv2_vid_with_emotion():
    """
    Continuously captures video and detects emotions.
    Predictions are added to the global emotion_aggregator.
    """
    device = "cpu"
    face_model_path = "./weights/Alignment_RetinaFace.pth"
    multitask_model_path = "./weights/MTL_backbone.pth"
    emo_names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

    face_detector = FaceDetector(model_path=face_model_path, device=device)
    mtl = MultitaskPredictor(model_path=multitask_model_path, device=device)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    PROCESS_EVERY = 3  # Process every 3rd frame for efficiency

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY == 0:
            try:
                cropped_face, dets = face_detector.get_face(frame)

                if cropped_face is not None and dets is not None and len(dets) > 0:
                    # Get best face detection
                    best = dets[np.argmax([d[4] for d in dets])]

                    # Predict emotion with full probability distribution
                    logits, _, _ = mtl.predict(cropped_face)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                    idx = int(np.argmax(probs))

                    # Add prediction to aggregator with full distribution
                    emotion_aggregator.add_prediction(
                        emotion=emo_names[idx],
                        confidence=float(probs[idx]),
                        probs=probs  # Store full probability distribution
                    )

            except Exception as e:
                pass  # Silently handle errors to keep perception running

    cam.release()
    print("Perception thread stopped.")


# FURHAT + GEMINI MAIN LOOP

def start_furhat_conversation(API_KEY):

    furhat = FurhatRemoteAPI("localhost")

    client = genai.Client(api_key=API_KEY)

    # Create chat with emotion-aware system instruction
    from google.genai import types
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are Furhat, an empathetic social robot. "
                "You can detect the user's emotions through facial analysis. "
                "IMPORTANT: Always acknowledge and respond appropriately to the user's emotional state. "
                "If they seem Happy, be warm and enthusiastic. "
                "If they seem Sad, be supportive and compassionate. "
                "If they seem Angry, be calm and understanding. "
                "If they seem Fearful or Surprised, be reassuring. "
                "Keep responses brief (1-2 sentences) as you are speaking aloud."
            )
        )
    )

    print("Furhat conversation ready.")

    while True:
        # Clear emotion buffer before listening to new response
        emotion_aggregator.clear()

        # Listen to user (emotion is collected during this time)
        response = furhat.listen()
        text = response.message.strip() if response.message else ""

        # Handle speech recognition failures
        if not text or text.upper() in ["NOMATCH", "NOINPUT"]:
            print(f"[SYSTEM] Speech not recognized (got: '{text}')")

            # Get emotion to provide contextual prompt
            emotion_result = emotion_aggregator.get_aggregated_emotion()
            emotion = emotion_result["emotion"]
            conf = emotion_result["confidence"]

            if conf > 0.3:
                furhat.say(text=f"I didn't quite catch that. You seem {emotion.lower()}, could you repeat?", blocking=True)
            else:
                furhat.say(text="I didn't quite catch that. Could you please repeat?", blocking=True)
            continue

        print(f"USER: {text}")

        # Get aggregated emotion from entire response window
        emotion_result = emotion_aggregator.get_aggregated_emotion()
        emotion = emotion_result["emotion"]
        conf = emotion_result["confidence"]

        # Show aggregated result
        print(f"[AGGREGATED EMOTION] {emotion} ({conf*100:.1f}%)")
        if "distribution" in emotion_result:
            top_3 = sorted(emotion_result["distribution"].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top 3: {', '.join([f'{e}={v*100:.0f}%' for e, v in top_3])}")

        # Build emotion-aware prompt
        if conf > 0.3:  # Only mention emotion if confidence is reasonable
            prompt = (
                f"[DETECTED EMOTION: {emotion} ({conf*100:.0f}% confidence)]\n"
                f"User said: '{text}'\n\n"
                f"Respond empathetically to their {emotion.lower()} emotion."
            )
        else:
            prompt = f"User said: '{text}'\n\nRespond naturally."

        reply = chat.send_message(prompt).text.strip()
        furhat.say(text=reply, blocking=True)

        print(f"FURHAT: {reply}")

        if text.lower() in ["bye", "exit", "quit"]:
            furhat.say(text="Goodbye!", blocking=True)
            break


if __name__ == "__main__":

    API_KEY = os.environ.get("GEMINI_API_KEY")

    # Start perception system in the background
    print("Starting emotion perception system...")
    threading.Thread(
        target=cv2_vid_with_emotion,
        daemon=True
    ).start()
    # Give camera time to initialize
    time.sleep(2)

    # Start Furhat + LLM system
    print("Starting conversation system...")
    start_furhat_conversation(API_KEY)
