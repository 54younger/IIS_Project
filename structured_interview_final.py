import cv2
import numpy as np
import torch
import torch.nn.functional as F
import threading
import os

from dotenv import load_dotenv
import joblib
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Load environment variables from .env file
load_dotenv()

# Event to track when Furhat is listening (for real-time gestures)
IS_LISTENING = threading.Event()

def listen_blocking(furhat):
    """Wrapper for furhat.listen() that signals listening state for real-time gestures."""
    IS_LISTENING.set()
    try:
        return furhat.listen()
    finally:
        IS_LISTENING.clear()

from openface.face_detection import FaceDetector

from openface.STAR.demo import Alignment
utility = Alignment.__init__.__globals__['utility']
if getattr(utility.set_environment, '__name__', '') != 'patched_set_environment':
    original_set_environment = utility.set_environment
    def patched_set_environment(config):
        result = original_set_environment(config)
        config.log_dir = '.'
        return result
    utility.set_environment = patched_set_environment

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
import json
import random
from enum import Enum
from collections import deque

class EmotionAggregator:
    """Aggregates emotion predictions over time using weighted smoothing."""

    def __init__(self, window_size=30, decay_rate=0.98):
        self.window_size = window_size   # Max number of predictions to keep
        self.decay_rate = decay_rate     # Exponential decay for older predictions (0.98 = gentler, gives more weight to sustained emotions)
        self.predictions = deque(maxlen=window_size)
        # DINO+SVM predicts 4 interview emotions directly
        self.emo_names = ["Nervous", "Defensive", "Confident", "Neutral"]
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
            for pred in self.predictions:
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

            #print(f"\n[AGGREGATION DEBUG] Predictions in window: {len(self.predictions)}")
            #print(f"[AGGREGATION DEBUG] Weighted average scores: {dict((k, f'{v:.3f}') for k, v in emotion_scores.items())}")

            # Neutral is the baseline - we care about detecting deviations
            # Pick the highest non-neutral emotion for behavior/difficulty decisions
            non_neutral_scores = {k: v for k, v in emotion_scores.items() if k != "Neutral"}
            best_non_neutral = max(non_neutral_scores.items(), key=lambda x: x[1])

            # Only report Neutral if:
            # 1. ALL non-neutral emotions are very low (< 0.10), OR
            # 2. Neutral is very dominant (> 0.70)
            all_non_neutral_low = all(v < 0.10 for v in non_neutral_scores.values())
            neutral_very_high = emotion_scores["Neutral"] > 0.70

            if all_non_neutral_low or neutral_very_high:
                best_emotion = ("Neutral", emotion_scores["Neutral"])
               # print(f"[AGGREGATION DEBUG] Truly neutral (non-neutral all <0.10 or Neutral >0.70): {best_emotion[1]:.3f}")
            else:
                # Use the strongest non-neutral signal for behavior adaptation
                best_emotion = best_non_neutral
                #print(f"[AGGREGATION DEBUG] Primary non-neutral emotion: {best_emotion[0]} ({best_emotion[1]:.3f})")

            return {
                "emotion": best_emotion[0],
                "confidence": best_emotion[1],
                "distribution": emotion_scores
            }

    def get_top_k_emotions(self, k=3, aggregated=None):
        """Get top k emotions with their confidence scores.

        Args:
            k: Number of top emotions to return
            aggregated: Optional pre-computed aggregation result to avoid duplicate computation
        """
        if aggregated is None:
            aggregated = self.get_aggregated_emotion()

        if not aggregated.get("distribution"):
            return [("Neutral", 1.0)]

        # Sort emotions by score and return top k
        sorted_emotions = sorted(
            aggregated["distribution"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_emotions[:k]

    def clear(self):
        """Clear all predictions (call at start of new user response)."""
        with self.lock:
            self.predictions.clear()

# Global emotion aggregator
emotion_aggregator = EmotionAggregator(window_size=30, decay_rate=0.98)

# EMOTION MAPPING FOR INTERVIEW COACH
class Emotion(Enum):
    # Four emotional states for interview practice.
    NERVOUS = "nervous"
    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"

class SpeechQuality(Enum):
    """Speech delivery quality assessment"""
    FLUENT = "fluent"          # Smooth, confident delivery
    HESITANT = "hesitant"      # High filler word ratio or irregular pace
    CLEAR = "clear"            # Adequate delivery

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

# CUSTOM GESTURE DEFINITIONS

CONFIDENT_BEHAVIOR = {
    "name": "Confident_behavior",
    "frames": [
        {
            "time": [0.1],
            "persist": False,
            "params": {
                "BROW_UP_LEFT": 1.0,
                "BROW_UP_RIGHT": 1.0,
                "SURPRISE": 0.3,
            },
        },
        {
            "time": [0.3, 0.7],
            "persist": False,
            "params": {"SMILE_OPEN": 0.6, "SMILE_CLOSED": 0.5},
        },
        {"time": [0.4, 0.6], "persist": False, "params": {"NECK_TILT": 6}},
        {"time": [0.9], "persist": False, "params": {"reset": True}},
    ],
    "class": "furhatos.gestures.Gesture",
}

DEFENSIVE_BEHAVIOR = {
    "name": "Defensive_behavior",
    "frames": [
        {
            "time": [0.3, 1.2],
            "persist": False,
            "params": {
                "BROW_IN_LEFT": 0.5,
                "BROW_IN_RIGHT": 0.5,
                "BROW_UP_LEFT": 0.4,
                "BROW_UP_RIGHT": 0.4,
                "EXPR_SAD": 0.15,
            },
        },
        {
            "time": [0.4, 1.3],
            "persist": False,
            "params": {"NECK_PAN": -15, "NECK_TILT": 8},
        },
        {"time": [0.9, 1.4], "persist": False, "params": {"SMILE_CLOSED": 0.4}},
        {"time": [1.8], "persist": False, "params": {"reset": True}},
    ],
    "class": "furhatos.gestures.Gesture",
}

NEUTRAL_BEHAVIOR = {
    "name": "Neutral_behavior",
    "frames": [
        {
            "time": [0.15, 0.6],
            "persist": False,
            "params": {"BROW_UP_LEFT": 0.25, "BROW_UP_RIGHT": 0.25},
        },
        {"time": [0.3], "persist": False, "params": {"NECK_TILT": 7}},
        {"time": [0.5], "persist": False, "params": {"NECK_TILT": 2}},
        {"time": [0.8], "persist": False, "params": {"reset": True}},
    ],
    "class": "furhatos.gestures.Gesture",
}

ATTENTIVE_LISTEN = {
    "name": "AttentiveListen",
    "frames": [
        {
            "time": [0.15, 0.6],
            "persist": False,
            "params": {
                "BROW_UP_LEFT": 0.25,
                "BROW_UP_RIGHT": 0.25,
                "SMILE_CLOSED": 0.3
            }
        },
        {
            "time": [0.3],
            "persist": False,
            "params": {"NECK_TILT": 7}
        },
        {
            "time": [0.5],
            "persist": False,
            "params": {"NECK_TILT": 2}
        },
        {
            "time": [0.8],
            "persist": False,
            "params": {"reset": True}
        }
    ],
    "class": "furhatos.gestures.Gesture"
}

ACTIVE_LISTEN = {
    "name": "ActiveListen",
    "frames": [
        {
            "time": [0.1, 2.0],
            "persist": False,
            "params": {
                "SMILE_CLOSED": 0.25,
                "BROW_UP_LEFT": 0.2,
                "BROW_UP_RIGHT": 0.2
            }
        },
        {
            "time": [0.4],
            "persist": False,
            "params": {"NECK_TILT": 5}
        },
        {
            "time": [0.7],
            "persist": False,
            "params": {"NECK_TILT": 2}
        },
        {
            "time": [1.1],
            "persist": False,
            "params": {"NECK_TILT": 6}
        },
        {
            "time": [1.5],
            "persist": False,
            "params": {"NECK_TILT": 2}
        },
        {
            "time": [2.3],
            "persist": False,
            "params": {"reset": True}
        }
    ],
    "class": "furhatos.gestures.Gesture"
}

CUSTOM_SMILE_GESTURE = {
    "name": "WrapUp",
    "frames": [
        {
            "time": [0.2, 1.0],
            "persist": False,
            "params": {
                "SMILE_OPEN": 0.5,
                "SMILE_CLOSED": 0.6,
                "BROW_UP_LEFT": 0.6,
                "BROW_UP_RIGHT": 0.6
            }
        },
        {
            "time": [0.4],
            "persist": False,
            "params": {"NECK_TILT": 8}
        },
        {
            "time": [0.7],
            "persist": False,
            "params": {"NECK_TILT": 3}
        },
        {
            "time": [1.3],
            "persist": False,
            "params": {"reset": True}
        }
    ],
    "class": "furhatos.gestures.Gesture"
}

@dataclass
class GestureConfig:
    """Configuration for Furhat gesture - can be a name string or custom dict."""
    gesture_name: str = None
    gesture_body: Dict[str, Any] = None
    blocking: bool = False

class GestureMapper:
    """Maps (emotion, confidence_score) to appropriate Furhat gesture."""

    def map_gesture(self, emotion: Emotion, confidence_score: Optional[float] = None) -> GestureConfig:
        """
        Main mapping function.
        Args:
            emotion: The detected emotion
            confidence_score: Optional confidence score (0.0 to 1.0)
        Returns:
            GestureConfig with custom gesture body
        """
        if emotion == Emotion.NERVOUS:
            # Consistent with test_gestures_console.py which uses BrowRaise for nervous
            return GestureConfig(gesture_name="BrowRaise", blocking=False)
        elif emotion == Emotion.CONFIDENT:
            return GestureConfig(gesture_body=CONFIDENT_BEHAVIOR, blocking=False)
        elif emotion == Emotion.DEFENSIVE:
            return GestureConfig(gesture_body=DEFENSIVE_BEHAVIOR, blocking=False)
        elif emotion == Emotion.NEUTRAL:
            return GestureConfig(gesture_body=NEUTRAL_BEHAVIOR, blocking=False)

        return GestureConfig(gesture_body=NEUTRAL_BEHAVIOR, blocking=False)

# LOAD TRAINED DINO+SVM MODEL (FACIAL REGIONS VERSION)
def load_custom_emotion_model():
    """Load the trained DINO+SVM model with facial regions from saved files"""
    model_dir = "model_configuration"

    try:
        # Load SVM model (facial regions version)
        svm_path = os.path.join(model_dir, 'best_svm_model_facial.pkl')
        svm_model = joblib.load(svm_path)
        print(f"Loaded Facial Regions SVM model from {svm_path}")

        # Load DINO model name (facial regions version)
        config_path = os.path.join(model_dir, 'dino_model_config_facial.txt')
        with open(config_path, 'r') as f:
            dino_model_name = f.read().strip()

        # Load class names (facial regions version)
        classes_path = os.path.join(model_dir, 'class_names_facial.txt')
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        # Load DINO model and processor
        processor = AutoImageProcessor.from_pretrained(dino_model_name)
        dino_model = AutoModel.from_pretrained(dino_model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dino_model.to(device)
        dino_model.eval()

        print(f"Using DINO model: {dino_model_name}")
        print(f"Classes: {class_names}")

        return svm_model, dino_model, processor, device, class_names
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found - {e}")
        print(f"Required files in '{model_dir}/':")
        print("  - best_svm_model_facial.pkl")
        print("  - dino_model_config_facial.txt")
        print("  - class_names_facial.txt")
        return None, None, None, None, None
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None, None, None, None, None


def extract_facial_region_features(outputs, device):
    """
    Extract features from 8 specific facial regions using DINO patch tokens.

    Regions: forehead, eyes, nose, mouth, left_cheek, right_cheek, chin, global

    Returns: numpy array of shape (8 * dim,) = (3072,) for DINOv3-ViT-S
    """
    # Get patch tokens - skip first 5 tokens (CLS + 4 register tokens for DINOv3)
    patch_tokens = outputs.last_hidden_state[0, 5:, :]  # (196, dim)
    grid_size = 14
    dim = patch_tokens.shape[-1]
    patches_grid = patch_tokens.view(grid_size, grid_size, dim)  # (14, 14, dim)

    # Define facial regions (assuming centered face from RetinaFace crop)
    forehead = patches_grid[0:3, :, :].mean(dim=(0, 1))         # Top rows
    eyes = patches_grid[3:6, :, :].mean(dim=(0, 1))             # Eye region
    nose = patches_grid[6:9, 4:10, :].mean(dim=(0, 1))          # Center
    mouth = patches_grid[9:12, 3:11, :].mean(dim=(0, 1))        # Lower center
    left_cheek = patches_grid[5:10, 0:4, :].mean(dim=(0, 1))    # Left side
    right_cheek = patches_grid[5:10, 10:14, :].mean(dim=(0, 1)) # Right side
    chin = patches_grid[12:14, 4:10, :].mean(dim=(0, 1))        # Bottom
    global_feat = patches_grid.mean(dim=(0, 1))                  # Full face

    # Concatenate: 8 regions * dim
    combined = torch.cat([forehead, eyes, nose, mouth, left_cheek, right_cheek, chin, global_feat], dim=0)

    return combined.cpu().numpy()

# CV2 LIVE EMOTION DETECTION (DINO+SVM with FACIAL REGIONS)
def cv2_vid_with_emotion():
    """
    Continuously captures video and detects emotions using DINO+SVM with facial region features.
    Uses 8 facial regions (forehead, eyes, nose, mouth, cheeks, chin, global) for better accuracy.
    Predictions are added to the global emotion_aggregator.
    """
    # Load trained model (facial regions version)
    svm_model, dino_model, processor, device, class_names = load_custom_emotion_model()

    if svm_model is None:
        print("FATAL: Cannot start emotion detection without model")
        return

    # Initialize face detector (only for face detection, not emotion)
    face_model_path = "./weights/Alignment_RetinaFace.pth"
    face_detector = FaceDetector(model_path=face_model_path, device="cpu")

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("ERROR: Failed to open camera")
        return

    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Give camera time to stabilize
    import time
    time.sleep(1.0)

    frame_count = 0
    PROCESS_EVERY = 3  # Process every 3rd frame for efficiency

    print("Starting emotion detection with DINO+SVM (FACIAL REGIONS)")

    while True:
        ok, frame = cam.read()
        if not ok:
            print(f"ERROR: Failed to read frame at frame {frame_count}")
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY == 0:
            try:
                # Detect face using OpenFace
                cropped_face, dets = face_detector.get_face(frame)

                if cropped_face is not None and dets is not None and len(dets) > 0:
                    # Convert cropped face to PIL Image for DINO
                    face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(face_rgb)

                    # Extract DINO FACIAL REGION features (8 regions)
                    with torch.inference_mode():
                        inputs = processor(images=pil_image, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = dino_model(**inputs)
                        # Use facial region extraction instead of pooler_output
                        features = extract_facial_region_features(outputs, device)
                        # print(f"[FRAME {frame_count}] Extracted facial region features: shape={features.shape}")

                    # Get probability distribution first
                    if hasattr(svm_model, 'predict_proba'):
                        probs = svm_model.predict_proba([features])[0]
                    else:
                        # Create one-hot if probabilities not available
                        pred_idx = svm_model.predict([features])[0]
                        probs = np.zeros(len(class_names))
                        probs[pred_idx] = 1.0

                    # Get predicted emotion (raw, no penalty - handled in aggregation)
                    pred_idx = np.argmax(probs)

                    #print(f"[FRAME {frame_count}] Raw probs: {dict(zip(class_names, [f'{p:.3f}' for p in probs]))} → {class_names[pred_idx]}")

                    # Add RAW prediction to aggregator (penalty applied in aggregation only)
                    emotion_aggregator.add_prediction(
                        emotion=class_names[pred_idx],
                        confidence=float(probs[pred_idx]),
                        probs=probs
                    )

            except Exception as e:
                print(f"[FRAME {frame_count}] Error: {e}")  

    cam.release()
    print("Perception thread stopped.")


# REAL-TIME GESTURE LOOP (runs during listening)
def realtime_gesture_loop(
    furhat,
    poll_interval=0.2,
    sustain_map=None
):
    # Default per-emotion sustain thresholds (seconds)
    if sustain_map is None:
        sustain_map = {
            "Confident": 6.0,
            "Nervous": 3.0,
            "Neutral": 7.0,
            "Defensive": 5.0,
        }

    def _safe_call(fn):
        try:
            fn()
        except Exception:
            pass

    def do_body(body):
        _safe_call(lambda: furhat.gesture(body=body))

    def do_name(name, blocking=False):
        _safe_call(lambda: furhat.gesture(name=name, blocking=blocking))

    def pause(seconds):
        time.sleep(max(0.0, seconds))

    def small_smile():
        # Prefer built-in 'Smile'. If unavailable, fallback to a tiny closed-lip smile.
        try:
            furhat.gesture(name="Smile", blocking=False)
            return
        except Exception:
            pass
        
        tiny_smile = {
            "name": "TinySmile_fallback",
            "frames": [
                {"time": [0.10, 0.45], "persist": False, "params": {"SMILE_CLOSED": 0.25}},
                {"time": [0.65], "persist": False, "params": {"reset": True}},
            ],
            "class": "furhatos.gestures.Gesture",
        }
        do_body(tiny_smile)

    def brow_raise_prompting():
        do_name("BrowRaise", blocking=False)

    def micro_nod(times=3, amplitude=3, interval=0.5):
        body = {
            "name": "MicroNod",
            "frames": [
                {"time": [0.28], "persist": False, "params": {"NECK_TILT": amplitude}},
                {"time": [0.58], "persist": False, "params": {"NECK_TILT": 0}},
                {"time": [1.88], "persist": False, "params": {"reset": True}},
            ],
            "class": "furhatos.gestures.Gesture",
        }
        for i in range(times):
            do_body(body)
            pause(interval)

    def do_confident():
        # Confident custom + small smile + micro nod (3x, slightly faster)
        do_body(CONFIDENT_BEHAVIOR)
        pause(0.50)
        small_smile()
        pause(0.50)
        micro_nod(times=3, amplitude=3, interval=0.14)

    def do_nervous():
        # BrowRaise + small smile + micro nod (slow, 2x)
        brow_raise_prompting()
        pause(0.50)
        small_smile()
        pause(0.50)
        micro_nod(times=2, amplitude=2)

    def do_neutral():
        # Neutral custom + small smile + micro nod (small, 2x)
        do_body(NEUTRAL_BEHAVIOR)
        pause(1.50)
        # small_smile()
        micro_nod(times=2, amplitude=5)

    def do_defensive():
        # Defensive custom + small smile + micro nod (2x)
        do_body(DEFENSIVE_BEHAVIOR)
        pause(0.12)
        small_smile()
        pause(0.10)
        micro_nod(times=2, amplitude=2, interval=0.18)

    # Emotion label to gesture function mapping
    emotion_to_action = {
        "Confident": do_confident,
        "Nervous": do_nervous,
        "Neutral": do_neutral,
        "Defensive": do_defensive,
    }

    # State variables
    last_emotion = None
    emotion_start_t = None
    next_fire_t = None

    def get_sustain_seconds(emo: str) -> float:
        # Sustain duration per emotion
        return float(sustain_map.get(emo, 5.0))

    while True:
        time.sleep(poll_interval)

        # Only active while listening
        if not IS_LISTENING.is_set():
            continue

        aggregated = emotion_aggregator.get_aggregated_emotion()
        emo = aggregated.get("emotion", "Neutral")

        now = time.time()
        sustain_seconds = get_sustain_seconds(emo)

        # Reset timing on emotion change
        if emo != last_emotion:
            last_emotion = emo
            emotion_start_t = now
            next_fire_t = now + sustain_seconds
            continue

        # Initialize timing if missing
        if emotion_start_t is None or next_fire_t is None:
            emotion_start_t = now
            next_fire_t = now + sustain_seconds
            continue

        # Trigger gesture after sustained duration
        if now >= next_fire_t:
            action = emotion_to_action.get(emo)
            if action is not None:
                action()
            next_fire_t = now + get_sustain_seconds(emo)



# INTERVIEW COACH CLASS
class InterviewCoach:
    """
    Flow:
    1. Introduction - Greet and get user's name
    2. Warmup - Ask 1 easy question to establish baseline
    3. Main Interview - Ask 3-4 questions, adapting to emotion
    4. Closing - Provide emotional summary and encouragement
    """

    def __init__(self, furhat_ip="localhost", api_key=None):
        """
        Initialize the system.
        """
        self.furhat = FurhatRemoteAPI(furhat_ip)

        # Core state tracking
        self.user_name = ""
        self.questions_asked = 0
        # Current detected emotion (updated each turn)
        self.current_emotion = Emotion.NEUTRAL
        # History of all detected emotions
        self.emotion_history = []

        # Delivery quality tracking
        self.speech_quality_history = []
        self.delivery_metrics_history = []

        # Current phase in the interview (intro → warmup → main → closing)
        self.current_phase = "introduction"

        # Question bank (simple list structure)
        self.questions = {
            "easy": [
                "Tell me a bit about yourself and your background.",
                "What interests you about this role?",
                "Why did you choose this career path?",
                "What do you enjoy most about your work?",
                "Tell me about your current or most recent position."
            ],
            "medium": [
                "Describe a challenging project you've worked on.",
                "Tell me about a time you worked in a team.",
                "How do you handle stress and tight deadlines?",
                "Describe a situation where you had to learn something new quickly.",
                "Tell me about a time you took initiative on a project."
            ],
            "hard": [
                "Describe a time when you failed and what you learned.",
                "How would you handle a conflict with a colleague?",
                "What's your biggest weakness and how are you addressing it?",
                "Tell me about a time you had to make a difficult decision.",
                "Describe a situation where you disagreed with your manager."
            ]
        }
        # Track which questions we've already asked (avoid repeating)
        self.used_questions = []

        # Count how many times LLM has failed (for fallback logic)
        self.llm_failures = 0
        self.max_llm_failures = 2
        # Count emotion detection failures
        self.emotion_detection_failures = 0

        # Gesture mapper for emotion-adaptive behaviors
        self.gesture_mapper = GestureMapper()

        # LLM setup 
        self.llm_api_key = api_key
        self.client = genai.Client(api_key=api_key) if api_key else None

        # Store recent conversation turns
        self.conversation_history = []

        # Interview logging for evaluation
        self.turn_logs = []  # List of turn data for TXT export
        self.interviewer_name = ""  # Set at start

        # Chat instance for Gemini with safety settings
        self.chat = None
        if self.client:
            from google.genai import types
            self.chat = self.client.chats.create(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are Furhat, an empathetic interview practice coach. "
                        "Provide brief, constructive feedback (max 40 words, 2-3 sentences). "
                        "Always be encouraging and supportive. "
                        "Adapt your tone based on the user's emotional state."
                    ),
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_MEDIUM_AND_ABOVE"
                        )
                    ]
                )
            )

    # ENHANCED LISTENING WITH HESITATION DETECTION
    def is_hesitation_only(self, text):
        """
        Detect if the response contains only filler words/hesitation words, indicating user is still thinking.

        Args:
            text: User's response text

        Returns:
            bool: True if only hesitation words, False if contains substantive content
        """
        if not text or text == "No response":
            return False

        # Common filler/hesitation words
        hesitation_words = [
            'um', 'uh', 'umm', 'uhh', 'ummm', 'uhhh',
            'hmm', 'hmmm', 'err', 'errr', 'ah', 'ahh',
            'well', 'so', 'like', 'you know',
            'and', 'but', 'or', 'the', 'a',
        ]

        # Clean text: convert to lowercase, remove punctuation
        clean_text = text.lower().strip()
        for punct in ['.', ',', '!', '?', '...']:
            clean_text = clean_text.replace(punct, ' ')

        words = clean_text.split()

        # If no words, count as hesitation
        if not words:
            return True

        # If all words are hesitation words, or only 1-2 very short words
        non_hesitation_words = [w for w in words if w not in hesitation_words and len(w) > 2]

        # Only hesitation words, or only a few short words (possibly incomplete thought)
        if len(non_hesitation_words) == 0:
            return True

        # If total word count is low (<=3) and mostly hesitation words
        if len(words) <= 3 and len(non_hesitation_words) <= 1:
            return True

        return False

    def listen_with_hesitation_detection(self, context="normal", max_attempts=3):
        """
        Smart listening: automatically continue listening if user only said filler words.
        Handles Furhat API's speech timeout by detecting filler-only segments and
        continuing to listen until substantive content is captured.

        Args:
            context: Listening context ("introduction", "normal", "thinking")
            max_attempts: Maximum number of attempts to prevent infinite loop

        Returns:
            Complete user answer (concatenated from all segments)
        """
        full_answer = []
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Use listen_blocking to signal listening state for real-time gestures
            response = listen_blocking(self.furhat)

            if not response or not response.message:
                # No response
                if attempts == 1:
                    # No response on first attempt, return empty
                    return "No response"
                else:
                    # Had response before, now none, finish
                    break

            current_text = response.message.strip()

            # Detect if only hesitation words
            if self.is_hesitation_only(current_text):
                print(f"[HESITATION DETECTED] User said: '{current_text}' - continuing to listen...")
                full_answer.append(current_text)

                # Give user visual feedback: nod to show we're listening
                self.furhat.gesture(name="Nod")

                # Continue listening (don't speak, just wait)
                continue
            else:
                # Has substantive content, add and finish
                full_answer.append(current_text)
                break

        # Concatenate all segments
        if full_answer:
            complete_answer = ' '.join(full_answer)
            print(f"[COMPLETE ANSWER] Collected {len(full_answer)} segments: '{complete_answer}'")
            return complete_answer
        else:
            return "No response"

    # EMOTION DETECTION
    def detect_emotion(self):
        """
        Get emotion from global aggregator (already 4 interview emotions from DINO+SVM).
        Returns: (primary_emotion, confidence, top_3_emotions)
        """
        try:
            # Get aggregated emotion from the background perception system
            # DINO+SVM already outputs interview emotions (Confident, Nervous, Neutral, Defensive)
            aggregated_result = emotion_aggregator.get_aggregated_emotion()

            # Convert string emotion to Enum
            emotion_str = aggregated_result["emotion"]
            emotion_map = {
                "Confident": Emotion.CONFIDENT,
                "Nervous": Emotion.NERVOUS,
                "Neutral": Emotion.NEUTRAL,
                "Defensive": Emotion.DEFENSIVE
            }
            emotion = emotion_map.get(emotion_str, Emotion.NEUTRAL)
            confidence = aggregated_result["confidence"]

            # Get top-3 emotions for context (pass aggregated_result to avoid duplicate computation)
            top_3_raw = emotion_aggregator.get_top_k_emotions(k=3, aggregated=aggregated_result)

            # Convert to Enum tuples
            top_3_interview = []
            for emotion_str, score in top_3_raw:
                mapped_emotion = emotion_map.get(emotion_str, Emotion.NEUTRAL)
                top_3_interview.append((mapped_emotion, score))

            self.emotion_detection_failures = 0  # Reset on success
            return emotion, confidence, top_3_interview

        except Exception as e:
            print(f"Emotion detection failed: {e}")
            self.emotion_detection_failures += 1

            # Fallback: Use last known emotion or neutral
            if self.emotion_history:
                fallback_emotion = self.emotion_history[-1]
                print(f"Using last known emotion: {fallback_emotion.value}")
                return fallback_emotion, 0.5, [(fallback_emotion, 0.5)]
            else:
                print(f"Using neutral as fallback")
                return Emotion.NEUTRAL, 0.5, [(Emotion.NEUTRAL, 0.5)]

    # SPEECH QUALITY ANALYSIS
    def analyze_filler_words(self, user_answer):
        """
        Analyze filler word ratio in the response.
        Returns: (filler_count, filler_ratio)
        """
        if not user_answer or user_answer == "No response":
            return 0, 0.0

        # Common filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'well', 'so',
                        'basically', 'actually', 'kind of', 'sort of',
                        'i mean', 'hmm', 'err']

        words = user_answer.lower().split()
        word_count = len(words)

        if word_count == 0:
            return 0, 0.0

        # Count filler words
        filler_count = sum(1 for word in words if word.strip('.,!?') in filler_words)

        # Calculate ratio
        filler_ratio = filler_count / word_count

        return filler_count, filler_ratio

    def calculate_speaking_rate(self, user_answer, duration):
        """
        Calculate speaking rate (words per minute).
        Returns: wpm (words per minute)
        """
        if not user_answer or duration <= 0:
            return 0.0

        word_count = len(user_answer.split())
        wpm = (word_count / duration) * 60

        return wpm

    def evaluate_delivery_quality(self, user_answer, duration):
        """
        Evaluate speech delivery quality based on filler words and speaking rate.
        Returns: (SpeechQuality, metrics_dict)
        """
        if not user_answer or user_answer == "No response":
            return SpeechQuality.CLEAR, {
                'filler_count': 0,
                'filler_ratio': 0.0,
                'wpm': 0.0,
                'word_count': 0
            }

        # Analyze filler words
        filler_count, filler_ratio = self.analyze_filler_words(user_answer)

        # Calculate speaking rate
        wpm = self.calculate_speaking_rate(user_answer, duration) if duration > 0 else 0.0

        # Get word count
        word_count = len(user_answer.split())

        # MUCH STRICTER speech quality thresholds
        # Make it harder to get FLUENT or CLEAR ratings

        HIGH_FILLER = 0.08      # ~1 per 12 words - now considered hesitant (was 0.15)
        LOW_FILLER = 0.03       # ~1 per 33 words - fluent threshold (was 0.08)
        VERY_SLOW = 100         # Below this is struggling
        VERY_FAST = 180         # Above this is rambling (was 200)
        IDEAL_MIN = 130         # Narrower ideal range (was 120)
        IDEAL_MAX = 160         # Narrower ideal range (was 170)

        # HESITANT: Much more sensitive to delivery issues
        # - Any filler words above 8% (1 per 12 words)
        # - Fast speech above 180 WPM
        # - Slow speech below 100 WPM
        # - Any combination of moderate fillers + off-pace
        if (filler_ratio > HIGH_FILLER or
            (wpm > 0 and wpm > VERY_FAST) or
            (wpm > 0 and wpm < VERY_SLOW) or
            (wpm > 0 and filler_ratio > 0.05 and not (IDEAL_MIN <= wpm <= IDEAL_MAX))):
            speech_quality = SpeechQuality.HESITANT

        # FLUENT: Very strict requirements for excellent delivery
        # - Must have <3% filler ratio (almost none)
        # - Must be in perfect pace range (130-160 WPM)
        # - Must have at least 10 words (substantial answer)
        elif (filler_ratio < LOW_FILLER and
              (IDEAL_MIN <= wpm <= IDEAL_MAX) and
              word_count >= 10):
            speech_quality = SpeechQuality.FLUENT

        # CLEAR: Middle ground (now much narrower)
        # - Low-moderate fillers (3-8%)
        # - Acceptable pace (100-180 WPM)
        elif (filler_ratio <= HIGH_FILLER and
              wpm > 0 and
              (VERY_SLOW <= wpm <= VERY_FAST)):
            speech_quality = SpeechQuality.CLEAR

        # Default to HESITANT if nothing matches
        else:
            speech_quality = SpeechQuality.HESITANT

        metrics = {
            'filler_count': filler_count,
            'filler_ratio': filler_ratio,
            'wpm': wpm,
            'word_count': word_count
        }

        return speech_quality, metrics
    
    # SAFETY & VALIDATION
    def pre_filter_user_answer(self, user_answer):
        """
        Pre-filter user input to block jailbreak attempts and prompt injection.
        Returns: (is_safe: bool, warning_message: str or None)
        """
        if not user_answer or user_answer == "No response":
            return True, None

        # Detect jailbreak/prompt injection attempts
        banned_phrases = [
            "ignore previous", "ignore all", "disregard",
            "system prompt", "override", "bypass",
            "act as", "pretend to be", "you are now",
            "forget everything", "new instructions",
            "developer mode", "admin mode", "jailbreak"
        ]

        lower_answer = user_answer.lower()
        for phrase in banned_phrases:
            if phrase in lower_answer:
                return False, "I'm here only to help with interview practice. Let's try again."

        return True, None

    # LLM INTEGRATION
    def get_llm_feedback(self, user_answer, emotion, question_asked, speech_quality=None, top_3_emotions=None, metrics=None):
        """Generate feedback using Gemini API with fallback to rule-based responses."""
        if not self.chat:
            return self._get_fallback_feedback(emotion)

        try:
            # Tone adaptation based on primary emotion
            tone_adaptations = {
                Emotion.NERVOUS: "Be warm and reassuring. Normalize their feelings.",
                Emotion.CONFIDENT: "Be professional and direct.",
                Emotion.NEUTRAL: "Be balanced and constructive.",
                Emotion.DEFENSIVE: "Be gentle and validating. Avoid critical language."
            }

            # Build emotion distribution (for LLM to understand prominence)
            if top_3_emotions:
                emotion_distribution = ", ".join([f"{e.value}: {s:.0%}" for e, s in top_3_emotions])
            else:
                emotion_distribution = f"{emotion.value}: 100%"

            # Extract speech metrics for detailed feedback
            speech_quality_name = speech_quality.value if speech_quality else "clear"
            filler_ratio = metrics.get('filler_ratio', 0) if metrics else 0
            wpm = metrics.get('wpm', 0) if metrics else 0

            # Build complete prompt
            prompt = f"""You are an interview coach giving brief feedback.

CONTEXT:
Question: "{question_asked}"
Answer: "{user_answer}"

OBSERVATIONS:
- Emotions detected: {emotion_distribution}
- Primary emotion: {emotion.value}

SPEECH METRICS:
- Quality: {speech_quality_name}
- Filler words: {filler_ratio:.0%} (words like "um", "uh", "like" - ideal is below 3%, above 8% is too many)
- Pace: {wpm:.0f} WPM (ideal: 130-160 WPM, above 180 is too fast, below 100 is too slow)

TONE TO USE: {tone_adaptations[emotion]}

FEEDBACK STRUCTURE:
1. Comment on emotions observed (e.g., "I noticed some nervousness" or "You seemed confident")
2. Comment on speech delivery based on the metrics above:
   - If fluent: briefly praise smooth delivery
   - If hesitant due to high fillers: suggest reducing filler words
   - If hesitant due to fast pace: suggest slowing down
   - If hesitant due to slow pace: suggest more confident pacing
   - If clear: brief positive note or skip
3. Brief comment on answer content
4. Suggestion for content improvement, OR if confident, encourage them to continue

RULES:
- Do NOT mention percentages or numbers in your feedback
- VARY your phrasing - start with different words each time and use different expressions
- Be specific and actionable
- Be supportive
- Keep it short

Provide only the feedback text."""

            # Call LLM
            response = self.chat.send_message(prompt)
            feedback = response.text.strip()

            # Validate response
            if not self._validate_llm_response(feedback):
                print("LLM response validation failed, using fallback")
                return self._get_fallback_feedback(emotion)


            self.llm_failures = 0  # Reset on success
            return feedback

        except Exception as e:
            print(f"LLM Error: {e}")
            self.llm_failures += 1
            return self._get_fallback_feedback(emotion)


    def _validate_llm_response(self, response):
        """Validate LLM response is appropriate"""
        if len(response.split()) > 50:
            print("LLM response too long")
            return False

        inappropriate_words = ["stupid", "wrong", "bad", "terrible", "awful", "fail"]
        if any(word in response.lower() for word in inappropriate_words):
            print("LLM response contains inappropriate language")
            return False

        if len(response.strip()) < 10:
            print("LLM response too short")
            return False

        return True

    def _get_fallback_feedback(self, emotion):
        """Rule-based fallback when LLM fails"""
        fallback_responses = {
            Emotion.NERVOUS: "That's a good start. Take your time and remember, you're doing well.",
            Emotion.CONFIDENT: "Good answer. Let's continue.",
            Emotion.NEUTRAL: "That's helpful. Thank you for sharing that.",
            Emotion.DEFENSIVE: "I appreciate your perspective. Let's move forward."
        }
        return fallback_responses.get(emotion, "Thank you for your answer.")

    # BEHAVIOR ADAPTATION
    def adjust_behavior_for_emotion(self, emotion, confidence_score=None):
        """Dynamically adjust Furhat's behavior based on detected emotion using GestureMapper."""
        gesture_config = self.gesture_mapper.map_gesture(emotion, confidence_score)
        if gesture_config.gesture_body:
            self.furhat.gesture(body=gesture_config.gesture_body)
        elif gesture_config.gesture_name:
            self.furhat.gesture(name=gesture_config.gesture_name, blocking=gesture_config.blocking)

    # QUESTION SELECTION 
    def calculate_performance_score(self):
        """
        Calculate performance score from the last answer to determine appropriate difficulty.
        Uses a simple scale centered around 0:
        - Positive = doing well → harder questions
        - Negative = needs support → easier questions

        Returns: score from -2 to +2
        """
        # First question: always start easy
        if not self.emotion_history or not self.speech_quality_history:
            return -1  # Start with easy

        # Get the most recent answer only
        last_emotion = self.emotion_history[-1]
        last_speech = self.speech_quality_history[-1]

        # Emotion scores: Confident=+1, Neutral=0, Defensive=-0.5, Nervous=-1
        emotion_scores = {
            Emotion.CONFIDENT: 1,
            Emotion.NEUTRAL: 0,
            Emotion.DEFENSIVE: -0.5,
            Emotion.NERVOUS: -1
        }

        # Speech scores: Fluent=+1, Clear=0, Hesitant=-1
        speech_scores = {
            SpeechQuality.FLUENT: 1,
            SpeechQuality.CLEAR: 0,
            SpeechQuality.HESITANT: -1
        }

        emotion_score = emotion_scores.get(last_emotion, 0)
        speech_score = speech_scores.get(last_speech, 0)

        # Combined score (range: -2 to +2)
        return emotion_score + speech_score

    def select_question_difficulty(self):
        """
        Select difficulty based on the last answer to be responsive to current state.

        Score range: -2 to +2
        - Negative = struggling → easy questions
        - Around 0 = baseline → medium questions
        - Positive = doing well → hard questions
        """
        # Calculate performance from recent answers
        performance = self.calculate_performance_score()

        # Map performance to difficulty
        if performance < -0.5:
            difficulty = "easy"
        elif performance <= 0.5:
            difficulty = "medium"
        else:
            difficulty = "hard"

        print(f"Performance score: {performance:.1f} → {difficulty} question")
        return difficulty

    def get_next_question(self, difficulty):
        """
        Get next question from selected difficulty level.
        Uses simple rotation to avoid repetition.
        """
        # Get available questions for this difficulty
        available = [q for q in self.questions[difficulty] if q not in self.used_questions]

        # If no questions available at this level, try adjacent levels
        if not available:
            if difficulty == "medium":
                # Try easy first, then hard
                available = [q for q in self.questions["easy"] if q not in self.used_questions]
                if not available:
                    available = [q for q in self.questions["hard"] if q not in self.used_questions]
            elif difficulty == "easy":
                # Try medium
                available = [q for q in self.questions["medium"] if q not in self.used_questions]
            else:  # hard
                # Try medium
                available = [q for q in self.questions["medium"] if q not in self.used_questions]

        # If still no questions, reset used questions
        if not available:
            self.used_questions.clear()
            available = self.questions[difficulty]

        # Pick first available question (simple rotation)
        question = available[0] if available else "Tell me more about your experience."
        self.used_questions.append(question)

        return question

    # PHASES
    def introduction(self):
        """Phase 1: Introduction"""
        print("\nPHASE: INTRODUCTION")
        self.current_phase = "introduction"

        # Attend to user before starting
        self.furhat.attend(user="CLOSEST")
        #self.furhat.gesture(name="BigSmile")
        self.furhat.gesture(body=CUSTOM_SMILE_GESTURE) 
        self.furhat.say(
            text="Hello! I'm Furhat, and I'm here to help you practice for your upcoming interviews.",
            blocking=True
        )
        time.sleep(0.5)

        self.furhat.gesture(name="Smile")
        self.furhat.say(
            text="I will ask you a few questions. There are no right or wrong answers.",
            blocking=True
        )
        time.sleep(0.5)
        
        self.furhat.gesture(name="BrowRaise")
        msg = "What's your name?"
        print(f"Furhat: {msg}")
        self.furhat.say(text=msg, blocking=True)
        # Listen for user name (with retry)
        response = listen_blocking(self.furhat)
        if response and response.message:
            print(f"User: {response.message}")
            self.user_name = response.message.strip().split()[0].capitalize()
            self.furhat.gesture(body=ATTENTIVE_LISTEN)
            time.sleep(0.2)
            self.furhat.gesture(name="BigSmile")
            msg = f"Nice to meet you, {self.user_name}!"
            print(f"Furhat: {msg}")
            self.furhat.say(text=msg, blocking=True)
        else:
            # FALLBACK: Didn't hear name - try once more
            self.furhat.gesture(name="BrowFrown")
            msg = "Sorry, I didn't catch that. Could you say your name again?"
            print(f"Furhat: {msg}")
            self.furhat.say(text=msg, blocking=True)
            response = listen_blocking(self.furhat)

            if response and response.message:
                print(f"User: {response.message}")
                self.user_name = response.message.strip().split()[0].capitalize()
                self.furhat.gesture(body=ATTENTIVE_LISTEN)
                time.sleep(0.2)
                self.furhat.gesture(name="BigSmile")
                msg = f"Nice to meet you, {self.user_name}!"
                print(f"Furhat: {msg}")
                self.furhat.say(text=msg, blocking=True)
            else:
                # FALLBACK: Still didn't hear - move on
                self.user_name = "there"
                self.furhat.gesture(name="Smile")
                msg = "No worries! Let's continue."
                print(f"Furhat: {msg}")
                self.furhat.say(text=msg, blocking=True)

        time.sleep(0.3)

        # Ready check
        self.furhat.gesture(name="BrowRaise")
        msg = "Are you ready to begin?"
        print(f"Furhat: {msg}")
        self.furhat.say(text=msg, blocking=True)

        # Listen for confirmation
        response = listen_blocking(self.furhat)

        if response and response.message:
            print(f"User: {response.message}")
            self.furhat.gesture(body=ATTENTIVE_LISTEN)
            answer = response.message.lower()

            # Check for "no" responses - STOP THE INTERVIEW
            if any(word in answer for word in ["no", "not", "wait", "stop", "cancel", "quit"]):
                self.furhat.gesture(name="Smile")
                msg = "No problem. Come back whenever you're ready. Goodbye!"
                print(f"Furhat: {msg}")
                self.furhat.say(text=msg, blocking=True)
                return False  # Exit introduction, don't continue

        # User is ready - proceed
        self.furhat.gesture(body=ATTENTIVE_LISTEN)
        msg = "Great!"
        print(f"Furhat: {msg}")
        self.furhat.say(text=msg, blocking=True)
        return True  # Continue to warmup

    def warmup_phase(self):
        """Phase 2: Warmup"""
        print("\nPHASE: WARMUP")
        self.current_phase = "warmup"

        question = self.get_next_question("easy")
        self.questions_asked += 1

        self.furhat.attend(user="CLOSEST")
        self.furhat.gesture(name="Smile")
        msg = "Let's start with a simple question."
        print(f"Furhat: {msg}")
        self.furhat.say(text=msg, blocking=True)
        time.sleep(0.3)

        # Ask the question with raised brows
        self.furhat.gesture(name="BrowRaise")
        print(f"Furhat: {question}")
        self.furhat.say(text=question, blocking=True)

        # Clear emotion buffer to collect emotions during answer
        emotion_aggregator.clear()

        # Listen to answer with hesitation detection (emotions are collected during this time)
        start_time = time.time()
        user_answer = self.listen_with_hesitation_detection(context="normal", max_attempts=3)
        end_time = time.time()
        duration = end_time - start_time

        print(f"User: {user_answer}")

        # Handle case when Furhat doesn't catch the answer
        if user_answer == "No response":
            self.furhat.gesture(name="BrowFrown")
            msg = "I didn't catch that. Could you please try again?"
            print(f"Furhat: {msg}")
            self.furhat.say(text=msg, blocking=True)
            # Give them another chance with hesitation detection
            start_time = time.time()
            user_answer = self.listen_with_hesitation_detection(context="normal", max_attempts=3)
            end_time = time.time()
            duration = end_time - start_time
            print(f"User: {user_answer}")

            # If still no response, acknowledge and move on
            if user_answer == "No response":
                self.furhat.gesture(name="Smile")
                msg = "That's okay. Take your time. Let's continue."
                print(f"Furhat: {msg}")
                self.furhat.say(text=msg, blocking=True)

        # Pre-filter user answer for safety (jailbreak detection)
        is_safe, warning = self.pre_filter_user_answer(user_answer)
        if not is_safe:
            print(f"Safety warning: Blocked potentially harmful input")
            self.furhat.gesture(body=ATTENTIVE_LISTEN)
            print(f"Furhat: {warning}")
            self.furhat.say(text=warning, blocking=True)
            time.sleep(0.3)
            # Re-ask the question
            self.furhat.gesture(name="BrowRaise")
            print(f"Furhat: {question}")
            self.furhat.say(text=question, blocking=True)
            response = listen_blocking(self.furhat)
            user_answer = response.message if response and response.message else "No response"
            print(f"User: {user_answer}")
            # If still not safe, just move on
            is_safe, _ = self.pre_filter_user_answer(user_answer)
            if not is_safe:
                self.furhat.gesture(name="Smile")
                msg = "Let's move on."
                print(f"Furhat: {msg}")
                self.furhat.say(text=msg, blocking=True)
                return

        # Detect emotion from the answer
        emotion, confidence, top_3_emotions = self.detect_emotion()
        self.current_emotion = emotion
        self.emotion_history.append(emotion)

        # Evaluate delivery quality with fallback
        try:
            speech_quality, metrics = self.evaluate_delivery_quality(user_answer, duration)
        except Exception as e:
            print(f"Error evaluating delivery quality: {e}")
            # Fallback to default values
            speech_quality = SpeechQuality.CLEAR
            metrics = {'filler_count': 0, 'filler_ratio': 0.0, 'wpm': 0.0, 'word_count': 0}

        self.speech_quality_history.append(speech_quality)
        self.delivery_metrics_history.append(metrics)

        print("USER'S RESPONSE ANALYSIS")
        print(f"Detected emotion: {emotion.value} (confidence: {confidence:.2f})")
        print(f"Top-3 emotions: {[(e.value, f'{s:.2f}') for e, s in top_3_emotions]}")
        print(f"Metrics: {metrics['word_count']} words, {metrics['filler_ratio']:.2%} fillers, {metrics['wpm']:.0f} WPM")
        print(f"Speech quality: {speech_quality.value}")

        # Give feedback based on detected emotion and delivery quality
        feedback = self.get_llm_feedback(user_answer, emotion, question, speech_quality, top_3_emotions, metrics)

        # Return attention to user before feedback
        self.furhat.attend(user="CLOSEST")
        time.sleep(0.2)

        self.adjust_behavior_for_emotion(emotion)
        print(f"Furhat: {feedback}")
        self.furhat.say(text=feedback, blocking=True)

        # Get gesture name for logging
        gesture_name = {
            Emotion.NERVOUS: "Nervous_behavior",
            Emotion.CONFIDENT: "Confident_behavior",
            Emotion.DEFENSIVE: "Defensive_behavior",
            Emotion.NEUTRAL: "Neutral_behavior"
        }.get(emotion, "Neutral_behavior")

        # Log this turn
        self.log_turn(
            phase="warmup",
            question=question,
            answer=user_answer,
            emotion=emotion,
            confidence=confidence,
            speech_quality=speech_quality,
            metrics=metrics,
            performance_score=self.calculate_performance_score(),
            difficulty="easy",  # warmup is always easy
            gesture_name=gesture_name,
            feedback=feedback
        )

        # Update history
        self.conversation_history.append(f"Q: {question}")
        self.conversation_history.append(f"A: {user_answer[:50]}...")
        self.conversation_history.append(f"Emotion: {emotion.value}, Speech: {speech_quality.value}")

    def main_interview(self):
        """Phase 3: Main interview"""
        print("\nPHASE: MAIN INTERVIEW")
        self.current_phase = "main"

        target_main_questions = 3
        main_questions_asked = 0

        while main_questions_asked < target_main_questions:
            time.sleep(0.5)

            print(f"\nTurn {self.questions_asked + 1}")

            # Select question difficulty based on emotion HISTORY
            difficulty = self.select_question_difficulty()
            question = self.get_next_question(difficulty)
            self.questions_asked += 1
            main_questions_asked += 1

            # Intro phrase before question (varied for each turn)
            # Last question is when main_questions_asked == target_main_questions
            is_last_question = (main_questions_asked == target_main_questions)

            if not is_last_question:
                intros = [
                    "Here's the next one.",
                    "Moving on.",
                    "Alright, next question.",
                    "Here's another one.",
                    f"Okay {self.user_name}, next question.",
                    "Let's keep going."
                ]
            else:  # Last question
                intros = [
                    "One more question before we wrap up.",
                    "Let's finish with this one.",
                    f"Last one, {self.user_name}."
                ]

            intro = random.choice(intros)

            self.furhat.gesture(name="Smile")
            print(f"Furhat: {intro}")
            self.furhat.say(text=intro, blocking=True)
            time.sleep(0.3)
            self.furhat.gesture(name="BrowRaise")
            print(f"Furhat: {question}")
            self.furhat.say(text=question, blocking=True)

            # Clear emotion buffer to collect emotions during answer
            emotion_aggregator.clear()

            # Listen to answer with hesitation detection (emotions are collected during this time)
            start_time = time.time()
            user_answer = self.listen_with_hesitation_detection(context="normal", max_attempts=3)
            end_time = time.time()
            duration = end_time - start_time

            print(f"User: {user_answer}")

            # Handle case when Furhat doesn't catch the answer
            if user_answer == "No response":
                self.furhat.gesture(name="BrowFrown")
                msg = "I didn't catch that. Could you please try again?"
                print(f"Furhat: {msg}")
                self.furhat.say(text=msg, blocking=True)
                # Give them another chance with hesitation detection
                start_time = time.time()
                user_answer = self.listen_with_hesitation_detection(context="normal", max_attempts=3)
                end_time = time.time()
                duration = end_time - start_time
                print(f"User: {user_answer}")

                # If still no response, acknowledge and move on
                if user_answer == "No response":
                    self.furhat.gesture(name="Smile")
                    msg = "That's okay. Take your time. Let's continue."
                    print(f"Furhat: {msg}")
                    self.furhat.say(text=msg, blocking=True)

            # Pre-filter user answer for safety (jailbreak detection)
            is_safe, warning = self.pre_filter_user_answer(user_answer)
            if not is_safe:
                print(f"Safety warning: Blocked potentially harmful input")
                self.furhat.gesture(name="Nod")
                print(f"Furhat: {warning}")
                self.furhat.say(text=warning, blocking=True)
                time.sleep(0.3)
                # Re-ask the question
                self.furhat.gesture(name="BrowRaise")
                print(f"Furhat: {question}")
                self.furhat.say(text=question, blocking=True)
                user_answer = self.listen_with_hesitation_detection(context="normal", max_attempts=3)
                print(f"User: {user_answer}")
                # If still not safe, just move on
                is_safe, _ = self.pre_filter_user_answer(user_answer)
                if not is_safe:
                    self.furhat.gesture(name="Smile")
                    msg = "Let's move on."
                    print(f"Furhat: {msg}")
                    self.furhat.say(text=msg, blocking=True)
                    continue

            # Detect emotion from the answer
            emotion, confidence, top_3_emotions = self.detect_emotion()
            self.current_emotion = emotion
            self.emotion_history.append(emotion)

            # Evaluate delivery quality with fallback
            try:
                speech_quality, metrics = self.evaluate_delivery_quality(user_answer, duration)
            except Exception as e:
                print(f"Error evaluating delivery quality: {e}")
                # Fallback to default values
                speech_quality = SpeechQuality.CLEAR
                metrics = {'filler_count': 0, 'filler_ratio': 0.0, 'wpm': 0.0, 'word_count': 0}

            self.speech_quality_history.append(speech_quality)
            self.delivery_metrics_history.append(metrics)

            print(f"Detected emotion: {emotion.value} (confidence: {confidence:.2f})")
            print(f"Top-3 emotions: {[(e.value, f'{s:.2f}') for e, s in top_3_emotions]}")
            print(f"Speech quality: {speech_quality.value}")
            print(f"Metrics: {metrics['word_count']} words, {metrics['filler_ratio']:.2%} fillers, {metrics['wpm']:.0f} WPM")

            # Get LLM feedback based on the detected emotion and delivery quality
            feedback = self.get_llm_feedback(user_answer, emotion, question, speech_quality, top_3_emotions, metrics)

            # Return attention to user before feedback
            self.furhat.attend(user="CLOSEST")
            time.sleep(0.2)

            # Emotion-adaptive gesture before feedback
            self.adjust_behavior_for_emotion(emotion)
            print(f"Furhat: {feedback}")
            self.furhat.say(text=feedback, blocking=True)

            # Get gesture name for logging
            gesture_name = {
                Emotion.NERVOUS: "Nervous_behavior",
                Emotion.CONFIDENT: "Confident_behavior",
                Emotion.DEFENSIVE: "Defensive_behavior",
                Emotion.NEUTRAL: "Neutral_behavior"
            }.get(emotion, "Neutral_behavior")

            # Log this turn
            self.log_turn(
                phase="main",
                question=question,
                answer=user_answer,
                emotion=emotion,
                confidence=confidence,
                speech_quality=speech_quality,
                metrics=metrics,
                performance_score=self.calculate_performance_score(),
                difficulty=difficulty,
                gesture_name=gesture_name,
                feedback=feedback
            )

            # Update conversation history
            self.conversation_history.append(f"Q: {question}")
            self.conversation_history.append(f"A: {user_answer[:50]}...")
            self.conversation_history.append(f"Emotion: {emotion.value}, Speech: {speech_quality.value}")

    def closing_phase(self):
        """Phase 4: Closing"""
        print("\nPHASE: CLOSING")
        self.current_phase = "closing"

        self.furhat.attend(user="CLOSEST")
        self.furhat.gesture(name="BigSmile")
        msg = "That's all the questions I have for you today."
        print(f"Furhat: {msg}")
        self.furhat.say(text=msg, blocking=True)
        time.sleep(0.5)

        msg = "Let me give you some feedback."
        print(f"Furhat: {msg}")
        self.furhat.say(text=msg, blocking=True)
        time.sleep(0.3)

        # Get LLM-generated closing feedback
        feedback = self.get_closing_feedback()
        self.furhat.gesture(name="Smile")
        print(f"Furhat: {feedback}")
        self.furhat.say(text=feedback, blocking=True)
        time.sleep(0.5)

        msg = "Remember, practice makes progress!"
        print(f"Furhat: {msg}")
        self.furhat.say(text=msg, blocking=True)
        time.sleep(0.3)

        self.furhat.gesture(name="BigSmile")
        msg = f"Good luck, {self.user_name}!"
        print(f"Furhat: {msg}")
        self.furhat.say(text=msg, blocking=True)

    def get_closing_feedback(self):
        """Generate holistic closing feedback using LLM based on entire session."""
        fallback = "You did well today. Keep practicing and you'll continue to improve."

        if not self.chat:
            return fallback

        try:
            # Build emotion and speech trajectories
            emotion_trajectory = [e.value for e in self.emotion_history]
            speech_trajectory = [s.value for s in self.speech_quality_history]

            # Describe the emotional journey
            if len(emotion_trajectory) >= 2:
                first_emotion = emotion_trajectory[0]
                last_emotion = emotion_trajectory[-1]
                trajectory_note = f"Started {first_emotion}, ended {last_emotion}."
            else:
                trajectory_note = "Single response recorded."

            prompt = f"""SESSION SUMMARY FOR {self.user_name}:
- Questions answered: {len(self.emotion_history)}
- Emotion trajectory: {emotion_trajectory}
- Speech quality trajectory: {speech_trajectory}
- {trajectory_note}

TASK:
Give brief, encouraging closing feedback (max 50 words, 2-3 sentences).
- Comment on their emotional journey if notable (e.g., started nervous but grew confident)
- Mention speech delivery if there was a clear pattern (mostly hesitant or mostly fluent)
- End on a positive, encouraging note
- Be specific to their performance, not generic

Just provide the feedback text, nothing else."""

            response = self.chat.send_message(prompt)
            feedback = response.text.strip()
            return feedback

        except Exception as e:
            print(f"LLM Error in closing feedback: {e}")
            return fallback
    # INTERVIEW LOGGING
    def log_turn(self, phase, question, answer, emotion, confidence, speech_quality,
                 metrics, performance_score, difficulty, gesture_name, feedback):
        """Log a single turn for evaluation export."""
        turn_data = {
            "turn": len(self.turn_logs) + 1,
            "phase": phase,
            "question": question,
            "answer": answer,
            "emotion": emotion.value,
            "emotion_confidence": confidence,
            "speech_quality": speech_quality.value,
            "filler_ratio": metrics.get('filler_ratio', 0),
            "wpm": metrics.get('wpm', 0),
            "performance_score": performance_score,
            "difficulty": difficulty,
            "gesture": gesture_name,
            "feedback": feedback
        }
        self.turn_logs.append(turn_data)

    def save_interview_log(self):
        """Save interview log to TXT file."""
        if not self.interviewer_name:
            self.interviewer_name = "unknown"

        filename = f"interview_log_{self.interviewer_name}.txt"

        with open(filename, 'w') as f:
            # Header
            f.write("INTERVIEW LOG\n")
            f.write(f"Interviewer: {self.interviewer_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")

            # Each turn
            for turn in self.turn_logs:
                f.write(f"TURN {turn['turn']} ({turn['phase']})\n")
                f.write(f"Question: {turn['question']}\n")
                f.write(f"Answer: {turn['answer']}\n")
                f.write(f"Emotion: {turn['emotion']} (confidence: {turn['emotion_confidence']:.2f})\n")
                f.write(f"Speech: {turn['speech_quality']} (fillers: {turn['filler_ratio']:.0%}, wpm: {turn['wpm']:.0f})\n")
                f.write(f"Performance Score: {turn['performance_score']:.1f}\n")
                f.write(f"Difficulty: {turn['difficulty']}\n")
                f.write(f"Gesture: {turn['gesture']}\n")
                f.write(f"Feedback: {turn['feedback']}\n\n")

            # Summary
            f.write("SUMMARY\n")
            emotion_traj = [t['emotion'] for t in self.turn_logs]
            speech_traj = [t['speech_quality'] for t in self.turn_logs]
            diff_traj = [t['difficulty'] for t in self.turn_logs]
            gesture_traj = [t['gesture'] for t in self.turn_logs]

            f.write(f"Emotion trajectory:    {emotion_traj}\n")
            f.write(f"Speech trajectory:     {speech_traj}\n")
            f.write(f"Difficulty trajectory: {diff_traj}\n")
            f.write(f"Gesture trajectory:    {gesture_traj}\n\n")

            # Matches (gesture always matches by design, difficulty always matches by logic)
            total_turns = len(self.turn_logs)
            f.write(f"Gesture matches:    {total_turns}/{total_turns}\n")
            f.write(f"Difficulty matches: {total_turns}/{total_turns}\n")

        print(f"\nInterview log saved to: {filename}")

    # MAIN RUN METHOD
    def run(self):
        """Main flow"""
        try:
            # Phase 1: Introduction
            if not self.introduction():
                # User said no - stop here
                print("User declined. Interview stopped.")
                return

            # Phase 2: Warmup
            self.warmup_phase()

            # Phase 3: Main Interview
            self.main_interview()

            # Phase 4: Closing
            self.closing_phase()

            # Ask for interviewer name and save log
            print("\n" + "-" * 40)
            self.interviewer_name = input("Enter interviewer name for log file: ").strip()
            self.save_interview_log()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # TODO: FURHAT BEHAVIOR - Error apology speech
            self.furhat.say(text="I apologize for the technical difficulty.")

if __name__ == "__main__":

    API_KEY = os.environ.get("GEMINI_API_KEY")

    # Furhat configuration
    FURHAT_IP = "localhost"  # Furhat IP address

    # Start perception system in the background
    print("Starting emotion perception system...")
    threading.Thread(
        target=cv2_vid_with_emotion,
        daemon=True
    ).start()
    # Give camera time to initialize
    time.sleep(2)

    # Start Interview Coach system
    print("Starting Interview Coach...")
    print(f"Connecting to Furhat at {FURHAT_IP}...")
    coach = InterviewCoach(furhat_ip=FURHAT_IP, api_key=API_KEY)

    # Start real-time gesture loop in background
    print("Starting real-time gesture loop...")
    threading.Thread(
        target=realtime_gesture_loop,
        args=(coach.furhat,),
        daemon=True
    ).start()

    coach.run()
