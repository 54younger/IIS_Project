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

# The OpenFace detector (for face detection only, not emotion)
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
from enum import Enum
from collections import deque

class EmotionAggregator:
    """Aggregates emotion predictions over time using weighted smoothing."""

    def __init__(self, window_size=30, decay_rate=0.98):
        self.window_size = window_size   # Max number of predictions to keep
        self.decay_rate = decay_rate     # Exponential decay for older predictions (0.98 = gentler, gives more weight to sustained emotions)
        self.predictions = deque(maxlen=window_size)
        # DINO+SVM predicts 4 interview emotions directly
        self.emo_names = ["Confident", "Nervous", "Neutral", "Defensive"]
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

            # Apply same balancing as per-frame detection
            # Neutral was overrepresented in training (41%), so penalize it
            emotion_scores["Neutral"] *= 0.50

            # Renormalize (this automatically gives more room to other emotions)
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                for emo in emotion_scores:
                    emotion_scores[emo] /= total_score

            # Only allow Neutral if it's REALLY dominant (>60%)
            if emotion_scores["Neutral"] < 0.60 and emotion_scores["Neutral"] == max(emotion_scores.values()):
                # Neutral won but wasn't dominant, pick second best
                temp_scores = emotion_scores.copy()
                temp_scores["Neutral"] = 0
                best_emotion = max(temp_scores.items(), key=lambda x: x[1])
            else:
                # Get most confident emotion normally
                best_emotion = max(emotion_scores.items(), key=lambda x: x[1])

            return {
                "emotion": best_emotion[0],
                "confidence": best_emotion[1],
                "distribution": emotion_scores
            }

    def get_top_k_emotions(self, k=3):
        """Get top k emotions with their confidence scores."""
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

# LOAD TRAINED DINO+SVM MODEL
def load_custom_emotion_model():
    """Load the trained DINO+SVM model from saved files"""
    model_dir = "model_configuration"

    try:
        # Load SVM model
        svm_path = os.path.join(model_dir, 'best_svm_model.pkl')
        svm_model = joblib.load(svm_path)
        print(f"Loaded SVM model from {svm_path}")

        # Load DINO model name
        config_path = os.path.join(model_dir, 'dino_model_config.txt')
        with open(config_path, 'r') as f:
            dino_model_name = f.read().strip()
        print(f"DINO model: {dino_model_name}")

        # Load class names
        classes_path = os.path.join(model_dir, 'class_names.txt')
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Classes: {class_names}")

        # Load DINO model and processor
        processor = AutoImageProcessor.from_pretrained(dino_model_name)
        dino_model = AutoModel.from_pretrained(dino_model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dino_model.to(device)
        dino_model.eval()
        print(f"DINO model loaded on {device}")

        return svm_model, dino_model, processor, device, class_names
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found - {e}")
        print("Please run the notebook to train and save the model first.")
        print(f"Required files in '{model_dir}/':")
        print("  - best_svm_model.pkl")
        print("  - dino_model_config.txt")
        print("  - class_names.txt")
        return None, None, None, None, None
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None, None, None, None, None

# CV2 LIVE EMOTION DETECTION (DINO+SVM)
def cv2_vid_with_emotion():
    """
    Continuously captures video and detects emotions using DINO+SVM.
    Predictions are added to the global emotion_aggregator.
    """
    # Load trained model
    svm_model, dino_model, processor, device, class_names = load_custom_emotion_model()

    if svm_model is None:
        print("FATAL: Cannot start emotion detection without model")
        return

    # Initialize face detector (only for face detection, not emotion)
    face_model_path = "./weights/Alignment_RetinaFace.pth"
    face_detector = FaceDetector(model_path=face_model_path, device="cpu")

    cam = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cam.isOpened():
        print("ERROR: Failed to open camera")
        return

    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Give camera time to initialize
    import time
    time.sleep(0.5)

    frame_count = 0
    PROCESS_EVERY = 3  # Process every 3rd frame for efficiency

    print("Starting emotion detection with DINO+SVM...")

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

                    # Extract DINO features
                    with torch.inference_mode():
                        inputs = processor(images=pil_image, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = dino_model(**inputs)
                        features = outputs.pooler_output.detach().cpu().numpy()[0]

                    # Get probability distribution first
                    if hasattr(svm_model, 'predict_proba'):
                        probs = svm_model.predict_proba([features])[0]
                    else:
                        # Create one-hot if probabilities not available
                        pred_idx = svm_model.predict([features])[0]
                        probs = np.zeros(len(class_names))
                        probs[pred_idx] = 1.0

                    # Apply class balancing - Neutral was overrepresented in training (41%)
                    neutral_idx = class_names.index("Neutral")
                    adjusted_probs = probs.copy()
                    adjusted_probs[neutral_idx] *= 0.50  # Penalize Neutral by 50%

                    # Renormalize (this automatically gives more room to other emotions)
                    adjusted_probs /= adjusted_probs.sum()

                    # Only predict Neutral if it's REALLY dominant (>60% after adjustment)
                    if adjusted_probs[neutral_idx] < 0.60 and adjusted_probs[neutral_idx] == adjusted_probs.max():
                        # If Neutral won but wasn't dominant, pick second best
                        adjusted_probs[neutral_idx] = 0
                        pred_idx = np.argmax(adjusted_probs)
                    else:
                        pred_idx = np.argmax(adjusted_probs)

                    # DEBUG: Print predictions every 30 frames
                    if frame_count % 90 == 0:
                        print(f"[DEBUG] Frame {frame_count}: Predicted {class_names[pred_idx]} (idx={pred_idx}, conf={probs[pred_idx]:.2f})")
                        print(f"[DEBUG] Probs: {dict(zip(class_names, probs))}")

                    # Add prediction to aggregator
                    emotion_aggregator.add_prediction(
                        emotion=class_names[pred_idx],
                        confidence=float(probs[pred_idx]),
                        probs=probs
                    )

            except Exception:
                pass  # Silently handle errors to keep perception running

    cam.release()
    print("Perception thread stopped.")

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

        # LLM setup 
        self.llm_api_key = api_key
        self.client = genai.Client(api_key=api_key) if api_key else None

        # Store recent conversation turns
        self.conversation_history = []

        # Chat instance for Gemini with safety settings
        self.chat = None
        if self.client:
            from google.genai import types
            self.chat = self.client.chats.create(
                model="gemini-1.5-flash",  # Using 1.5-flash (higher free tier: 1500 req/day vs 20 req/day for 2.5-flash)
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

            # Get top-3 emotions for context (already interview emotions from DINO+SVM)
            top_3_raw = emotion_aggregator.get_top_k_emotions(k=3)

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

    # DELIVERY QUALITY ANALYSIS
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
                return False, "Please focus on answering the interview question naturally."

        return True, None

    # LLM INTEGRATION
    def get_llm_feedback(self, user_answer, emotion, question_asked, speech_quality=None, top_3_emotions=None):
        """Generate feedback using Gemini API with fallback to rule-based responses."""
        if not self.chat:
            return self._get_fallback_feedback(emotion)

        try:
            # Build emotion-specific strategy
            emotion_strategies = {
                Emotion.NERVOUS: "Be very reassuring. Use phrases like 'That's a good start' or 'I understand'. Normalize nervousness.",
                Emotion.CONFIDENT: "Be professional and brief. Acknowledge competence. Keep it to 1-2 sentences.",
                Emotion.NEUTRAL: "Be balanced. Provide constructive feedback.",
                Emotion.DEFENSIVE: "Be gentle and non-judgmental. Avoid words like 'wrong' or 'mistake'. Validate their perspective."
            }

            # Build conversation history context
            recent_history = self.conversation_history[-3:] if len(self.conversation_history) > 0 else []
            history_text = "\n".join([f"- {h}" for h in recent_history])

            # Build emotion distribution context
            emotion_context = f"- Primary emotion: {emotion.value}"
            if top_3_emotions and len(top_3_emotions) > 1:
                distribution_str = ", ".join([f"{e.value} ({s:.0%})" for e, s in top_3_emotions[:3]])
                emotion_context += f"\n- Emotion distribution during answer: {distribution_str}"

                if top_3_emotions[0][1] > 0.6:
                    emotion_context += f"\n- Note: Candidate was predominantly {top_3_emotions[0][0].value} throughout the answer."
                elif len(top_3_emotions) >= 2 and top_3_emotions[1][1] > 0.25:
                    emotion_context += f"\n- Note: Mixed emotions detected - candidate showed {top_3_emotions[0][0].value} and {top_3_emotions[1][0].value}."

            # Build delivery quality context
            delivery_context = ""
            if speech_quality:
                delivery_context = f"""
DELIVERY QUALITY ANALYSIS:
- Speech fluency: {speech_quality.value}"""

                if speech_quality == SpeechQuality.HESITANT:
                    delivery_context += "\n- Note: If hesitant, gently encourage speaking more confidently without mentioning filler words directly."
                elif speech_quality == SpeechQuality.FLUENT:
                    delivery_context += "\n- Note: Excellent delivery - acknowledge this briefly."

            # Build complete prompt
            prompt = f"""MULTI-MODAL CONTEXT:
{emotion_context}
- Questions asked so far: {self.questions_asked}
- Current phase: {self.current_phase}{delivery_context}

CONVERSATION HISTORY:
{history_text if history_text else "This is early in the conversation"}

CURRENT SITUATION:
Question asked: "{question_asked}"
User's answer: "{user_answer}"

STRATEGY FOR {emotion.value.upper()}:
{emotion_strategies[emotion]}

TASK:
Provide brief, empathetic feedback (max 40 words, 2-3 sentences).
- Focus primarily on the content and delivery of this specific answer
- Use the emotion distribution to acknowledge emotional journey if relevant (e.g., "I can see you were a bit nervous but pushed through")
- Be supportive and help build confidence

Just provide the feedback text, nothing else."""

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
    def adjust_behavior_for_emotion(self, emotion):
        """Dynamically adjust Furhat's behavior based on detected emotion."""
        # TODO: FURHAT BEHAVIOR - Emotion-adaptive gestures
        if emotion == Emotion.NERVOUS:
            self.furhat.gesture(name="Smile")  # TODO: FURHAT BEHAVIOR - Reassuring gesture
        elif emotion == Emotion.CONFIDENT:
            self.furhat.gesture(name="Nod")  # TODO: FURHAT BEHAVIOR - Acknowledging gesture
        elif emotion == Emotion.DEFENSIVE:
            self.furhat.gesture(name="Oh")  # TODO: FURHAT BEHAVIOR - Gentle gesture
        elif emotion == Emotion.NEUTRAL:
            self.furhat.gesture(name="Nod")  # TODO: FURHAT BEHAVIOR - Neutral gesture

    # IMPROVED QUESTION SELECTION 
    def calculate_performance_score(self):
        """
        Calculate performance score from recent answers (last 2) to determine appropriate difficulty.
        This helps users practice at the right level based on their current emotional state and delivery.

        Returns: score from 0-100 (higher = better recent performance)
        """
        # First question: always start easy
        if not self.emotion_history or not self.speech_quality_history:
            return 15  # Easy level

        # Look at last 2 answers (or fewer if we don't have 2 yet)
        recent_emotions = self.emotion_history[-2:]
        recent_speech = self.speech_quality_history[-2:]

        scores = []

        # Calculate score for each recent answer
        for i in range(len(recent_emotions)):
            emotion = recent_emotions[i]
            speech = recent_speech[i] if i < len(recent_speech) else SpeechQuality.CLEAR

            # Emotion score (0-40 points)
            emotion_score = 0
            if emotion == Emotion.CONFIDENT:
                emotion_score = 40
            elif emotion == Emotion.NEUTRAL:
                emotion_score = 25
            elif emotion == Emotion.DEFENSIVE:
                emotion_score = 15
            elif emotion == Emotion.NERVOUS:
                emotion_score = 10

            # Speech quality score (0-40 points)
            speech_score = 0
            if speech == SpeechQuality.FLUENT:
                speech_score = 40
            elif speech == SpeechQuality.CLEAR:
                speech_score = 25
            elif speech == SpeechQuality.HESITANT:
                speech_score = 10

            # Combined score for this answer (max 80)
            answer_score = emotion_score + speech_score
            scores.append(answer_score)

        # Average the recent scores
        avg_score = sum(scores) / len(scores)

        # Normalize to 0-100 scale
        normalized_score = (avg_score / 80) * 100

        return normalized_score

    def select_question_difficulty(self):
        """
        Select difficulty based on recent performance to support user growth.
        Uses last 2 answers to be responsive without being jumpy.
        """
        # Calculate performance from recent answers
        performance = self.calculate_performance_score()

        # Map performance to difficulty
        # Thresholds designed to be supportive:
        # - 0-35: easy (nervous or hesitant delivery)
        # - 35-65: medium (neutral/clear or mixed performance)
        # - 65-100: hard (confident and fluent)

        if performance < 35:
            difficulty = "easy"
        elif performance < 65:
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

    # PHASE METHODS
    def introduction(self):
        """Phase 1: Introduction"""
        print("\nPHASE: INTRODUCTION")
        self.current_phase = "introduction"

        # TODO: FURHAT BEHAVIOR - Welcome gesture
        self.furhat.gesture(name="Smile")
        # TODO: FURHAT BEHAVIOR - Greeting speech (consider voice parameters: rate, pitch)
        self.furhat.say(text="Hello! Welcome to Interview Practice Coach.", blocking=True)
        time.sleep(0.5)
        # TODO: FURHAT BEHAVIOR - Introduction speech
        self.furhat.say(text="I'm Furhat, and I'm here to help you practice for your upcoming interviews.", blocking=True)
        time.sleep(0.5)
        # TODO: FURHAT BEHAVIOR - Question speech
        self.furhat.say(text="What's your name?", blocking=True)

        # TODO: FURHAT BEHAVIOR - Listen for user name
        response = self.furhat.listen()
        if response and response.message:
            self.user_name = response.message.strip().split()[0].capitalize()
            # TODO: FURHAT BEHAVIOR - Personalized greeting
            self.furhat.say(text=f"Nice to meet you, {self.user_name}!", blocking=True)
        else:
            self.user_name = "there"

        time.sleep(0.5)
        # TODO: FURHAT BEHAVIOR - Ready check speech
        self.furhat.say(text="Are you ready to begin?", blocking=True)
        # TODO: FURHAT BEHAVIOR - Listen for confirmation
        self.furhat.listen()

    def warmup_phase(self):
        """Phase 2: Warmup"""
        print("\nPHASE: WARMUP")
        self.current_phase = "warmup"

        question = self.get_next_question("easy")
        self.questions_asked += 1

        # TODO: FURHAT BEHAVIOR - Warmup introduction speech
        self.furhat.say(text="Let's start with a simple question.", blocking=True)
        time.sleep(0.5)
        # TODO: FURHAT BEHAVIOR - Ask first question
        self.furhat.say(text=question, blocking=True)

        # Clear emotion buffer to collect emotions during answer
        emotion_aggregator.clear()

        # Listen to answer (emotions are collected during this time)
        start_time = time.time()
        # TODO: FURHAT BEHAVIOR - Listen for answer
        response = self.furhat.listen()
        end_time = time.time()
        duration = end_time - start_time

        user_answer = response.message if response and response.message else "No response"

        # Handle case when Furhat doesn't catch the answer
        if user_answer == "No response":
            # TODO: FURHAT BEHAVIOR - Polite retry speech
            self.furhat.say(text="I didn't catch that. Could you please try again?", blocking=True)
            # Give them another chance
            start_time = time.time()
            # TODO: FURHAT BEHAVIOR - Listen again
            response = self.furhat.listen()
            end_time = time.time()
            duration = end_time - start_time
            user_answer = response.message if response and response.message else "No response"

            # If still no response, acknowledge and move on
            if user_answer == "No response":
                # TODO: FURHAT BEHAVIOR - Reassuring speech
                self.furhat.say(text="That's okay. Take your time. Let's continue.", blocking=True)

        # Pre-filter user answer for safety (jailbreak detection)
        is_safe, warning = self.pre_filter_user_answer(user_answer)
        if not is_safe:
            print(f"Safety warning: Blocked potentially harmful input")
            # TODO: FURHAT BEHAVIOR - Safety warning speech
            self.furhat.say(text=warning, blocking=True)
            return  # Skip processing this answer

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

        # Give feedback based on detected emotion and delivery quality
        feedback = self.get_llm_feedback(user_answer, emotion, question, speech_quality, top_3_emotions)

        # TODO: FURHAT BEHAVIOR - Emotion-adaptive gesture before feedback
        self.adjust_behavior_for_emotion(emotion)
        # TODO: FURHAT BEHAVIOR - Deliver personalized feedback (consider voice parameters based on emotion)
        self.furhat.say(text=feedback, blocking=True)

        # Update history
        self.conversation_history.append(f"Q: {question}")
        self.conversation_history.append(f"A: {user_answer[:50]}...")
        self.conversation_history.append(f"Emotion: {emotion.value}, Speech: {speech_quality.value}")

    def main_interview(self):
        """Phase 3: Main interview with dynamic state handling"""
        print("\nPHASE: MAIN INTERVIEW")
        self.current_phase = "main"

        target_questions = 3

        while self.questions_asked < target_questions + 1:  # +1 for warmup
            time.sleep(0.5)

            print(f"\nTurn {self.questions_asked + 1}")

            # Select question difficulty based on emotion HISTORY
            difficulty = self.select_question_difficulty()
            question = self.get_next_question(difficulty)
            self.questions_asked += 1

            print(f"Asking {difficulty} question")

            # TODO: FURHAT BEHAVIOR - Adjust gesture based on previous emotion
            if self.emotion_history:
                self.adjust_behavior_for_emotion(self.current_emotion)

            # TODO: FURHAT BEHAVIOR - Ask adaptive difficulty question
            self.furhat.say(text=question, blocking=True)

            # Clear emotion buffer to collect emotions during answer
            emotion_aggregator.clear()

            # Listen to answer (emotions are collected during this time)
            start_time = time.time()
            # TODO: FURHAT BEHAVIOR - Listen for answer
            response = self.furhat.listen()
            end_time = time.time()
            duration = end_time - start_time

            user_answer = response.message if response and response.message else "No response"

            # Handle case when Furhat doesn't catch the answer
            if user_answer == "No response":
                # TODO: FURHAT BEHAVIOR - Polite retry speech
                self.furhat.say(text="I didn't catch that. Could you please try again?", blocking=True)
                # Give them another chance
                start_time = time.time()
                # TODO: FURHAT BEHAVIOR - Listen again
                response = self.furhat.listen()
                end_time = time.time()
                duration = end_time - start_time
                user_answer = response.message if response and response.message else "No response"

                # If still no response, acknowledge and move on
                if user_answer == "No response":
                    # TODO: FURHAT BEHAVIOR - Reassuring speech
                    self.furhat.say(text="That's okay. Take your time. Let's continue.", blocking=True)

            # Pre-filter user answer for safety (jailbreak detection)
            is_safe, warning = self.pre_filter_user_answer(user_answer)
            if not is_safe:
                print(f"Safety warning: Blocked potentially harmful input")
                # TODO: FURHAT BEHAVIOR - Safety warning speech
                self.furhat.say(text=warning, blocking=True)
                continue  # Skip processing this answer and move to next question

            # NOW detect emotion from the answer
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
            feedback = self.get_llm_feedback(user_answer, emotion, question, speech_quality, top_3_emotions)

            # TODO: FURHAT BEHAVIOR - Emotion-adaptive gesture before feedback
            self.adjust_behavior_for_emotion(emotion)
            # TODO: FURHAT BEHAVIOR - Deliver personalized feedback (consider voice parameters based on emotion)
            self.furhat.say(text=feedback, blocking=True)

            # Update conversation history
            self.conversation_history.append(f"Q: {question}")
            self.conversation_history.append(f"A: {user_answer[:50]}...")
            self.conversation_history.append(f"Emotion: {emotion.value}, Speech: {speech_quality.value}")

    def closing_summary(self):
        """Phase 4: Closing"""
        print("\nPHASE: CLOSING")
        self.current_phase = "closing"

        # TODO: FURHAT BEHAVIOR - Positive closing gesture
        self.furhat.gesture(name="Smile")
        # TODO: FURHAT BEHAVIOR - Closing speech (consider warm, encouraging tone)
        self.furhat.say(text="Great job today! Let me give you some feedback.", blocking=True)
        time.sleep(0.5)

        summary = self.generate_emotional_summary()
        # TODO: FURHAT BEHAVIOR - Deliver emotional summary
        self.furhat.say(text=summary, blocking=True)
        time.sleep(1.0)

        # TODO: FURHAT BEHAVIOR - Encouraging closing message
        self.furhat.say(text="Remember, practice makes progress!", blocking=True)
        # TODO: FURHAT BEHAVIOR - Personalized goodbye
        self.furhat.say(text=f"Good luck, {self.user_name}!", blocking=True)

    def generate_emotional_summary(self):
        """Generate summary based on emotion trajectory"""
        total = len(self.emotion_history)
        if total == 0:
            return "You did well today."

        nervous = self.emotion_history.count(Emotion.NERVOUS)
        confident = self.emotion_history.count(Emotion.CONFIDENT)

        nervous_pct = (nervous / total) * 100
        confident_pct = (confident / total) * 100

        if nervous_pct > 60:
            return "I noticed you were nervous, which is normal. With practice, this will get easier."
        elif confident_pct > 60:
            return "You showed great confidence! Keep that energy in real interviews."
        else:
            return "You showed good composure during our practice. Keep up the confidence!"
    # SESSION REPORT
    def generate_session_report(self):
        """Generate detailed report for evaluation."""
        print(f"  Total questions: {self.questions_asked}")
        print(f"  Total turns: {len(self.emotion_history)}")
        print(f"  LLM failures: {self.llm_failures}")
        print(f"  Emotion detection failures: {self.emotion_detection_failures}")

        print(f"\nEMOTION TRAJECTORY:")
        for i, emotion in enumerate(self.emotion_history, 1):
            print(f"  Turn {i}: {emotion.value}")

        # Calculate emotion percentages
        total = len(self.emotion_history)
        if total > 0:
            nervous_pct = (self.emotion_history.count(Emotion.NERVOUS) / total) * 100
            confident_pct = (self.emotion_history.count(Emotion.CONFIDENT) / total) * 100
            neutral_pct = (self.emotion_history.count(Emotion.NEUTRAL) / total) * 100
            defensive_pct = (self.emotion_history.count(Emotion.DEFENSIVE) / total) * 100

            print(f"\nEMOTION DISTRIBUTION:")
            print(f"  Nervous: {nervous_pct:.1f}%")
            print(f"  Confident: {confident_pct:.1f}%")
            print(f"  Neutral: {neutral_pct:.1f}%")
            print(f"  Defensive: {defensive_pct:.1f}%")

        # Calculate delivery quality statistics
        if len(self.speech_quality_history) > 0:
            fluent_count = self.speech_quality_history.count(SpeechQuality.FLUENT)
            hesitant_count = self.speech_quality_history.count(SpeechQuality.HESITANT)
            clear_count = self.speech_quality_history.count(SpeechQuality.CLEAR)

            print(f"\nSPEECH QUALITY ANALYSIS:")
            print(f"  Fluent: {fluent_count}, Clear: {clear_count}, Hesitant: {hesitant_count}")

            # Average metrics
            if self.delivery_metrics_history:
                avg_word_count = sum(m['word_count'] for m in self.delivery_metrics_history) / len(self.delivery_metrics_history)
                avg_filler_ratio = sum(m['filler_ratio'] for m in self.delivery_metrics_history) / len(self.delivery_metrics_history)
                valid_wpm = [m['wpm'] for m in self.delivery_metrics_history if m['wpm'] > 0]
                avg_wpm = sum(valid_wpm) / len(valid_wpm) if valid_wpm else 0

                print(f"\n  Average Metrics:")
                print(f"    Words per response: {avg_word_count:.1f}")
                print(f"    Filler word ratio: {avg_filler_ratio:.1%}")
                print(f"    Speaking rate: {avg_wpm:.0f} WPM")

        print("="*60 + "\n")

        return {
            "questions_asked": self.questions_asked,
            "emotion_trajectory": [e.value for e in self.emotion_history],
            "speech_quality_trajectory": [s.value for s in self.speech_quality_history],
            "delivery_metrics": self.delivery_metrics_history,
            "llm_failures": self.llm_failures,
            "emotion_failures": self.emotion_detection_failures
        }

    # MAIN RUN METHOD
    def run(self):
        """Main flow"""
        try:
            # Phase 1: Introduction
            self.introduction()

            # Phase 2: Warmup
            self.warmup_phase()

            # Phase 3: Main Interview
            self.main_interview()

            # Phase 4: Closing
            self.closing_summary()

            # Generate evaluation report
            report = self.generate_session_report()

            # Save report for evaluation
            with open('session_report.json', 'w') as f:
                json.dump(report, f, indent=2)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # TODO: FURHAT BEHAVIOR - Error apology speech
            self.furhat.say(text="I apologize for the technical difficulty.")

if __name__ == "__main__":

    API_KEY = os.environ.get("GEMINI_API_KEY")

    # Furhat configuration - update this to your Furhat's IP address
    FURHAT_IP = "192.168.1.5"  # Change this to your Furhat's IP address

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
    coach.run()
