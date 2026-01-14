# Emotion-Aware Interview Coaching System

An adaptive interview practice system using the Furhat robot with real-time emotion classification.

## Project Structure

```
├── structured_interview_final.py           # Main interview system
├── emotion_classification_experiments.ipynb # ML experiments notebook
├── model_configuration/                     # Trained model files
│   ├── best_svm_model_facial.pkl           # SVM classifier
│   ├── dino_model_config_facial.txt        # DiNOv3 model config
│   └── class_names_facial.txt              # Emotion labels
└── .env                                     # API keys
```

## Setup

### 1. Create virtual environment

```bash
python3 -m venv env
source env/bin/activate  # Linux/macOS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# Install OpenFace separately
pip install --no-deps openface-test==0.1.26
openface download
```

### 3. Configure API key

Create `.env` file:
```
GEMINI_API_KEY=your_key_here
```

## Usage

### Run Interview System
```bash
python structured_interview_final.py
```

## Dataset

Training uses DiffusionFER dataset with emotions mapped to 4 interview-relevant classes:

| Original Emotion | Interview Class |
|------------------|-----------------|
| sad, fear | Nervous |
| happy | Confident |
| neutral | Neutral |
| angry, disgust, contempt | Defensive |

## Model Performance

| Model | F1 Score |
|-------|----------|
| DiNOv3 + 8 Facial Regions + SVM | 88.1% |

## Interview Flow

1. **Introduction** - Greeting, get user name, ready check
2. **Warmup** - 1 easy question to establish baseline
3. **Main Interview** - 3 adaptive questions based on performance
4. **Closing** - Summary and encouragement

## System Overview

- **Emotion Classification**: DiNOv3 + SVM (4 classes: Nervous, Defensive, Confident, Neutral)
- **Temporal Aggregation**: Exponential decay smoothing (window=30, decay=0.98)
- **Adaptive Difficulty**: Question selection based on emotion + speech quality
- **LLM Feedback**: Gemini API with emotion-aware prompting
- **Robot Gestures**: Emotion-adaptive behaviors via Furhat API

## Architecture

### Core Components

| Component | Function | Description |
|-----------|----------|-------------|
| `cv2_vid_with_emotion()` | Perception | Captures webcam frames, extracts DiNOv3 facial region features, predicts emotions via SVM |
| `EmotionAggregator` | Temporal Smoothing | Aggregates predictions using exponential decay (window=30, λ=0.98) |
| `realtime_gesture_loop()` | Real-time Gestures | Triggers robot gestures when emotion sustained for threshold duration |
| `InterviewCoach` | Main Controller | Manages interview flow, question selection, feedback generation |

### InterviewCoach Methods

| Method | Description |
|--------|-------------|
| `detect_emotion()` | Gets aggregated emotion from perception system |
| `evaluate_delivery_quality()` | Analyzes filler words and speaking rate |
| `calculate_performance_score()` | Combines emotion + speech quality for difficulty selection |
| `get_llm_feedback()` | Generates emotion-aware feedback via Gemini API |
| `adjust_behavior_for_emotion()` | Triggers emotion-specific robot gestures |
| `listen_with_hesitation_detection()` | Continues listening if user says only filler words |

### Data Flow

```
Webcam → Face Detection → DiNOv3 Features → SVM → Emotion Aggregator
                                                         ↓
User Speech → Furhat ASR → Speech Analysis ────→ Performance Score
                                                         ↓
                                            Question Difficulty Selection
                                                         ↓
                              Gemini API ←── Emotion + Answer ──→ Feedback
                                                         ↓
                                               Robot Gesture + Speech
```
