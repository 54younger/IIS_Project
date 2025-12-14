
"""
gesture_mapping.py

Mapping from (emotion_state, answer_score) → GestureConfig
NO voice mapping. Does not call Furhat.

Gesture names must match those returned by:
    GET /furhat/gestures   (Remote API)  
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EmotionState(str, Enum):
    NERVOUS = "nervous"
    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"


@dataclass
class GestureConfig:
    """
    Configuration for /furhat/gesture endpoint.

    Remote API:
        POST /furhat/gesture?name=<gesture_name>&blocking=<bool>
    """
    gesture_name: str
    blocking: bool = False


class GestureMapper:
    """
    Pure logic: (emotion, answer_score) -> GestureConfig
    NO Furhat calls, NO voice mapping.
    """

    def map_gesture(
        self,
        emotion: EmotionState,
        answer_score: Optional[float]
    ) -> GestureConfig:
        """
        Main public API.
        """
        score = self._normalize_score(answer_score)

        if emotion == EmotionState.NERVOUS:
            return self._map_nervous(score)
        if emotion == EmotionState.CONFIDENT:
            return self._map_confident(score)
        if emotion == EmotionState.NEUTRAL:
            return self._map_neutral(score)
        if emotion == EmotionState.DEFENSIVE:
            return self._map_defensive(score)

        return self._map_neutral(score)

    # ------------ Internal helpers ------------

    @staticmethod
    def _normalize_score(score: Optional[float]) -> float:
        if score is None:
            return 0.5
        return max(0.0, min(1.0, float(score)))

    # Gesture mappings — you may replace gesture_name with robot-defined gestures

    def _map_nervous(self, score: float) -> GestureConfig:
        """
        Nervous → gentle & supportive.
        """
        return GestureConfig(
            gesture_name="Smile",   # Example; ensure it exists on robot
            blocking=False
        )

    def _map_confident(self, score: float) -> GestureConfig:
        """
        Confident → nodding or expressive motion depending on score.
        """
        if score >= 0.7:
            gesture = "Nod"
        else:
            gesture = "BigSmile" if score >= 0.5 else "Smile"

        return GestureConfig(
            gesture_name=gesture,
            blocking=False
        )

    def _map_neutral(self, score: float) -> GestureConfig:
        """
        Neutral → minimal gesture (Blink etc.)
        """
        return GestureConfig(
            gesture_name="Blink",
            blocking=False
        )

    def _map_defensive(self, score: float) -> GestureConfig:
        """
        Defensive → low-intensity, empathetic gestures.
        """
        return GestureConfig(
            gesture_name="ExpressSad",
            blocking=False
        )


# ------------ Self Test ------------

def _run_self_test():
    mapper = GestureMapper()
    tests = [
        (EmotionState.NERVOUS, 0.2),
        (EmotionState.NERVOUS, 0.9),
        (EmotionState.CONFIDENT, 0.3),
        (EmotionState.CONFIDENT, 0.85),
        (EmotionState.NEUTRAL, 0.5),
        (EmotionState.DEFENSIVE, 0.1),
    ]

    print("=== gesture_mapping test ===")
    for emotion, score in tests:
        gesture = mapper.map_gesture(emotion, score)
        print(f"Emotion={emotion.value}, score={score} → gesture={gesture}")


if __name__ == "__main__":
    _run_self_test()
