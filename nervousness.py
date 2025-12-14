
# from nervousness import NervousnessAnalyzer

# nervousness_analyzer = NervousnessAnalyzer(window_size=5)

# while True:
#     response = furhat.listen()
#     text = response.message.strip() if response.message else ""

#     print(f"USER: {text}")

#     nervous_result = nervousness_analyzer.analyze_utterance(text)
#     nervous_score = nervous_result["smoothed_score"]

#     print(
#         f"[NERVOUSNESS] score={nervous_score:.2f} "
#         f"(filler_ratio={nervous_result['filler_ratio']:.2f}, "
#         f"repeat_ratio={nervous_result['repeat_ratio']:.2f}, "
#         f"pause_marks={nervous_result['pause_marks']})"
#     )


from __future__ import annotations

import re
from collections import deque
from typing import Deque, Dict, Optional, Set


class NervousnessAnalyzer:
    """
    NervousnessAnalyzer computes a nervousness score from text features.

    Input:
        text: str  (one utterance / one user message)

    Output (dict):
        {
          "raw_score": float,          # 0~1
          "smoothed_score": float,     # 0~1, moving average over last window_size
          "filler_ratio": float,       # filler_count / token_count
          "repeat_ratio": float,       # repeat_count / token_count
          "pause_marks": int,          # count of "...", "…", "--"
          "utterance_length": int,     # token_count
        }
    """

    def __init__(
        self,
        window_size: int = 5,
        filler_words: Optional[Set[str]] = None,
        w_filler: float = 2.0,
        w_repeat: float = 3.0,
        w_pause: float = 0.05,
    ):
        self.window_size = window_size
        self.scores: Deque[float] = deque(maxlen=window_size)

        self.filler_words: Set[str] = filler_words or {
            "um", "uh", "emm", "em", "umm", "hmm",
            "like", "you know", "kind of", "sort of",
            "and",
        }

        self.w_filler = w_filler
        self.w_repeat = w_repeat
        self.w_pause = w_pause

    def reset(self) -> None:
        """Clear smoothing history."""
        self.scores.clear()

    def _tokenize(self, text: str):
        # simple whitespace + punctuation tokenizer
        return re.findall(r"\w+|[^\s\w]", text.lower())

    def analyze_utterance(self, text: str) -> Dict[str, float | int]:
        """
        Analyze one utterance and return raw & smoothed nervousness scores + features.
        """
        text = text or ""
        tokens = self._tokenize(text)
        num_tokens = max(len(tokens), 1)

        text_lower = text.lower()

        # 1) filler / hesitation
        filler_count = 0
        for fw in self.filler_words:
            if " " in fw:
                filler_count += text_lower.count(fw)
            else:
                filler_count += sum(1 for t in tokens if t == fw)
        filler_ratio = filler_count / num_tokens

        # 2) immediate repeats
        repeat_count = 0
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i - 1] and tokens[i].isalpha():
                repeat_count += 1
        repeat_ratio = repeat_count / num_tokens

        # 3) pause marks
        pause_marks = text.count("...") + text.count("…") + text.count("--")

        # weighted sum -> clamp [0, 1]
        score = (
            self.w_filler * filler_ratio +
            self.w_repeat * repeat_ratio +
            self.w_pause  * pause_marks
        )
        score = max(0.0, min(1.0, score))

        # smoothing
        self.scores.append(score)
        smoothed = sum(self.scores) / len(self.scores)

        return {
            "raw_score": float(score),
            "smoothed_score": float(smoothed),
            "filler_ratio": float(filler_ratio),
            "repeat_ratio": float(repeat_ratio),
            "pause_marks": int(pause_marks),
            "utterance_length": int(num_tokens),
        }
