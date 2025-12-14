"""
furhat_integration.py

Integration layer:
- Uses GestureMapper for gesture selection
- Sends gesture via Furhat Remote API

Remote API reference: 
    POST /furhat/gesture?name=<gesture_name>&blocking=<bool>
"""

from __future__ import annotations
import requests

from gesture_mapping import GestureMapper, EmotionState, GestureConfig


class FurhatRemoteClient:
    """
    Minimal HTTP client for Furhat Remote API.
    """

    def __init__(self, host: str, port: int = 54321, dry_run: bool = True):
        self.base_url = f"http://{host}:{port}/furhat"
        self.dry_run = dry_run

    def _post(self, path: str, params: dict):
        url = f"{self.base_url}{path}"
        if self.dry_run:
            print(f"[DRY RUN] POST {url} params={params}")
            return

        try:
            r = requests.post(url, params=params, timeout=2)
            r.raise_for_status()
        except Exception as e:
            print(f"[ERROR] POST {url} failed: {e}")

    def send_gesture(self, config: GestureConfig):
        params = {
            "name": config.gesture_name,
            "blocking": str(config.blocking).lower()
        }
        self._post("/gesture", params)


class GestureApplier:
    """
    Connects GestureMapper and FurhatRemoteClient.
    """

    def __init__(self, mapper=None, client=None, host="172.18.160.1"):
        self.mapper = mapper if mapper else GestureMapper()
        self.client = client if client else FurhatRemoteClient(host, dry_run=True)

    def apply(
        self,
        emotion: EmotionState,
        answer_score: float | None
    ) -> GestureConfig:
        """
        Compute gesture and forward to robot.
        """
        gesture = self.mapper.map_gesture(emotion, answer_score)
        self.client.send_gesture(gesture)
        return gesture


# ------------ Integration Test ------------

def _run_integration_test():
    import time
    client = FurhatRemoteClient(host="172.18.160.1", dry_run=False)
    applier = GestureApplier(client=client)

    tests = [
        (EmotionState.NERVOUS, 0.3),
        (EmotionState.CONFIDENT, 0.9),
        (EmotionState.NEUTRAL, 0.5),
        (EmotionState.DEFENSIVE, 0.1),
    ]

    print("=== furhat_integration test ===")
    for emotion, score in tests:
        print(f"\n[CASE] emotion={emotion.value}, score={score}")
        gesture = applier.apply(emotion, score)
        print(f"  Gesture selected → {gesture}")
        time.sleep(1)

if __name__ == "__main__":
    _run_integration_test()
