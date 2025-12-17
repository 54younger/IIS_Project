from furhat_remote_api import FurhatRemoteAPI
import time

# Connect to Furhat
FURHAT_IP = "localhost"
furhat = FurhatRemoteAPI(FURHAT_IP)

Nervous_behavior = {
    "name": "Nervous_behavior",
    "frames": [
        {
            "time": [0.3, 1.2],  # slower onset, longer hold
            "persist": False,
            "params": {
                "SMILE_CLOSED": 0.7,
                "EYE_SQUINT_LEFT": 0.25,  # warm "smiling eyes"
                "EYE_SQUINT_RIGHT": 0.25,
                "BROW_UP_LEFT": 0.2,
                "BROW_UP_RIGHT": 0.2
            }
        },
        {
            "time": [0.5],
            "persist": False,
            "params": {"NECK_TILT": 10}  # slow nod down
        },
        {
            "time": [0.9],
            "persist": False,
            "params": {"NECK_TILT": 5}
        },
        {
            "time": [1.2],
            "persist": False,
            "params": {"NECK_PAN": 3}  # slight head tilt = caring
        },
        {
            "time": [1.8],
            "persist": False,
            "params": {"reset": True}
        }
    ],
    "class": "furhatos.gestures.Gesture"
}

Confident_behavior = {
    "name": "Condident_behavior",
    "frames": [
        {
            "time": [0.1],
            "persist": False,
            "params": {
                "BROW_UP_LEFT": 1.0,
                "BROW_UP_RIGHT": 1.0,
                "SURPRISE": 0.3
            }
        },
        {
            "time": [0.3, 0.7],
            "persist": False,
            "params": {
                "SMILE_OPEN": 0.6,
                "SMILE_CLOSED": 0.5
            }
        },
        {
            "time": [0.4, 0.6],
            "persist": False,
            "params": {
                "NECK_TILT": 6
            }
        },
        {
            "time": [0.9],
            "persist": False,
            "params": {"reset": True}
        }
    ],
    "class": "furhatos.gestures.Gesture"
}

Defensive_behavior = {
    "name": "Defensive_behavior",
    "frames": [
        {
            "time": [0.3, 1.2],
            "persist": False,
            "params": {
                "BROW_IN_LEFT": 0.5,      # more visible furrow
                "BROW_IN_RIGHT": 0.5,
                "BROW_UP_LEFT": 0.4,      # inner brow raise = empathy
                "BROW_UP_RIGHT": 0.4,
                "EXPR_SAD": 0.15          # subtle concern
            }
        },
        {
            "time": [0.4, 1.3],
            "persist": False,
            "params": {
                "NECK_PAN": -15,          # clear head tilt to side
                "NECK_TILT": 8
            }
        },
        {
            "time": [0.9, 1.4],
            "persist": False,
            "params": {
                "SMILE_CLOSED": 0.4       # gentle reassuring smile at end
            }
        },
        {
            "time": [1.8],
            "persist": False,
            "params": {"reset": True}
        }
    ],
    "class": "furhatos.gestures.Gesture"
}

Neutral_behavior = {
    "name": "AttentiveListen",
    "frames": [
        {
            "time": [0.15, 0.6],
            "persist": False,
            "params": {
                "BROW_UP_LEFT": 0.25,
                "BROW_UP_RIGHT": 0.25
            }
        },
        {
            "time": [0.3],
            "persist": False,
            "params": {"NECK_TILT": 7}    # single nod
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

ATTENTIVE_LISTEN = {
    "name": "AttentiveListen",
    "frames": [
        {
            "time": [0.15, 0.6],
            "persist": False,
            "params": {
                "BROW_UP_LEFT": 0.25,
                "BROW_UP_RIGHT": 0.25,
                "SMILE_CLOSED": 0.3       # gentle smile
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

# Active listening with micro-nods
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
# Transitioning to next question
TRANSITION = {
    "name": "Transition",
    "frames": [
        {
            "time": [0.1],
            "persist": False,
            "params": {
                "BROW_UP_LEFT": 0.5,
                "BROW_UP_RIGHT": 0.5
            }
        },
        {
            "time": [0.25, 0.5],
            "persist": False,
            "params": {
                "SMILE_CLOSED": 0.4,
                "NECK_TILT": 4
            }
        },
        {
            "time": [0.7],
            "persist": False,
            "params": {"reset": True}
        }
    ],
    "class": "furhatos.gestures.Gesture"
}

WRAP_UP = {
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

# All gestures in a dict for easy testing
CUSTOM_GESTURES = {
    "1": ("NERVOUS", Nervous_behavior),
    "2": ("CONFIDENT", Confident_behavior),
    "3": ("DEFENSIVE", Defensive_behavior),
    "4": ("NEUTRAL", Neutral_behavior),
    "5":("ATTENTIVE_LISTEN", ATTENTIVE_LISTEN),
    "6":("ACTIVE_LISTEN", ACTIVE_LISTEN),
    "7":("TRANSITION", TRANSITION),
    "8":("WRAP_UP", WRAP_UP)
}

def test_gesture(gesture_def):
    """Send a custom gesture to Furhat."""
    furhat.gesture(body=gesture_def)


def main():
    print("\nAvailable gestures:")
    for key, (name, _) in CUSTOM_GESTURES.items():
        print(f"  {key}: {name}")
    print("  q: Quit")
    print("  a: Test all gestures in sequence")
    print()

    while True:
        choice = input("Enter gesture number (or 'q' to quit, 'a' for all): ").strip().lower()

        if choice == 'q':
            print("Goodbye!")
            break
        elif choice == 'a':
            print("\nTesting all gestures...")
            for key, (name, gesture_def) in CUSTOM_GESTURES.items():
                print(f"  Playing: {name}")
                test_gesture(gesture_def)
                time.sleep(2.5)
        elif choice in CUSTOM_GESTURES:
            name, gesture_def = CUSTOM_GESTURES[choice]
            print(f"  Playing: {name}")
            test_gesture(gesture_def)
            time.sleep(1.5)
        else:
            print("  Invalid choice. Try again.")


if __name__ == "__main__":
    main()
