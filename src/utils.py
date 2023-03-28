"""Utils Functions."""
import argparse
import sys

LR = 'L vs R'
HAND_FEET = 'Hands vs feet'

tasks = [
    {
        "name": "Real moving hands",
        "real": True,
        "type": LR,
        "runs": [3, 7, 11],
        "labels": {0: "rest", 1: "left hand", 2: "right hand"},
        "events": {"T0": 0, "T1": 1, "T2": 2},
    },
    {
        "name": "Imagine moving hands",
        "real": False,
        "type": LR,
        "runs": [4, 8, 12],
        "labels": {0: "rest", 1: "imagine left hand", 2: "imagine right hand"},
        "events": {"T0": 0, "T1": 1, "T2": 2},
    },
    {
        "name": "move hands vs feet",
        "real": True,
        "type": HAND_FEET,
        "runs": [5, 9, 13],
        "labels": {0: "rest", 1: "hands", 2: "feet"},
        "events": {"T0": 0, "T1": 1, "T2": 2},
    },
    {
        "name": "imagine hands vs feet",
        "real": False,
        "type": HAND_FEET,
        "runs": [6, 10, 14],
        "labels": {0: "rest", 1: "imagine both hands", 2: "imagine both feet"},
        "events": {"T0": 0, "T1": 1, "T2": 2},
    }
]


def get_tpv_args():
    """Get program arguments."""
    parser = argparse.ArgumentParser(description="Pipeline for training or predicting on EEG data.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--train', action='store_true', help='Train the model')
    group.add_argument('--predict', action='store_true', help='Make predictions using the model')
    parser.add_argument("--subject", type=int, choices=range(1, 109), metavar="int > 0 && int <= 109",
                        help="Choose the number of the subject ",
                        required=('--train' in sys.argv or '--predict' in sys.argv))
    infos = f"{[{str(index) : str(x['name'])} for index, x, in enumerate(tasks)]}"
    parser.add_argument("--run_idx", type=int, choices=range(0, 4),
                        help=f"""Choose the index corresponding to the experiment you want to analyze.
                        {infos}
                        """,
                        required=('--train' in sys.argv or '--predict' in sys.argv))
    parser.add_argument("--my_csp", action="store_true",
                        help="Use my own csp")
    return parser.parse_args()
