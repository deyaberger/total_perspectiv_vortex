"""Utils Functions."""
import argparse
import sys

EXPERIMENTS = [
    {
        "description": "move fists",
        "runs": [3, 7, 11],
        "mapping": {0: "rest", 1: "left fist", 2: "right fist"},
    },
    {
        "description": "imagine movement of fists",
        "runs": [4, 8, 12],
        "mapping": {0: "rest", 1: "imagine left fist", 2: "imagine right fist"},
    },
    {
        "description": "move fists and feets",
        "runs": [5, 9, 13],
        "mapping": {0: "rest", 1: "both fists", 2: "both feets"},
    },
    {
        "description": "imagine movement of fists and feets",
        "runs": [6, 10, 14],
        "mapping": {0: "rest", 1: "imagine both fists", 2: "imagine both feets"},
    },
    {
        "description": "movement (real or imagine) of fists",
        "runs": [3, 7, 11, 4, 8, 12],
        "mapping": {0: "rest", 1: "left fist", 2: "right fist"},
    },
    {
        "description": "movement (real or imagine) of fists or feet",
        "runs": [5, 9, 13, 6, 10, 14],
        "mapping": {0: "rest", 1: "both fists", 2: "both feets"},
    },
]


def get_tpv_args():
    """Get program arguments."""
    parser = argparse.ArgumentParser(description="Pipeline for training or predicting on EEG data.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--train', action='store_true', help='Train the model')
    group.add_argument('--predict', action='store_true', help='Make predictions using the model')
    parser.add_argument("--subject", type=int, choices=range(0, 109), metavar="int <= 109",
                        help="Choose the number of the subject ",
                        required=('--train' in sys.argv or '--predict' in sys.argv))
    parser.add_argument("--run_idx", type=int, choices=range(0, 6),
                        help="""Choose the index corresponding to the experiment you want to analyze.
                        {0: move fists},
                        {1: imagine movement of fists},
                        {2: move fists and feets},
                        {3: imagine movement of fists and feets},
                        {4: movement (real or imagine) of fists},
                        {5: movement (real or imagine) of fists or feet},
                        """,
                        required=('--train' in sys.argv or '--predict' in sys.argv))
    return parser.parse_args()
