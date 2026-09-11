#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from minimal_intervention_shared_control.evaluation.prediction_metrics import (
    gaussian_trajectory_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Gaussian trajectory predictions stored in NPZ"
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="NPZ containing truth, mean, and covariance arrays",
    )
    args = parser.parse_args()
    with np.load(args.predictions) as values:
        result = gaussian_trajectory_metrics(
            values["truth"], values["mean"], values["covariance"]
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
