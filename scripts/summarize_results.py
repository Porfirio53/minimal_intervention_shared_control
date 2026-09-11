#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize trajectory-level JSONL results"
    )
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with args.results.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            grouped[row["method"]].append(row)
    columns = (
        "collision",
        "success",
        "minimum_clearance",
        "intervention_budget",
        "filter_modification",
        "authority_total_variation",
    )
    print("method," + ",".join(columns))
    for method in sorted(grouped):
        values = []
        for column in columns:
            values.append(
                str(float(np.mean([float(row[column]) for row in grouped[method]])))
            )
        print(method + "," + ",".join(values))


if __name__ == "__main__":
    main()
