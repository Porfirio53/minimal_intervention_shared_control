#!/usr/bin/env python3
"""Convert one THOR Qualisys wide TSV into normalized 2-D person tracks."""

from __future__ import annotations

import argparse
from pathlib import Path

from minimal_intervention_shared_control.datasets.thor import convert_thor_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, default=1e-3)
    parser.add_argument("--min-markers", type=int, default=1)
    parser.add_argument("--max-gap-factor", type=float, default=1.5)
    args = parser.parse_args()
    rows = convert_thor_tsv(
        args.input,
        args.output,
        scale=args.scale,
        min_markers=args.min_markers,
        max_gap_factor=args.max_gap_factor,
    )
    print(f"wrote {rows} observations to {args.output}")


if __name__ == "__main__":
    main()
