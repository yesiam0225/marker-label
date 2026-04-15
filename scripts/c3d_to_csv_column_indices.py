#!/usr/bin/env python3
"""
Export an unlabeled C3D to CSV; marker column names default to **1-based** index
(``UNLABELED_NUMERIC_LABEL_BASE``), matching labeled-pipeline unlabeled marker names.
Use ``--zero-based`` for raw 0-based indices (same as ``load_c3d`` column index).

Column headers are ``{name}_x``, ``{name}_y``, ``{name}_z``.
Prints 0-based index -> C3D label (if any) to stderr for cross-check.

Usage (from repo root):
  PYTHONPATH=src python scripts/c3d_to_csv_column_indices.py data/BBC10\\ Trial\\ 06.c3d -o out/trial06_raw_by_index.csv
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export C3D trajectories to CSV; marker names are 0-based column indices.",
    )
    parser.add_argument("c3d", help="Input C3D path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: input path with .csv extension)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Coordinate scale after load (default 1; use 1000 if file is in meters -> mm)",
    )
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = str(args.c3d).rsplit(".", 1)[0] + ".csv"

    from marker_label.io import load_c3d
    from marker_label.export import export_csv
    from marker_label.constants import UNLABELED_NUMERIC_LABEL_BASE

    data = load_c3d(args.c3d, scale_factor=args.scale)
    pts = data["points"]
    raw_labels = data.get("labels") or []
    rate = float(data.get("rate") or 0.0)
    first_frame = int(data.get("first_frame") or 1)

    n = pts.shape[1]
    base = 0 if args.zero_based else UNLABELED_NUMERIC_LABEL_BASE
    labels = [str(i + base) for i in range(n)]

    print("Column index (0-based) -> label in C3D file:", file=sys.stderr)
    for i in range(n):
        lab = raw_labels[i] if i < len(raw_labels) else ""
        print(f"  {i}: {lab!r}", file=sys.stderr)

    export_csv(out, pts, labels, rate=rate, first_frame=first_frame)
    print(f"Wrote {out} ({pts.shape[0]} frames, {n} markers).", file=sys.stderr)


if __name__ == "__main__":
    main()
