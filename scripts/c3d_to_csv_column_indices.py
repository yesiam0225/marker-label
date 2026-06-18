#!/usr/bin/env python3
"""
Export a C3D to CSV with ``frame``, ``time``, and ``{marker}_x/y/z`` columns.

By default, marker names come from the C3D point labels (same as ``load_c3d``).
Empty / missing names fall back to 1-based column indices (or 0-based with
``--zero-based``). Use ``--numeric-labels`` to force index-only names (legacy
behavior for unlabeled-pipeline workflows).

Column headers are ``{name}_x``, ``{name}_y``, ``{name}_z``.
Prints 0-based index -> label used in CSV to stderr for cross-check.

Usage (from repo root):
  PYTHONPATH=src python scripts/c3d_to_csv_column_indices.py data/SUBJ03\\ Trial\\ 06.c3d -o out/trial06.csv
"""

from __future__ import annotations

import argparse
import sys


def _build_csv_labels(raw_labels: list[str], n: int, *, numeric_only: bool, zero_based: bool, base: int) -> list[str]:
    """Use C3D names when present; uniquify duplicates; else numeric fallback."""
    if numeric_only:
        b = 0 if zero_based else base
        return [str(i + b) for i in range(n)]

    out: list[str] = []
    for i in range(n):
        lab = ""
        if i < len(raw_labels) and raw_labels[i] is not None:
            lab = str(raw_labels[i]).strip()
        if not lab:
            lab = str(i + (0 if zero_based else base))
        out.append(lab)

    seen: dict[str, int] = {}
    unique: list[str] = []
    for lab in out:
        if lab not in seen:
            seen[lab] = 0
            unique.append(lab)
        else:
            seen[lab] += 1
            unique.append(f"{lab}__dup{seen[lab]}")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export C3D trajectories to CSV (C3D point labels as column stems by default).",
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
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="Fallback numeric column names are 0,1,... (default: 1-based when name missing).",
    )
    parser.add_argument(
        "--numeric-labels",
        action="store_true",
        help="Force marker columns to numeric indices only (ignore C3D point names).",
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
    base = UNLABELED_NUMERIC_LABEL_BASE
    labels = _build_csv_labels(
        list(raw_labels),
        n,
        numeric_only=bool(args.numeric_labels),
        zero_based=bool(args.zero_based),
        base=base,
    )

    print("Column index (0-based) -> C3D label -> CSV stem:", file=sys.stderr)
    for i in range(n):
        raw = raw_labels[i] if i < len(raw_labels) else ""
        print(f"  {i}: {raw!r} -> {labels[i]!r}", file=sys.stderr)

    export_csv(out, pts, labels, rate=rate, first_frame=first_frame)
    print(f"Wrote {out} ({pts.shape[0]} frames, {n} markers).", file=sys.stderr)


if __name__ == "__main__":
    main()
