#!/usr/bin/env python3
"""
Run the labeling pipeline and export only the 4 head markers (LFHD, RFHD, LBHD, RBHD)
so you can verify head labeling before proceeding with full body.

Usage:
  python scripts/run_and_export_head_only.py <static.c3d> <dynamic.c3d> -o <out_prefix>

Example:
  python scripts/run_and_export_head_only.py "data/SUBJ01 Cal 01.c3d" "data/SUBJ01 Trial 05.c3d" -o out/SUBJ01_trial05_head_only

Uses check_screened_count=False so it runs even when dynamic has other than 41 channels.
Writes <out_prefix>_head_only.csv (and prints a short summary).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marker_label.constants import HEAD_MARKERS
from marker_label.pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Run pipeline and export only head markers (LFHD, RFHD, LBHD, RBHD).")
    ap.add_argument("static", help="Path to labeled static C3D")
    ap.add_argument("dynamic", help="Path to unlabeled dynamic C3D")
    ap.add_argument("-o", "--output", required=True, help="Output path prefix (e.g. out/SUBJ01_trial05_head_only)")
    ap.add_argument("--static-facing", choices=("x", "-x", "y", "-y"), default=None)
    ap.add_argument("--dynamic-facing", choices=("x", "-x", "y", "-y"), default=None)
    ap.add_argument("--no-filled", action="store_true", help="Do not export filled full output")
    args = ap.parse_args()

    out_prefix = args.output.rstrip("_")
    # Pipeline writes <out_prefix>_labeled.csv; we write <out_prefix>_head_only.csv
    info = run_pipeline(
        args.static,
        args.dynamic,
        out_prefix,
        static_facing_axis=args.static_facing,
        dynamic_facing_axis=args.dynamic_facing,
        export_filled=not args.no_filled,
        check_screened_count=False,
    )

    full_csv = out_prefix + "_labeled.csv"
    if not Path(full_csv).exists():
        print("Warning: full labeled CSV not found at", full_csv)
        return

    head_set = {m.upper() for m in HEAD_MARKERS}
    with open(full_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Keep frame, time, and columns for head markers only
    keep_cols = [0, 1]  # frame, time
    for i, col in enumerate(header):
        if i < 2:
            continue
        base = (col.rsplit("_", 1)[0] if "_" in col else col).strip().upper()
        if base in head_set:
            keep_cols.append(i)
    new_header = [header[i] for i in keep_cols]
    new_rows = [[row[i] for i in keep_cols] for row in rows]

    head_only_path = out_prefix + "_head_only.csv"
    with open(head_only_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(new_header)
        w.writerows(new_rows)

    print("Head-only output:", head_only_path)
    print("Columns:", ", ".join(new_header))
    print("Frames:", len(new_rows))
    print("Full pipeline output:", full_csv, "(and _filled if not --no-filled)")


if __name__ == "__main__":
    main()
