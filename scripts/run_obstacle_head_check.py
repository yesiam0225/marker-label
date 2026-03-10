#!/usr/bin/env python3
"""
Run the pipeline (screening → obstacle → head → C7/shoulders by Z, then Z-band),
then export obstacle + head + C7 + shoulders + CLAV + RBAK + STRN + T10 + arm + pelvis + arm/hand + leg/foot (41 markers) and open the QC viewer.

Usage:
  python scripts/run_obstacle_head_check.py <static.c3d> <dynamic.c3d> -o <out_prefix> [--no-viewer]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marker_label.constants import (
    HEAD_MARKERS,
    OBSTACLE_LABELS,
    C7_SHOULDER_MARKERS,
    CLAV_RBAK_MARKERS,
    STRN_T10_ARM_MARKERS,
    PELVIS_ARM12_MARKERS,
    LEG_FOOT12_MARKERS,
)
from marker_label.pipeline import run_pipeline
from marker_label.inspect_quality import load_labeled_csv
from marker_label.export import export_csv
from marker_label.io import save_c3d

# Obstacle + head + C7 + shoulders + CLAV + RBAK + STRN + T10 + arm + pelvis + arm/hand + leg/foot (41 markers)
OBSTACLE_HEAD_41 = (
    list(OBSTACLE_LABELS) + list(HEAD_MARKERS) + list(C7_SHOULDER_MARKERS)
    + list(CLAV_RBAK_MARKERS) + list(STRN_T10_ARM_MARKERS) + list(PELVIS_ARM12_MARKERS) + list(LEG_FOOT12_MARKERS)
)


def extract_subset_from_labeled_csv(
    full_csv_path: str,
    out_path_prefix: str,
    marker_names: list[str],
    *,
    rate: float = 0.0,
    first_frame: int = 1,
) -> None:
    """Load full labeled CSV, keep only given marker columns, write CSV and C3D."""
    points, labels, rate_in = load_labeled_csv(full_csv_path)
    if rate_in > 0:
        rate = rate_in
    indices = []
    out_labels = []
    for name in marker_names:
        found = False
        for i, lab in enumerate(labels):
            if (lab or "").strip().upper() == name.upper():
                indices.append(i)
                out_labels.append((lab or "").strip() or name)
                found = True
                break
        if not found:
            raise ValueError(f"Could not find marker '{name}' in {full_csv_path}. Expected {marker_names}.")
    pts_sub = points[:, indices, :]
    export_csv(f"{out_path_prefix}.csv", pts_sub, out_labels, rate=rate, first_frame=first_frame)
    save_c3d(f"{out_path_prefix}.c3d", pts_sub, out_labels, rate=rate, first_frame=first_frame)
    print("Wrote", f"{out_path_prefix}.csv", "and .c3d", f"({len(marker_names)} markers).")


def extract_obstacle_head_from_labeled_csv(
    full_csv_path: str,
    out_prefix: str,
    *,
    rate: float = 0.0,
    first_frame: int = 1,
) -> None:
    """Export obstacle + head + C7 + shoulders + CLAV + RBAK + STRN/T10/arm + pelvis/arm12 + leg/foot (41 markers) from full labeled CSV."""
    extract_subset_from_labeled_csv(
        full_csv_path,
        f"{out_prefix}_obstacle_head_c7_shoulders_labeled",
        OBSTACLE_HEAD_41,
        rate=rate,
        first_frame=first_frame,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run pipeline, export obstacle+head only, and open viewer to check labels.",
    )
    ap.add_argument("static", help="Path to labeled static C3D")
    ap.add_argument("dynamic", help="Path to unlabeled dynamic C3D")
    ap.add_argument("-o", "--output", required=True, help="Output path prefix (e.g. out/BBA01_trial05)")
    ap.add_argument("--static-facing", choices=("x", "-x", "y", "-y"), default=None)
    ap.add_argument("--dynamic-facing", choices=("x", "-x", "y", "-y"), default=None)
    ap.add_argument("--no-viewer", action="store_true", help="Do not open the viewer after export")
    ap.add_argument("--no-filled", action="store_true", help="Do not export filled full output")
    args = ap.parse_args()

    out_prefix = args.output.rstrip("_")
    run_pipeline(
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
        print("Expected", full_csv, "not found.")
        sys.exit(1)

    # Get first_frame from first row of CSV
    import csv as csv_module
    with open(full_csv, newline="") as f:
        r = csv_module.reader(f)
        next(r)  # header
        row0 = next(r, None)
    first_frame = 1
    if row0 and len(row0) >= 1:
        try:
            first_frame = int(float(row0[0]))
        except (ValueError, TypeError):
            pass

    extract_obstacle_head_from_labeled_csv(full_csv, out_prefix, first_frame=first_frame)

    if not args.no_viewer:
        viewer_path = out_prefix + "_obstacle_head_c7_shoulders_labeled.c3d"
        if not Path(viewer_path).exists():
            viewer_path = out_prefix + "_obstacle_head_c7_shoulders_labeled.csv"
        print("Opening viewer:", viewer_path)
        from marker_label.qc_viewer import run_viewer
        run_viewer(viewer_path)


if __name__ == "__main__":
    main()
