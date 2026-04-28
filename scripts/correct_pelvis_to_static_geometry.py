#!/usr/bin/env python3
"""
Replace LASI/RASI/LPSI/RPSI in a labeled CSV with a rigid (Kabsch) copy of the static
pelvis template, fitted per frame to the current four points.

Use for frames where inter-marker geometry is far from the static (e.g. after bad
propagation). Does not correct wrong label-to-column assignment.

Example:
  PYTHONPATH=src python scripts/correct_pelvis_to_static_geometry.py \\
    "data/BBA02 Trial 26.c3d" "data/BBA02 Trial 82_bf216_labeled.csv" \\
    -o "data/BBA02 Trial 82_bf216_labeled_pelvisfix.csv" --rmse-threshold 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from marker_label.body_labeling import build_template_from_static
from marker_label.io import load_c3d
from marker_label.pelvis import rigid_pelvis_from_template_to_observed

PELV = ("LASI", "RASI", "LPSI", "RPSI")


def _template_pelvis_4x3(static_path: str) -> np.ndarray:
    s = load_c3d(str(static_path), 1.0)
    template, _, _ = build_template_from_static(s["points"], s["labels"], use_pelvis_frame=False)
    rows = []
    for name in PELV:
        v = None
        for k, val in template.items():
            if str(k).strip().upper() == name:
                a = np.asarray(val, dtype=np.float64).ravel()[:3]
                if a.size == 3 and bool(np.isfinite(a).all()):
                    v = a
                break
        if v is None:
            raise SystemExit(f"Static {static_path!r} has no finite template for {name}.")
        rows.append(v)
    return np.stack(rows, axis=0)


def _dist_rms(d_ref: np.ndarray, p4: np.ndarray) -> float:
    d = np.linalg.norm(p4[:, None, :] - p4[None, :, :], axis=-1)
    return float(np.sqrt(np.mean((d_ref - d) ** 2)))


def _column_triplets(df: pd.DataFrame) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    for p in PELV:
        for c in df.columns:
            if c.strip().upper().startswith(p) and c.endswith("_x"):
                out[p] = (c, c.replace("_x", "_y"), c.replace("_x", "_z"))
                break
        if p not in out:
            raise SystemExit(f"Column for {p}_x not found in CSV.")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("static_c3d", type=Path, help="Labeled static C3D (pelvis 4 in template).")
    p.add_argument("labeled_csv", type=Path, help="Labeled export CSV to correct.")
    p.add_argument(
        "-o", "--output", type=Path, required=True, help="Output CSV path."
    )
    p.add_argument(
        "--rmse-threshold",
        type=float,
        default=15.0,
        help="Only replace pelvis when dist-matrix RMSE to static (mm) exceeds this (default 15).",
    )
    p.add_argument(
        "--all-frames",
        action="store_true",
        help="Replace pelvis in every row with all four finite; ignore RMSE threshold.",
    )
    args = p.parse_args()

    tpl4 = _template_pelvis_4x3(str(args.static_c3d))
    d_ref = np.linalg.norm(tpl4[:, None, :] - tpl4[None, :, :], axis=-1)
    df = pd.read_csv(args.labeled_csv)
    trip = _column_triplets(df)

    n_in = 0
    n_replace = 0
    for i in range(len(df)):
        p4 = np.zeros((4, 3), dtype=np.float64)
        ok = True
        for j, name in enumerate(PELV):
            a, b, c = trip[name]
            for k, col in enumerate((a, b, c)):
                v = df.at[i, col]
                if not np.isfinite(v):
                    ok = False
                    break
                p4[j, k] = v
        if not ok:
            continue
        n_in += 1
        rms = _dist_rms(d_ref, p4)
        if not args.all_frames and rms <= float(args.rmse_threshold):
            continue
        out4 = rigid_pelvis_from_template_to_observed(p4, tpl4)
        if not bool(np.isfinite(out4).all()):
            continue
        for j, name in enumerate(PELV):
            a, b, c = trip[name]
            df.at[i, a] = out4[j, 0]
            df.at[i, b] = out4[j, 1]
            df.at[i, c] = out4[j, 2]
        n_replace += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(
        f"Wrote {args.output}  (rows with finite pelvis: {n_in}, rows replaced: {n_replace}, "
        f"threshold={args.rmse_threshold if not args.all_frames else 'N/A (all)'}).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
