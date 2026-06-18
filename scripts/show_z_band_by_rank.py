#!/usr/bin/env python3
"""
Show Z-band-by-rank: band structure and, for a dynamic trial, which point indices
fall in each band when bands are assigned by Z rank (highest Z → band 11, etc.).

Usage:
  python scripts/show_z_band_by_rank.py [dynamic.c3d]
  (Optional) With one argument, use that dynamic C3D; else data/SUBJ01 Trial 05.c3d
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from marker_label.io import load_c3d
from marker_label.constants import WHOLE_BODY_39, Z_BAND_LABELS, N_Z_BANDS
from marker_label.obstacle import detect_obstacle_markers
from marker_label.body_labeling import (
    best_frame_heel_contact,
    _point_band_indices_by_z_rank,
    match_markers_to_template_z_bands,
    build_template_from_static,
)
from marker_label.constants import Z_BAND_LABEL_TO_BAND


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    dynamic_path = base / "data" / "SUBJ01 Trial 05.c3d"
    if len(sys.argv) >= 2:
        dynamic_path = Path(sys.argv[1])
    if not dynamic_path.exists():
        print(f"File not found: {dynamic_path}")
        sys.exit(1)

    print("=" * 72)
    print("Z-BAND BY RANK (band index and label set; no static Z)")
    print("=" * 72)
    print("Valid points are sorted by Z descending. First size[11] points → band 11,")
    print("next size[10] → band 10, ... Band sizes from Z_BAND_LABELS.")
    print()

    band_sizes = [len(Z_BAND_LABELS[b]) for b in range(N_Z_BANDS)]
    print("Band (high Z → low Z)  Size  Labels")
    print("-" * 72)
    for b in range(N_Z_BANDS - 1, -1, -1):
        labels = list(Z_BAND_LABELS[b])
        print(f"  Band {b:2d} (rank)        {band_sizes[b]:2d}    {labels}")
    print(f"  Total slots: {sum(band_sizes)}")
    print()

    # Load dynamic, get body points and best frame
    dynamic = load_c3d(str(dynamic_path))
    points_d = dynamic["points"]
    residual_d = dynamic.get("residual")
    n_frames_d, n_pts_d, _ = points_d.shape
    visible_frac = np.sum(np.isfinite(points_d).all(axis=2), axis=0) / n_frames_d
    keep_vis = np.where(visible_frac > 0.5)[0]
    if len(keep_vis) < n_pts_d:
        points_d = points_d[:, keep_vis, :]
        if residual_d is not None:
            residual_d = residual_d[:, keep_vis]
    obstacle_indices, _ = detect_obstacle_markers(points_d, visibility_min=0.80)
    if obstacle_indices:
        keep = [i for i in range(points_d.shape[1]) if i not in obstacle_indices]
        points_d_body = points_d[:, keep, :]
        residual_d_body = residual_d[:, keep] if residual_d is not None else None
    else:
        points_d_body = points_d
        residual_d_body = residual_d

    best_f = best_frame_heel_contact(
        points_d_body,
        residual_d_body,
        middle_start=0.20,
        middle_end=0.80,
        visibility_min_fraction=0.9,
        walking_axis_x=True,
    )
    pts_f = points_d_body[best_f]

    point_bands = _point_band_indices_by_z_rank(pts_f, band_sizes=band_sizes, n_bands=N_Z_BANDS)

    # Build template and get assignments so we can show label per point
    static_path = base / "data" / "SUBJ01 Cal 01.c3d"
    if static_path.exists():
        static = load_c3d(str(static_path))
        labels_s = static["labels"]
        points_s = static["points"]
        whole_set = {m.upper() for m in WHOLE_BODY_39}
        body_labels_static = [l for l in labels_s if l.strip().upper() in whole_set]
        if not body_labels_static:
            body_labels_static = list(labels_s)
        keep_static = [i for i in range(len(labels_s)) if labels_s[i] in body_labels_static]
        points_s_body = points_s[:, keep_static, :] if keep_static else points_s
        labels_s_body = [labels_s[i] for i in keep_static] if keep_static else labels_s
        template, _, _ = build_template_from_static(points_s_body, labels_s_body, use_pelvis_frame=False)
        template_body = {k: v for k, v in template.items() if k in body_labels_static}
        from marker_label.body_labeling import get_lr_ap_axes_from_walking, compute_walking_direction_x
        wdx = compute_walking_direction_x(points_d_body)
        d_back, d_right = get_lr_ap_axes_from_walking(wdx, left_side_positive_lr=False)
        centroid_xy = np.nanmean(np.nanmean(points_d_body[:, :, :2], axis=1), axis=0).astype(np.float64)
        lr_ap = (d_back, d_right, centroid_xy)
        assignments = match_markers_to_template_z_bands(
            pts_f, template_body,
            z_band_by_rank=True,
            match_head_first=True,
            walking_axis_x=True,
            left_side_positive_lr=False,
            lr_ap_from_walking=lr_ap,
        )
        pt_to_lab = {pi: lab for pi, lab in assignments}
    else:
        pt_to_lab = {}

    print("=" * 72)
    print(f"Dynamic frame (best frame {best_f}): point index → band by Z rank")
    print("=" * 72)
    for b in range(N_Z_BANDS - 1, -1, -1):
        in_b = np.where(point_bands == b)[0]
        if len(in_b) == 0:
            print(f"  Band {b:2d}: (no points)")
            continue
        z_vals = [float(pts_f[i, 2]) for i in in_b]
        order = np.argsort(z_vals)[::-1]
        in_b_ordered = in_b[order]
        labels_in_b = list(Z_BAND_LABELS[b])
        parts = []
        for j, pi in enumerate(in_b_ordered):
            z = pts_f[pi, 2]
            lab = pt_to_lab.get(int(pi), "?")
            parts.append(f"pt{pi}(z={z:.0f}={lab})")
        print(f"  Band {b:2d}: {', '.join(parts)}")
    print()
    unassigned = np.where(point_bands == -1)[0]
    if len(unassigned) > 0:
        print(f"  Not in any band (Z rank beyond top {sum(band_sizes)}): {list(unassigned)}")


if __name__ == "__main__":
    main()
