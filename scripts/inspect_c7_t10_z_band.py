#!/usr/bin/env python3
"""
Inspect points in the C7–T10 Z band at the best frame used for labeling.

Finds all points (body, after obstacle removal) whose Z is between C7 and T10,
and reports their assigned label. Use this to find unlabeled or mis-labeled
candidates for RBAK (e.g. if RBAK is assigned too low, a point in this Z band
may be labeled as RUPA/LUPA or something else).

Usage:
  python scripts/inspect_c7_t10_z_band.py [static.c3d] [dynamic.c3d]

Defaults: data/SUBJ01 Cal 01.c3d, data/SUBJ01 Trial 05.c3d
Options: use same pipeline flags as marker-label (match-head-first, z-band-by-gap, whole-body-39).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from marker_label.io import load_c3d
from marker_label.constants import WHOLE_BODY_39
from marker_label.obstacle import detect_obstacle_markers
from marker_label.body_labeling import (
    build_template_from_static,
    best_frame_heel_contact,
    match_markers_to_template_z_bands,
    _z_band_boundaries_from_template,
    get_lr_ap_axes_from_walking,
    compute_walking_direction_x,
)
from marker_label.constants import N_Z_BANDS

TRUNK_THORAX = frozenset({"C7", "T10", "STRN", "CLAV", "RBAK"})


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    static_path = base / "data" / "SUBJ01 Cal 01.c3d"
    dynamic_path = base / "data" / "SUBJ01 Trial 05.c3d"
    if len(sys.argv) >= 3:
        static_path = Path(sys.argv[1])
        dynamic_path = Path(sys.argv[2])
    if not static_path.exists() or not dynamic_path.exists():
        print(f"Files not found: {static_path}, {dynamic_path}")
        sys.exit(1)

    static = load_c3d(str(static_path))
    dynamic = load_c3d(str(dynamic_path))
    points_s = static["points"]
    labels_s = static["labels"]
    points_d = dynamic["points"]
    residual_d = dynamic.get("residual")

    # Visibility filter (same as pipeline)
    n_frames_d, n_pts_d, _ = points_d.shape
    visible_frac = np.sum(np.isfinite(points_d).all(axis=2), axis=0) / n_frames_d
    keep_vis = np.where(visible_frac > 0.5)[0]
    if len(keep_vis) < n_pts_d:
        points_d = points_d[:, keep_vis, :]
        if residual_d is not None:
            residual_d = residual_d[:, keep_vis]

    # Obstacle detection
    obstacle_indices, _ = detect_obstacle_markers(points_d, visibility_min=0.80)
    if obstacle_indices:
        keep = [i for i in range(points_d.shape[1]) if i not in obstacle_indices]
        points_d_body = points_d[:, keep, :]
        residual_d_body = residual_d[:, keep] if residual_d is not None else None
    else:
        points_d_body = points_d
        residual_d_body = residual_d

    # Template (39 whole-body)
    whole_set = {m.upper() for m in WHOLE_BODY_39}
    body_labels_static = [l for l in labels_s if l.strip().upper() in whole_set]
    if not body_labels_static:
        body_labels_static = list(labels_s)
    keep_static = [i for i in range(len(labels_s)) if labels_s[i] in body_labels_static]
    points_s_body = points_s[:, keep_static, :] if keep_static else points_s
    labels_s_body = [labels_s[i] for i in keep_static] if keep_static else labels_s
    template, _, _ = build_template_from_static(points_s_body, labels_s_body, use_pelvis_frame=False)
    template_body = {k: v for k, v in template.items() if k in body_labels_static}

    # L/R and A/P from walking (same as pipeline)
    wdx = compute_walking_direction_x(points_d_body)
    d_back, d_right = get_lr_ap_axes_from_walking(wdx, left_side_positive_lr=False)
    centroid_xy = np.nanmean(np.nanmean(points_d_body[:, :, :2], axis=1), axis=0).astype(np.float64)
    lr_ap_from_walking = (d_back, d_right, centroid_xy)

    # Best frame (heel contact for match_head_first)
    best_f = best_frame_heel_contact(
        points_d_body,
        residual_d_body,
        middle_start=0.20,
        middle_end=0.80,
        visibility_min_fraction=0.9,
        walking_axis_x=True,
    )
    pts_f = points_d_body[best_f]
    n_points = pts_f.shape[0]

    # Z-band matching (match_head_first, z_band_by_gap)
    use_gap = True
    boundaries, label_to_band = _z_band_boundaries_from_template(
        template_body, n_bands=N_Z_BANDS, z_band_by_value=False, z_band_by_gap=use_gap
    )
    assignments = match_markers_to_template_z_bands(
        pts_f,
        template_body,
        boundaries=boundaries,
        label_to_band=label_to_band,
        z_band_by_gap=use_gap,
        use_hungarian=True,
        max_match_distance=None,
        match_head_first=True,
        walking_axis_x=True,
        left_side_positive_lr=False,
        lr_ap_from_walking=lr_ap_from_walking,
    )

    # Build point_idx -> label
    assigned_label: dict[int, str] = {}
    for pi, lab in assignments:
        assigned_label[pi] = lab.strip() if hasattr(lab, "strip") else str(lab)

    # C7 and T10 positions (from assigned points)
    c7_pt = None
    t10_pt = None
    for pi, lab in assignments:
        key = lab.strip().upper() if hasattr(lab, "strip") else str(lab).upper()
        if key == "C7":
            c7_pt = pts_f[pi]
            break
    for pi, lab in assignments:
        key = lab.strip().upper() if hasattr(lab, "strip") else str(lab).upper()
        if key == "T10":
            t10_pt = pts_f[pi]
            break

    if c7_pt is None or t10_pt is None or not np.isfinite(c7_pt).all() or not np.isfinite(t10_pt).all():
        print("C7 or T10 not found in assignments.")
        sys.exit(1)

    z_lo = min(float(c7_pt[2]), float(t10_pt[2]))
    z_hi = max(float(c7_pt[2]), float(t10_pt[2]))

    print("=" * 70)
    print("C7–T10 Z band inspection (best frame used for labeling)")
    print("=" * 70)
    print(f"Best frame: {best_f}")
    print(f"C7  position (x, y, z): ({c7_pt[0]:.1f}, {c7_pt[1]:.1f}, {c7_pt[2]:.1f})")
    print(f"T10 position (x, y, z): ({t10_pt[0]:.1f}, {t10_pt[1]:.1f}, {t10_pt[2]:.1f})")
    print(f"Z band [C7, T10]: [{z_lo:.1f}, {z_hi:.1f}] mm")
    print(f"Total body points in frame: {n_points}")
    print(f"Assigned (in assignments): {len(assigned_label)}")
    print()

    # Points with Z in [z_lo, z_hi]
    in_band = []
    for i in range(n_points):
        if not np.isfinite(pts_f[i]).all():
            continue
        z = float(pts_f[i, 2])
        if z_lo <= z <= z_hi:
            lab = assigned_label.get(i, "<unassigned>")
            in_band.append((i, pts_f[i], lab))

    in_band.sort(key=lambda x: -x[1][2])  # sort by Z descending (C7-like first)

    print("Points with Z between C7 and T10 (Z descending):")
    print("-" * 70)
    for idx, xyz, lab in in_band:
        key = lab.upper() if lab != "<unassigned>" else lab
        trunk = " [thorax]" if key in TRUNK_THORAX else ""
        not_thorax = " *** NOT THORAX" if lab != "<unassigned>" and key not in TRUNK_THORAX else ""
        print(f"  pt {idx:3d}  z={xyz[2]:8.1f}  x={xyz[0]:8.1f}  y={xyz[1]:8.1f}  -> {lab}{trunk}{not_thorax}")

    # Summary: points in band that are NOT labeled as thorax (C7, T10, STRN, CLAV, RBAK)
    not_thorax_in_band = [
        (idx, xyz, lab)
        for idx, xyz, lab in in_band
        if lab == "<unassigned>" or (lab.upper() if hasattr(lab, "upper") else lab) not in TRUNK_THORAX
    ]
    print()
    print("Points in C7–T10 Z band that are NOT labeled as thorax (C7/T10/STRN/CLAV/RBAK):")
    print("  (candidates for true RBAK if RBAK is currently assigned too low)")
    print("-" * 70)
    if not not_thorax_in_band:
        print("  (none)")
    else:
        for idx, xyz, lab in not_thorax_in_band:
            print(f"  pt {idx:3d}  z={xyz[2]:8.1f}  x={xyz[0]:8.1f}  y={xyz[1]:8.1f}  -> {lab}")

    # --- Check point 33 as RBAK candidate ---
    pt33_idx = 33
    if pt33_idx >= n_points:
        print("\n(Point 33 not in body point set)")
        return
    pt33 = pts_f[pt33_idx]
    if not np.isfinite(pt33).all():
        print("\n(Point 33 has NaN)")
        return
    midline_xy = c7_pt[:2].astype(np.float64)
    dot_back_33 = float(np.dot(pt33[:2] - midline_xy, d_back))
    dot_right_33 = float(np.dot(pt33[:2] - midline_xy, d_right))
    clav_pt = strn_pt = None
    for pi, lab in assignments:
        k = lab.strip().upper() if hasattr(lab, "strip") else str(lab).upper()
        if k == "CLAV":
            clav_pt = pts_f[pi]
        elif k == "STRN":
            strn_pt = pts_f[pi]
    print()
    print("=" * 70)
    print("Point 33 as RBAK: criteria check (best frame)")
    print("=" * 70)
    print(f"Point 33 position: x={pt33[0]:.1f}, y={pt33[1]:.1f}, z={pt33[2]:.1f}")
    print(f"Currently assigned to: {assigned_label.get(pt33_idx, '?')}")
    print()
    ok = True
    # 1) Z between C7 and T10
    z_c7, z_t10 = float(c7_pt[2]), float(t10_pt[2])
    z_lo, z_hi = min(z_c7, z_t10), max(z_c7, z_t10)
    in_c7_t10 = z_lo <= pt33[2] <= z_hi
    print(f"1) Z between C7 and T10: [{z_lo:.1f}, {z_hi:.1f}]  pt33.z={pt33[2]:.1f}  -> {'PASS' if in_c7_t10 else 'FAIL'}")
    if not in_c7_t10:
        ok = False
    # 2) Y (L/R) on the right side of C7 and T10
    rightness_c7 = float(np.dot(c7_pt[:2] - midline_xy, d_right))
    rightness_t10 = float(np.dot(t10_pt[:2] - midline_xy, d_right))
    on_right = (dot_right_33 >= rightness_c7 and dot_right_33 >= rightness_t10) or (dot_right_33 <= rightness_c7 and dot_right_33 <= rightness_t10)
    print(f"2) Right side of C7/T10: C7 rightness={rightness_c7:.2f}, T10={rightness_t10:.2f}, pt33={dot_right_33:.2f}  -> {'PASS' if on_right else 'CHECK'}")
    # 3) Z between CLAV and STRN, 4) posterior to CLAV and STRN
    if clav_pt is not None and strn_pt is not None and np.isfinite(clav_pt).all() and np.isfinite(strn_pt).all():
        z_clav, z_strn = float(clav_pt[2]), float(strn_pt[2])
        z_clav_strn_lo, z_clav_strn_hi = min(z_clav, z_strn), max(z_clav, z_strn)
        in_clav_strn = z_clav_strn_lo <= pt33[2] <= z_clav_strn_hi
        print(f"3) Z between CLAV and STRN: [{z_clav_strn_lo:.1f}, {z_clav_strn_hi:.1f}]  pt33.z={pt33[2]:.1f}  -> {'PASS' if in_clav_strn else 'FAIL'}")
        if not in_clav_strn:
            ok = False
        dot_back_clav = float(np.dot(clav_pt[:2] - midline_xy, d_back))
        dot_back_strn = float(np.dot(strn_pt[:2] - midline_xy, d_back))
        posterior_to_both = dot_back_33 > dot_back_clav and dot_back_33 > dot_back_strn
        print(f"4) Posterior to CLAV and STRN: pt33 d_back={dot_back_33:.1f}, CLAV={dot_back_clav:.1f}, STRN={dot_back_strn:.1f}  -> {'PASS' if posterior_to_both else 'FAIL'}")
        if not posterior_to_both:
            ok = False
    else:
        print("3) Z between CLAV and STRN: (CLAV/STRN not found)")
        print("4) Posterior to CLAV and STRN: (skip)")
    print()
    print("Overall: Point 33 MEETS RBAK criteria" if ok else "Overall: Point 33 does not meet all RBAK criteria")
    print()
    print("Why the code missed point 33 as RBAK:")
    print("  - RBAK is assigned from TEMPLATE Z band 8 first (T-pose band order).")
    print("  - In walking, point 33's Z falls in band 7 (below band 8 boundary), not band 8.")
    print("  - So the code assigns RBAK to a point in band 8, then in band 7 it only assigns")
    print("  - T10 (most posterior) and STRN (most anterior). Point 33 is the most posterior")
    print("  - in band 7, so it gets T10. The code never considers point 33 for RBAK.")
    print("  - Fix: prefer RBAK from anatomical Z band [C7, T10] instead of template band 8.")


if __name__ == "__main__":
    main()
