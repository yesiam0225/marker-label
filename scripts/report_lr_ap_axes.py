#!/usr/bin/env python3
"""
Report how L/R and A/P are defined in terms of X and Y for a dynamic trial.
Uses the same logic as the pipeline: walking direction from centroid X trend,
then get_lr_ap_axes_from_walking.
"""
from __future__ import annotations

import sys
import numpy as np

from marker_label.io import load_c3d
from marker_label.body_labeling import compute_walking_direction_x, get_lr_ap_axes_from_walking


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/BBA01 Trial 05.c3d"
    points = load_c3d(path)["points"]  # (n_frames, n_pts, 3)

    wdx = compute_walking_direction_x(points)
    if wdx > 0:
        walk = "+X (subject walks toward increasing X)"
    elif wdx < 0:
        walk = "-X (subject walks toward decreasing X)"
    else:
        walk = "unclear (centroid X trend < 5 mm/frame)"

    n_frames = points.shape[0]
    valid = np.isfinite(points).all(axis=2)
    centroid_x = np.array([np.nanmean(points[t, valid[t], 0]) for t in range(n_frames) if np.any(valid[t])])
    if len(centroid_x) >= 2:
        mean_dx = np.nanmean(np.diff(centroid_x))
    else:
        mean_dx = np.nan

    print(f"Trial: {path}")
    print(f"Frames: {points.shape[0]}, Points: {points.shape[1]}")
    print(f"Centroid X: first={centroid_x[0]:.1f}, last={centroid_x[-1]:.1f} mm (mean dX = {mean_dx:.2f} mm/frame)")
    print(f"Walking direction (pipeline): {walk}")
    print()

    for left_side_positive_lr in (False, True):
        d_back, d_right = get_lr_ap_axes_from_walking(wdx, left_side_positive_lr=left_side_positive_lr)
        flag = " --left-side-positive-y" if left_side_positive_lr else " (default)"
        print(f"With left_side_positive_lr={left_side_positive_lr}{flag}:")
        print(f"  d_back  = [{d_back[0]:+.1f}, {d_back[1]:+.1f}]  (posterior direction)")
        print(f"  d_right = [{d_right[0]:+.1f}, {d_right[1]:+.1f}]  (right side direction)")
        if abs(d_back[0]) > 0.9:
            if d_back[0] < 0:
                print(f"  A/P: anterior = larger X (forward), posterior = smaller X (back)")
            else:
                print(f"  A/P: anterior = smaller X (forward), posterior = larger X (back)")
        else:
            if d_back[1] < 0:
                print(f"  A/P: anterior = larger Y, posterior = smaller Y")
            else:
                print(f"  A/P: anterior = smaller Y, posterior = larger Y")
        if abs(d_right[1]) > 0.9:
            if d_right[1] > 0:
                print(f"  L/R: right = larger Y, left = smaller Y")
            else:
                print(f"  L/R: right = smaller Y, left = larger Y")
        else:
            if d_right[0] > 0:
                print(f"  L/R: right = larger X, left = smaller X")
            else:
                print(f"  L/R: right = smaller X, left = larger X")
        if left_side_positive_lr:
            print(f"  (With --left-side-positive-y the pipeline treats positive L/R axis as Left, so Left = larger Y, Right = smaller Y)")
        print()

    print("Head assignment in _match_head_markers_first (current code):")
    print("  Fore/Back (A/P): by X — larger X = anterior (LFHD, RFHD), smaller X = posterior (LBHD, RBHD)")
    print("  Left/Right: by Y — left_side_positive_lr True => larger Y = Left; False => smaller Y = Left")
    print("  So with --left-side-positive-y: larger Y = Left (LFHD, LBHD), smaller Y = Right (RFHD, RBHD)")


if __name__ == "__main__":
    main()
