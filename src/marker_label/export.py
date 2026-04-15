"""Export labeled data to C3D and CSV."""

from __future__ import annotations

import csv
from collections.abc import Sequence

import numpy as np

from .io import save_c3d
from .constants import OBSTACLE_LABELS, UNLABELED_NUMERIC_LABEL_BASE


def build_full_trajectory_matrix(
    body_labels_static: list[str],
    points_dynamic_body: np.ndarray,
    label_per_frame_body: np.ndarray,
    points_obstacle: np.ndarray,
    obstacle_labels: list[str],
    *,
    loaded_indices_for_body: Sequence[int] | None = None,
    reference_frame: int = 0,
    unlabeled_numeric_base: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build (n_frames, n_markers, 3) and full label list: body (static order) + obstacle.

    Body markers missing in dynamic get NaN trajectory. Obstacle trajectories
    are appended.

    When ``loaded_indices_for_body`` is set, any body point with no anatomical label
    at ``reference_frame`` (empty string in ``label_per_frame_body``) is appended as an
    extra column whose name is ``str(loaded_indices_for_body[pi] + base)`` where ``base`` is
    ``unlabeled_numeric_base`` or ``UNLABELED_NUMERIC_LABEL_BASE`` (default 1: 1-based display
    to match common unlabeled-trial viewers). The underlying index is still 0-based in the
    loaded C3D (before initial screening). Sorted by loaded 0-based index for stable export.

    Parameters
    ----------
    body_labels_static : list of body labels (from static, defines order)
    points_dynamic_body : (n_frames, n_body_points, 3) remaining after obstacle removal
    label_per_frame_body : (n_frames, n_body_points) object array, label per point per frame
    points_obstacle : (n_frames, 2, 3) obstacle marker trajectories
    obstacle_labels : [OBSTACLE_L, OBSTACLE_R] or similar
    loaded_indices_for_body : length n_body; loaded-file column index per body column.
    reference_frame : frame used to detect unlabeled points (typically best labeling frame).
    unlabeled_numeric_base : added to loaded 0-based index for display (None = use constant).

    Returns
    -------
    points : (n_frames, n_markers, 3)
    labels : body_labels_static + obstacle_labels + optional numeric strings
    """
    n_frames = points_dynamic_body.shape[0]
    n_body = points_dynamic_body.shape[1]
    n_body_labels = len(body_labels_static)
    n_out = n_body_labels + len(obstacle_labels)
    points_out = np.full((n_frames, n_out, 3), np.nan)
    for li, label in enumerate(body_labels_static):
        for f in range(n_frames):
            for pi in range(label_per_frame_body.shape[1]):
                if label_per_frame_body[f, pi] == label:
                    points_out[f, li, :] = points_dynamic_body[f, pi, :]
                    break
    for oi, lab in enumerate(obstacle_labels):
        points_out[:, n_body_labels + oi, :] = points_obstacle[:, oi, :]
    labels_out = list(body_labels_static) + list(obstacle_labels)

    if loaded_indices_for_body is not None:
        if len(loaded_indices_for_body) != n_body:
            raise ValueError(
                f"loaded_indices_for_body length {len(loaded_indices_for_body)} "
                f"!= n_body {n_body}",
            )
        ref = max(0, min(n_frames - 1, int(reference_frame)))
        unlabeled_pi: list[int] = []
        for pi in range(n_body):
            lab = label_per_frame_body[ref, pi]
            s = str(lab).strip() if lab is not None else ""
            if s == "":
                unlabeled_pi.append(pi)
        unlabeled_pi.sort(key=lambda i: int(loaded_indices_for_body[i]))
        if unlabeled_pi:
            base = (
                UNLABELED_NUMERIC_LABEL_BASE
                if unlabeled_numeric_base is None
                else int(unlabeled_numeric_base)
            )
            extra = np.stack(
                [points_dynamic_body[:, pi, :] for pi in unlabeled_pi],
                axis=1,
            )
            points_out = np.concatenate([points_out, extra], axis=1)
            labels_out.extend(
                str(int(loaded_indices_for_body[pi]) + base) for pi in unlabeled_pi
            )

    return points_out, labels_out


def export_csv(
    path: str,
    points: np.ndarray,
    labels: list[str],
    rate: float = 0.0,
    first_frame: int = 1,
) -> None:
    """
    Export trajectories to CSV: rows = frames, columns = frame, time, {marker}_x, _y, _z.

    Parameters
    ----------
    path : output path
    points : (n_frames, n_points, 3)
    labels : length n_points
    rate : point frame rate (Hz)
    first_frame : first frame index
    """
    n_frames = points.shape[0]
    time = (np.arange(n_frames) + first_frame - 1) / rate if rate > 0 else np.arange(n_frames)
    cols = ["frame", "time"]
    for lab in labels:
        cols.append(f"{lab}_x")
        cols.append(f"{lab}_y")
        cols.append(f"{lab}_z")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n_frames):
            row = [i + first_frame - 1, round(time[i], 6)]
            for j in range(points.shape[1]):
                row.extend(
                    [
                        points[i, j, 0] if np.isfinite(points[i, j, 0]) else "",
                        points[i, j, 1] if np.isfinite(points[i, j, 1]) else "",
                        points[i, j, 2] if np.isfinite(points[i, j, 2]) else "",
                    ]
                )
            w.writerow(row)


def export_labeled(
    points: np.ndarray,
    labels: list[str],
    out_prefix: str,
    *,
    rate: float = 0.0,
    first_frame: int = 1,
    residual: np.ndarray | None = None,
    export_filled: bool = True,
    filled_points: np.ndarray | None = None,
    max_interp_frames: int = 10,
) -> None:
    """
    Export both original (with NaNs) and optionally filled C3D and CSV.

    Parameters
    ----------
    points : (n_frames, n_points, 3) original
    labels : length n_points
    out_prefix : e.g. "trial_01" -> trial_01_labeled.c3d, trial_01_labeled.csv, ...
    rate, first_frame : for C3D/CSV
    residual : optional (n_frames, n_points) for C3D
    export_filled : if True, also export filled versions
    filled_points : if provided, use this for filled export; else fill from points
    max_interp_frames : used only when filled_points is None and export_filled True
    """
    from .gap_fill import fill_gaps_all_markers

    # Original
    save_c3d(
        f"{out_prefix}_labeled.c3d",
        points,
        labels,
        rate=rate,
        first_frame=first_frame,
        residual=residual,
    )
    export_csv(f"{out_prefix}_labeled.csv", points, labels, rate=rate, first_frame=first_frame)

    if export_filled:
        if filled_points is not None:
            filled = filled_points
        else:
            filled = fill_gaps_all_markers(points, max_interp_frames=max_interp_frames)
        save_c3d(
            f"{out_prefix}_labeled_filled.c3d",
            filled,
            labels,
            rate=rate,
            first_frame=first_frame,
            residual=residual,
        )
        export_csv(
            f"{out_prefix}_labeled_filled.csv",
            filled,
            labels,
            rate=rate,
            first_frame=first_frame,
        )
