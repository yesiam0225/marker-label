"""Obstacle marker detection by stationarity.

Extra stationary column dropping (``screened_indices_extra_stationary_to_drop``) is part of the
standard pipeline when two obstacles exist; disabling it is for exceptional cases — see README.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import OBSTACLE_LABELS, DEFAULT_OBSTACLE_VISIBILITY_MIN


def motion_score_per_marker(points: np.ndarray) -> np.ndarray:
    """
    Compute a motion score for each marker (higher = more motion).

    Uses mean velocity magnitude over time (finite differences).

    Parameters
    ----------
    points : (n_frames, n_points, 3)

    Returns
    -------
    (n_points,) float, NaN for markers with insufficient valid data
    """
    n_frames, n_points, _ = points.shape
    velocity = np.full((n_frames - 1, n_points), np.nan)
    for i in range(n_frames - 1):
        d = points[i + 1] - points[i]
        valid = np.isfinite(points[i]).all(axis=1) & np.isfinite(points[i + 1]).all(axis=1)
        vel_mag = np.linalg.norm(d, axis=1)
        velocity[i] = np.where(valid, vel_mag, np.nan)
    # Mean velocity (ignoring NaN); avoid "Mean of empty slice" when a column has no finite values
    mean_vel = np.full(n_points, np.nan)
    with np.errstate(invalid="ignore"):
        for j in range(n_points):
            col = velocity[:, j]
            if np.any(np.isfinite(col)):
                mean_vel[j] = np.nanmean(col)
    return mean_vel


def visibility_fraction(points: np.ndarray) -> np.ndarray:
    """Fraction of frames where marker is valid (non-NaN). Shape (n_points,)."""
    valid = np.isfinite(points).all(axis=2)
    return np.mean(valid, axis=0)


def detect_obstacle_markers(
    points: np.ndarray,
    labels: list[str] | None = None,
    *,
    n_obstacle: int = 2,
    visibility_min: float = DEFAULT_OBSTACLE_VISIBILITY_MIN,
    lr_axis: str = "y",
) -> tuple[list[int], list[str]]:
    """
    Detect obstacle markers by stationarity and visibility.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    labels : optional list of str (for exclusion; not used for assignment)
    n_obstacle : number of obstacle markers to select (default 2)
    visibility_min : minimum fraction of frames with valid data (default 0.8)
    lr_axis : 'y' or 'x'. When subject walks along x-axis, use 'y' to assign
        OBSTACLE_L / OBSTACLE_R by y position (smaller Y = L, larger Y = R).
        Use 'x' for lab x-based assignment (smaller x = L, larger x = R).

    Returns
    -------
    indices : list of int, length n_obstacle (point indices)
    obstacle_labels : list of str, OBSTACLE_L and OBSTACLE_R (L/R by lr_axis position)
    """
    n_points = points.shape[1]
    motion = motion_score_per_marker(points)
    visibility = visibility_fraction(points)
    # Only consider markers with sufficient visibility
    candidates = np.where(visibility >= visibility_min)[0]
    if len(candidates) < n_obstacle:
        return [], []
    # Among candidates, take the n_obstacle with lowest motion
    motion_cand = motion[candidates].copy()
    motion_cand[~np.isfinite(motion_cand)] = np.inf
    order = np.argsort(motion_cand)
    selected_candidates = candidates[order[:n_obstacle]]
    # Assign OBSTACLE_L and OBSTACLE_R by position along lr_axis (0=x, 1=y)
    axis_idx = 1 if str(lr_axis).strip().lower() == "y" else 0
    pos = np.nanmean(points[:, selected_candidates, axis_idx], axis=0)
    # Smaller position -> L, larger -> R
    lr_order = np.argsort(pos)
    obstacle_labels_ordered = [OBSTACLE_LABELS[i] for i in lr_order]
    indices = list(selected_candidates[lr_order])
    label_names = [obstacle_labels_ordered[i] for i in range(n_obstacle)]
    return indices, label_names


def screened_indices_extra_stationary_to_drop(
    points: np.ndarray,
    obstacle_indices: Sequence[int],
    *,
    motion_max_mm: float,
    visibility_min: float,
) -> list[int]:
    """
    Screened-column indices to drop: markers that are not obstacles but are as stationary
    as the threshold allows (mean inter-frame displacement < ``motion_max_mm``).

    Uses **all frames** in ``points`` (same statistic as :func:`motion_score_per_marker`).
    Labeling best frame (manual or automatic) is not used and must not affect this call.

    Obstacle columns are never selected. Markers below ``visibility_min`` are skipped
    (same idea as obstacle candidates).
    """
    n_points = points.shape[1]
    obs = frozenset(int(i) for i in obstacle_indices)
    motion = motion_score_per_marker(points)
    visibility = visibility_fraction(points)
    out: list[int] = []
    for j in range(n_points):
        if j in obs:
            continue
        if visibility[j] < visibility_min:
            continue
        if not np.isfinite(motion[j]):
            continue
        if float(motion[j]) < float(motion_max_mm):
            out.append(j)
    return out


def remap_indices_after_screened_drops(
    indices: Sequence[int],
    dropped_screened: Sequence[int],
) -> list[int]:
    """Remap column indices after removing screened columns ``dropped_screened`` (sorted ascending)."""
    drop_set = sorted(set(int(d) for d in dropped_screened))
    out: list[int] = []
    for i in indices:
        ii = int(i)
        shift = sum(1 for d in drop_set if d < ii)
        out.append(ii - shift)
    return out


def remove_indices_from_trajectories(
    points: np.ndarray,
    labels: list[str],
    indices_to_remove: list[int],
) -> tuple[np.ndarray, list[str]]:
    """Remove given point indices from points and labels."""
    keep = [i for i in range(points.shape[1]) if i not in indices_to_remove]
    points_new = points[:, keep, :]
    labels_new = [labels[i] for i in keep]
    return points_new, labels_new
