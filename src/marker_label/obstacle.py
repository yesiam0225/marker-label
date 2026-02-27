"""Obstacle marker detection by stationarity."""

from __future__ import annotations

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
) -> tuple[list[int], list[str]]:
    """
    Detect obstacle markers by stationarity and visibility.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    labels : optional list of str (for exclusion; not used for assignment)
    n_obstacle : number of obstacle markers to select (default 2)
    visibility_min : minimum fraction of frames with valid data (default 0.8)

    Returns
    -------
    indices : list of int, length n_obstacle (point indices)
    obstacle_labels : list of str, OBSTACLE_L and OBSTACLE_R (L/R by lab x at first valid frame)
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
    # Assign OBSTACLE_L and OBSTACLE_R by x position (first frame, or mean of valid)
    x_pos = np.nanmean(points[:, selected_candidates, 0], axis=0)
    # Smaller x -> L, larger x -> R (adjust if your lab convention differs)
    lr_order = np.argsort(x_pos)
    obstacle_labels_ordered = [OBSTACLE_LABELS[i] for i in lr_order]
    indices = list(selected_candidates[lr_order])
    label_names = [obstacle_labels_ordered[i] for i in range(n_obstacle)]
    return indices, label_names


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
