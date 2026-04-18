"""Obstacle marker detection by stationarity.

Extra stationary column dropping (``screened_indices_extra_stationary_to_drop``) is part of the
standard pipeline when two obstacles exist; disabling it is for exceptional cases — see README.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import (
    OBSTACLE_LABELS,
    DEFAULT_OBSTACLE_MAX_MOTION_MM,
    DEFAULT_OBSTACLE_VISIBILITY_FLOOR,
    DEFAULT_OBSTACLE_VISIBILITY_MIN,
    DEFAULT_OBSTACLE_ROD_LENGTH_MIN_MM,
    DEFAULT_OBSTACLE_ROD_MAX_PAIR_CANDIDATES,
    DEFAULT_OBSTACLE_ROD_MIN_OVERLAP_FRAMES,
    DEFAULT_OBSTACLE_ROD_VISIBILITY_MIN,
)


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


def median_motion_per_marker(points: np.ndarray) -> np.ndarray:
    """
    Median inter-frame displacement magnitude per marker (mm/frame).

    Uses the same valid-pair mask as :func:`motion_score_per_marker` but aggregates with
    ``nanmedian`` instead of ``nanmean`` for robustness to occasional tracking spikes.
    """
    n_frames, n_points, _ = points.shape
    velocity = np.full((n_frames - 1, n_points), np.nan)
    for i in range(n_frames - 1):
        d = points[i + 1] - points[i]
        valid = np.isfinite(points[i]).all(axis=1) & np.isfinite(points[i + 1]).all(axis=1)
        vel_mag = np.linalg.norm(d, axis=1)
        velocity[i] = np.where(valid, vel_mag, np.nan)
    median_vel = np.full(n_points, np.nan)
    with np.errstate(invalid="ignore"):
        for j in range(n_points):
            col = velocity[:, j]
            if np.any(np.isfinite(col)):
                median_vel[j] = np.nanmedian(col)
    return median_vel


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
    motion_max_mm: float | None = None,
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
    motion_max_mm : max allowed mean inter-frame speed (mm/frame) for a column to be an obstacle
        candidate; same statistic as :func:`motion_score_per_marker`. ``None`` uses
        ``DEFAULT_OBSTACLE_MAX_MOTION_MM``. Use ``0`` to disable (no upper limit on motion among
        visibility-qualified candidates).

    Returns
    -------
    indices : list of int, length n_obstacle (point indices)
    obstacle_labels : list of str, OBSTACLE_L and OBSTACLE_R (L/R by lr_axis position)
    """
    motion = motion_score_per_marker(points)
    visibility = visibility_fraction(points)
    # Only consider markers with sufficient visibility
    candidates = np.where(visibility >= visibility_min)[0]
    cap = DEFAULT_OBSTACLE_MAX_MOTION_MM if motion_max_mm is None else float(motion_max_mm)
    if cap > 0:
        candidates = np.array(
            [
                int(j)
                for j in candidates
                if np.isfinite(motion[j]) and float(motion[j]) <= cap
            ],
            dtype=np.int64,
        )
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


def detect_obstacle_markers_rod_pair(
    points: np.ndarray,
    labels: list[str] | None = None,
    *,
    n_obstacle: int = 2,
    visibility_min: float = DEFAULT_OBSTACLE_ROD_VISIBILITY_MIN,
    visibility_floor: float = DEFAULT_OBSTACLE_VISIBILITY_FLOOR,
    motion_max_mm: float | None = None,
    rod_separation_axis: str = "y",
    rod_length_min_mm: float = DEFAULT_OBSTACLE_ROD_LENGTH_MIN_MM,
    rod_length_target_mm: float | None = None,
    rod_max_pair_candidates: int = DEFAULT_OBSTACLE_ROD_MAX_PAIR_CANDIDATES,
    rod_min_overlap_frames: int = DEFAULT_OBSTACLE_ROD_MIN_OVERLAP_FRAMES,
    lr_axis: str = "y",
) -> tuple[list[int], list[str]]:
    """
    Detect two obstacle markers using a **rod** prior: endpoints share similar lateral coordinates
    (the two axes orthogonal to ``rod_separation_axis``) and are separated along the rod axis.

    **Process (order)**

    1. **Per-column robust motion** — :func:`median_motion_per_marker` (spike-resistant).
    2. **Per-column visibility** — exclude columns below ``visibility_floor`` (junk channels), then
       require visibility ≥ ``visibility_min`` (often relaxed vs legacy, e.g. 0.72).
    3. **Motion cap** — if ``motion_max_mm`` > 0, keep only columns with median motion ≤ cap
       (same cap convention as legacy; ``None`` uses ``DEFAULT_OBSTACLE_MAX_MOTION_MM``).
    4. **Shortlist** — sort by median motion ascending; keep the ``rod_max_pair_candidates`` lowest.
    5. **Pair scoring** — for each pair, use mean positions on **frames where both markers are
       finite** (avoids biased geometry when visibility windows differ).
       Lateral spread = sum of absolute differences on the two axes orthogonal to the rod;
       axial separation = absolute difference on ``rod_separation_axis``. Require axial ≥
       ``rod_length_min_mm``. Score primarily by lateral/axial (small = rod-like); tie-break by
       lower max(median motion of the two). Optional ``rod_length_target_mm`` adds a soft penalty
       when the lab rod length is known across trials.
    6. **L/R labels** — same as legacy: ``lr_axis`` on mean positions (smaller axis value → L).

    Returns ``[], []`` if fewer than two candidates remain or no pair satisfies ``rod_length_min_mm``.
    """
    if n_obstacle != 2:
        raise ValueError("detect_obstacle_markers_rod_pair currently supports n_obstacle=2 only")

    med = median_motion_per_marker(points)
    vis = visibility_fraction(points)
    cap = DEFAULT_OBSTACLE_MAX_MOTION_MM if motion_max_mm is None else float(motion_max_mm)

    candidates: list[int] = []
    for j in range(points.shape[1]):
        if vis[j] < float(visibility_floor):
            continue
        if vis[j] < float(visibility_min):
            continue
        if not np.isfinite(med[j]):
            continue
        if cap > 0 and float(med[j]) > cap:
            continue
        candidates.append(int(j))

    if len(candidates) < 2:
        return [], []

    candidates.sort(key=lambda j: float(med[j]))
    kmax = max(2, int(rod_max_pair_candidates))
    shortlist = candidates[: min(len(candidates), kmax)]

    axis = str(rod_separation_axis).strip().lower()
    if axis not in ("x", "y", "z"):
        axis = "y"
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    lateral_idx = tuple(i for i in range(3) if i != axis_idx)

    min_ov = int(rod_min_overlap_frames)

    best_pair: tuple[int, int] | None = None
    best_key: tuple[float, float] | None = None

    for a in range(len(shortlist)):
        for b in range(a + 1, len(shortlist)):
            i, k = shortlist[a], shortlist[b]
            both = np.isfinite(points[:, i]).all(axis=1) & np.isfinite(points[:, k]).all(axis=1)
            if int(np.sum(both)) < min_ov:
                continue
            pi = np.mean(points[both, i, :], axis=0)
            pk = np.mean(points[both, k, :], axis=0)
            if not (np.all(np.isfinite(pi)) and np.all(np.isfinite(pk))):
                continue
            d = np.abs(pi - pk)
            axial = float(d[axis_idx])
            lateral = float(d[lateral_idx[0]] + d[lateral_idx[1]])
            if axial < float(rod_length_min_mm):
                continue
            shape = lateral / (axial + 1e-9)
            if rod_length_target_mm is not None and float(rod_length_target_mm) > 0:
                tgt = float(rod_length_target_mm)
                shape += 0.01 * abs(axial - tgt) / tgt
            motion_tb = max(float(med[i]), float(med[k]))
            key = (shape, motion_tb)
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (i, k)

    if best_pair is None:
        return [], []

    i, k = best_pair
    both_lr = np.isfinite(points[:, i]).all(axis=1) & np.isfinite(points[:, k]).all(axis=1)
    axis_lr = 1 if str(lr_axis).strip().lower() == "y" else 0
    pos = np.array(
        [
            float(np.mean(points[both_lr, i, axis_lr])),
            float(np.mean(points[both_lr, k, axis_lr])),
        ]
    )
    lr_order = np.argsort(pos)
    pair_idx = np.array([i, k], dtype=np.int64)
    indices = [int(pair_idx[lr_order[0]]), int(pair_idx[lr_order[1]])]
    obstacle_labels_ordered = [OBSTACLE_LABELS[0], OBSTACLE_LABELS[1]]
    return indices, obstacle_labels_ordered


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
