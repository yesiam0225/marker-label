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
    DEFAULT_OBSTACLE_ROD_PAIR_DX_MAX_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_DZ_MAX_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_FRACTION,
    DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_MM,
    DEFAULT_OBSTACLE_ROD_PAIR_P90_MOTION_MAX_MM,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_LENGTH,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_MOTION,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_X,
    DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_Z,
    DEFAULT_OBSTACLE_VISIBILITY_FLOOR,
    DEFAULT_OBSTACLE_VISIBILITY_MIN,
    DEFAULT_OBSTACLE_ROD_LENGTH_MIN_MM,
    DEFAULT_OBSTACLE_ROD_MAX_PAIR_CANDIDATES,
    DEFAULT_OBSTACLE_ROD_MIN_OVERLAP_FRAMES,
    DEFAULT_OBSTACLE_ROD_VISIBILITY_MIN,
    MIN_FINITE_Y_SAMPLES_PER_OBSTACLE_MARKER,
)
from .errors import ERR_OBSTACLE_Y_BAND_INSUFFICIENT_DATA, LabelingPipelineError


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


def p90_motion_per_marker(points: np.ndarray) -> np.ndarray:
    """90th percentile inter-frame displacement magnitude per marker (mm/frame)."""
    n_frames, n_points, _ = points.shape
    velocity = np.full((n_frames - 1, n_points), np.nan)
    for i in range(n_frames - 1):
        d = points[i + 1] - points[i]
        valid = np.isfinite(points[i]).all(axis=1) & np.isfinite(points[i + 1]).all(axis=1)
        vel_mag = np.linalg.norm(d, axis=1)
        velocity[i] = np.where(valid, vel_mag, np.nan)
    p90 = np.full(n_points, np.nan)
    with np.errstate(invalid="ignore"):
        for j in range(n_points):
            col = velocity[:, j]
            finite = col[np.isfinite(col)]
            if finite.size:
                p90[j] = float(np.percentile(finite, 90))
    return p90


def visibility_fraction(points: np.ndarray) -> np.ndarray:
    """Fraction of frames where marker is valid (non-NaN). Shape (n_points,)."""
    valid = np.isfinite(points).all(axis=2)
    return np.mean(valid, axis=0)


def median_y_per_marker(points: np.ndarray) -> np.ndarray:
    """Per-column median lab Y (mm) over finite Y frames. NaN if a column has no finite Y."""
    n_frames, n_points, _ = points.shape
    out = np.full(n_points, np.nan, dtype=np.float64)
    for j in range(n_points):
        y = points[:, j, 1]
        finite = y[np.isfinite(y)]
        if finite.size:
            out[j] = float(np.median(finite))
    return out


def _obstacle_candidate_y_range_active(
    y_min: float | None, y_max: float | None
) -> bool:
    return (y_min is not None) or (y_max is not None)


def _validate_obstacle_candidate_y_range(y_min: float | None, y_max: float | None) -> None:
    if y_min is not None and y_max is not None and float(y_min) > float(y_max):
        raise ValueError(
            f"obstacle candidate y_min ({y_min}) must be <= y_max ({y_max})"
        )


def _median_y_in_candidate_band(
    med_y: float,
    y_min: float | None,
    y_max: float | None,
) -> bool:
    if not np.isfinite(med_y):
        return False
    if y_min is not None and med_y < float(y_min):
        return False
    if y_max is not None and med_y > float(y_max):
        return False
    return True


def trial_obstacle_y_band(
    obstacle_points: np.ndarray,
    *,
    aggregate: str = "median",
    min_finite_samples: int = MIN_FINITE_Y_SAMPLES_PER_OBSTACLE_MARKER,
) -> tuple[float, float]:
    """
    Trial-wide vertical band from two stationary obstacle markers.

    For each of the two endpoints, collect finite Y over all frames and take ``median`` or
    ``mean``. Then ``y_lo = min(Y0_agg, Y1_agg)``, ``y_hi = max(Y0_agg, Y1_agg)`` (same
    convention as the instantaneous [min, max] of the two obstacle Y values).

    Parameters
    ----------
    obstacle_points
        (n_frames, 2, 3) trajectories for OBSTACLE_L and OBSTACLE_R (same frame as body).
    aggregate
        ``\"median\"`` (default) or ``\"mean\"``.
    min_finite_samples
        Raise if either marker has fewer than this many frames with finite Y.

    Returns
    -------
    y_lo, y_hi : float
    """
    if obstacle_points.ndim != 3 or obstacle_points.shape[1] < 2:
        raise LabelingPipelineError(
            ERR_OBSTACLE_Y_BAND_INSUFFICIENT_DATA,
            "obstacle_points must have shape (n_frames, 2, 3) with two obstacle columns.",
            step="obstacle_y_band",
        )
    agg = str(aggregate).strip().lower()
    if agg not in ("median", "mean"):
        raise ValueError(f'aggregate must be \"median\" or \"mean\", got {aggregate!r}')

    y_vals = []
    for j in range(2):
        y = obstacle_points[:, j, 1]
        finite = y[np.isfinite(y)]
        if finite.size < int(min_finite_samples):
            raise LabelingPipelineError(
                ERR_OBSTACLE_Y_BAND_INSUFFICIENT_DATA,
                f"obstacle marker {j} has only {finite.size} finite-Y frames; "
                f"need >= {min_finite_samples}.",
                step="obstacle_y_band",
            )
        if agg == "median":
            y_vals.append(float(np.median(finite)))
        else:
            y_vals.append(float(np.mean(finite)))

    y_lo = min(y_vals[0], y_vals[1])
    y_hi = max(y_vals[0], y_vals[1])
    return y_lo, y_hi


def detect_obstacle_markers(
    points: np.ndarray,
    labels: list[str] | None = None,
    *,
    n_obstacle: int = 2,
    visibility_min: float = DEFAULT_OBSTACLE_VISIBILITY_MIN,
    lr_axis: str = "y",
    motion_max_mm: float | None = None,
    candidate_y_min_mm: float | None = None,
    candidate_y_max_mm: float | None = None,
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
    candidate_y_min_mm, candidate_y_max_mm : if either is set, only columns with median lab Y
        in ``[y_min, y_max]`` (inclusive) are obstacle candidates. ``(None, None)`` disables.

    Returns
    -------
    indices : list of int, length n_obstacle (point indices)
    obstacle_labels : list of str, OBSTACLE_L and OBSTACLE_R (L/R by lr_axis position)
    """
    _validate_obstacle_candidate_y_range(candidate_y_min_mm, candidate_y_max_mm)
    motion = motion_score_per_marker(points)
    visibility = visibility_fraction(points)
    med_y = (
        median_y_per_marker(points)
        if _obstacle_candidate_y_range_active(candidate_y_min_mm, candidate_y_max_mm)
        else None
    )
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
    if med_y is not None:
        candidates = np.array(
            [
                int(j)
                for j in candidates
                if _median_y_in_candidate_band(
                    med_y[j], candidate_y_min_mm, candidate_y_max_mm
                )
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
    rod_pair_dx_max_mm: float = DEFAULT_OBSTACLE_ROD_PAIR_DX_MAX_MM,
    rod_pair_dz_max_mm: float = DEFAULT_OBSTACLE_ROD_PAIR_DZ_MAX_MM,
    rod_pair_length_tol_mm: float = DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_MM,
    rod_pair_length_tol_fraction: float = DEFAULT_OBSTACLE_ROD_PAIR_LENGTH_TOL_FRACTION,
    rod_pair_p90_motion_max_mm: float = DEFAULT_OBSTACLE_ROD_PAIR_P90_MOTION_MAX_MM,
    rod_score_weight_x: float = DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_X,
    rod_score_weight_length: float = DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_LENGTH,
    rod_score_weight_z: float = DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_Z,
    rod_score_weight_motion: float = DEFAULT_OBSTACLE_ROD_SCORE_WEIGHT_MOTION,
    lr_axis: str = "y",
    candidate_y_min_mm: float | None = None,
    candidate_y_max_mm: float | None = None,
) -> tuple[list[int], list[str]]:
    """
    Detect two obstacle markers using a **rod** prior: endpoints share similar lateral coordinates
    (the two axes orthogonal to ``rod_separation_axis``) and are separated along the rod axis.

    **Process (order)**

    1. **Per-column robust motion** — :func:`median_motion_per_marker` (spike-resistant).
    2. **Per-column visibility** — exclude columns below ``visibility_floor`` (junk channels), then
       require visibility ≥ ``visibility_min`` (often relaxed vs legacy, e.g. 0.72).
    3. **Y band (optional)** — if ``candidate_y_min_mm`` / ``candidate_y_max_mm`` are set, require
       median lab Y in ``[y_min, y_max]`` (excludes spurious far-from-subject stationaries).
    4. **Motion cap** — if ``motion_max_mm`` > 0, keep only columns with median motion ≤ cap
       (same cap convention as legacy; ``None`` uses ``DEFAULT_OBSTACLE_MAX_MOTION_MM``).
       Also apply p90 motion cap (``rod_pair_p90_motion_max_mm``) when > 0.
    5. **Shortlist** — sort by median motion ascending; keep the ``rod_max_pair_candidates`` lowest.
    6. **Pair scoring** — for each pair, use mean positions on **frames where both markers are
       finite** (avoids biased geometry when visibility windows differ).
       Lateral spread = sum of absolute differences on the two axes orthogonal to the rod;
       axial separation = absolute difference on ``rod_separation_axis``. Require axial ≥
       ``rod_length_min_mm``. For ``rod_separation_axis='y'``, enforce hard gates
       ``|dx| <= rod_pair_dx_max_mm`` and ``|dz| <= rod_pair_dz_max_mm``.
       If ``rod_length_target_mm`` is set, require ``|axial-target|`` within tolerance
       ``max(rod_pair_length_tol_mm, rod_pair_length_tol_fraction*target)``.
       Score is a weighted sum of X/Z mismatch, length mismatch, and motion.
    7. **L/R labels** — same as legacy: ``lr_axis`` on mean positions (smaller axis value → L).

    Returns ``[], []`` if fewer than two candidates remain or no pair satisfies ``rod_length_min_mm``.
    """
    if n_obstacle != 2:
        raise ValueError("detect_obstacle_markers_rod_pair currently supports n_obstacle=2 only")

    _validate_obstacle_candidate_y_range(candidate_y_min_mm, candidate_y_max_mm)

    med = median_motion_per_marker(points)
    p90 = p90_motion_per_marker(points)
    vis = visibility_fraction(points)
    med_y = (
        median_y_per_marker(points)
        if _obstacle_candidate_y_range_active(candidate_y_min_mm, candidate_y_max_mm)
        else None
    )
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
        if float(rod_pair_p90_motion_max_mm) > 0 and (
            not np.isfinite(p90[j]) or float(p90[j]) > float(rod_pair_p90_motion_max_mm)
        ):
            continue
        if med_y is not None and not _median_y_in_candidate_band(
            med_y[j], candidate_y_min_mm, candidate_y_max_mm
        ):
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
    best_score: float | None = None
    best_tie_motion: float | None = None

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
            dx = float(d[0])
            dz = float(d[2])
            if axis == "y":
                if float(rod_pair_dx_max_mm) > 0 and dx > float(rod_pair_dx_max_mm):
                    continue
                if float(rod_pair_dz_max_mm) > 0 and dz > float(rod_pair_dz_max_mm):
                    continue
            length_penalty = 0.0
            if rod_length_target_mm is not None and float(rod_length_target_mm) > 0:
                tgt = float(rod_length_target_mm)
                tol = max(
                    float(rod_pair_length_tol_mm),
                    abs(tgt) * float(rod_pair_length_tol_fraction),
                )
                length_penalty = abs(axial - tgt)
                if length_penalty > tol:
                    continue
            motion_tb = max(float(med[i]), float(med[k]))
            score = (
                float(rod_score_weight_x) * dx
                + float(rod_score_weight_length) * length_penalty
                + float(rod_score_weight_z) * dz
                + float(rod_score_weight_motion) * motion_tb
            )
            tie_motion = motion_tb
            if (
                best_score is None
                or score < best_score
                or (np.isclose(score, best_score) and best_tie_motion is not None and tie_motion < best_tie_motion)
            ):
                best_score = score
                best_tie_motion = tie_motion
                best_pair = (i, k)

    if best_pair is None:
        return [], []

    return order_obstacle_l_r_for_screened_pair(
        points, int(best_pair[0]), int(best_pair[1]), lr_axis=lr_axis
    )


def order_obstacle_l_r_for_screened_pair(
    points: np.ndarray,
    screened_i: int,
    screened_k: int,
    *,
    lr_axis: str = "y",
) -> tuple[list[int], list[str]]:
    """
    Map two **screened** column indices to ``OBSTACLE_L`` / ``OBSTACLE_R`` (smaller
    position along ``lr_axis`` (default lab Y) → L, larger → R).

    If no frame has both points finite, falls back to **nanmean** of that axis per column
    (same L/R order).
    """
    i, k = int(screened_i), int(screened_k)
    both_lr = np.isfinite(points[:, i]).all(axis=1) & np.isfinite(points[:, k]).all(axis=1)
    axis_lr = 1 if str(lr_axis).strip().lower() == "y" else 0
    if np.any(both_lr):
        pos = np.array(
            [
                float(np.mean(points[both_lr, i, axis_lr])),
                float(np.mean(points[both_lr, k, axis_lr])),
            ],
            dtype=np.float64,
        )
    else:
        a = float(np.nanmean(points[:, i, axis_lr]))
        b = float(np.nanmean(points[:, k, axis_lr]))
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(
                f"obstacle pair: insufficient finite {lr_axis} for screened columns {i} and {k}."
            )
        pos = np.array([a, b], dtype=np.float64)
    lr_order = np.argsort(pos)
    pair_idx = np.array([i, k], dtype=np.int64)
    indices = [int(pair_idx[lr_order[0]]), int(pair_idx[lr_order[1]])]
    return indices, [OBSTACLE_LABELS[0], OBSTACLE_LABELS[1]]


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
