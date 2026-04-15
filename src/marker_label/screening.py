"""Dynamic trial initial screening (Steps 1–5). See docs/DYNAMIC_TRIAL_INITIAL_SCREENING_PLAN.md."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import (
    SCREENING_Y_MIN_MM,
    SCREENING_Y_MAX_MM,
    SCREENING_Y_MIN_FINITE_FRAMES,
    SCREENING_Y_OUTSIDE_FRACTION_THRESHOLD,
    SCREENING_VISIBILITY_MIN,
)
from .obstacle import visibility_fraction


class ScreeningError(Exception):
    """Raised when a step of the dynamic trial initial screening fails."""

    def __init__(self, step: int, message: str):
        self.step = step
        self.message = message
        super().__init__(f"Initial screening Step {step} failed: {message}")


def drop_columns_outside_y_range(
    points: np.ndarray,
    *,
    y_min_mm: float = SCREENING_Y_MIN_MM,
    y_max_mm: float = SCREENING_Y_MAX_MM,
    y_outside_fraction_threshold: float = SCREENING_Y_OUTSIDE_FRACTION_THRESHOLD,
    min_finite_y_frames: int = SCREENING_Y_MIN_FINITE_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 1: Drop columns using Y range over time (two rules).

    For each column j:

    - Let ``n_frames`` be the total frame count. If the number of frames with finite Y
      is **strictly less than** ``min_finite_y_frames``, drop the column.
    - Otherwise, let ``n_out`` be the count of frames where Y is finite and outside
      ``[y_min_mm, y_max_mm]``, and ``n_finite`` the count of frames with finite Y.
      If ``n_out / n_finite > y_outside_fraction_threshold``, drop the column.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    y_min_mm, y_max_mm : allowed Y range in mm
    y_outside_fraction_threshold : drop if ``n_out / n_finite`` exceeds this (among finite Y)
    min_finite_y_frames : drop if count of frames with finite Y is below this

    Returns
    -------
    points_reduced : (n_frames, n_keep, 3)
    keep_indices : (n_keep,) int, indices into original columns
    """
    n_frames, n_points, _ = points.shape
    if n_frames < 1:
        raise ScreeningError(1, "no frames in trajectory for Y screening.")
    y = points[:, :, 1]  # (n_frames, n_points)
    finite = np.isfinite(y)
    n_finite = np.sum(finite, axis=0).astype(np.float64)
    out_of_range = finite & ((y < y_min_mm) | (y > y_max_mm))
    n_out = np.sum(out_of_range, axis=0).astype(np.float64)
    # Fraction of *finite-Y frames* that are outside the band (NaN frames excluded from denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction_out = np.where(n_finite > 0, n_out / n_finite, 0.0)
    drop = (n_finite < min_finite_y_frames) | (fraction_out > y_outside_fraction_threshold)
    keep = np.where(~drop)[0]
    if len(keep) == 0:
        raise ScreeningError(
            1,
            "all marker columns were dropped by Step 1 (Y range fraction / min finite-Y rule).",
        )
    return points[:, keep, :].copy(), keep


def trim_frames_by_range(
    points: np.ndarray,
    residual: np.ndarray | None = None,
    *,
    first_frame_1based: int,
    last_frame_1based: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """
    Step 1.5: Trim to the user-specified frame range (1-based, inclusive).

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    residual : (n_frames, n_points) or None
    first_frame_1based : first frame to keep (1-based, inclusive)
    last_frame_1based : last frame to keep (1-based, inclusive)

    Returns
    -------
    points_trimmed : (n_keep_frames, n_points, 3)
    residual_trimmed : (n_keep_frames, n_points) or None
    frame_trim_start : int, 0-based index of first retained frame
    """
    n_frames, n_points, _ = points.shape
    first_idx = first_frame_1based - 1
    last_idx = last_frame_1based - 1
    if first_idx < 0 or last_idx >= n_frames or first_idx > last_idx:
        raise ScreeningError(
            1.5,
            f"invalid frame range: first={first_frame_1based}, last={last_frame_1based} "
            f"(trial has {n_frames} frames, 1-based range 1..{n_frames}).",
        )
    pts_out = points[first_idx : last_idx + 1].copy()
    res_out = residual[first_idx : last_idx + 1].copy() if residual is not None else None
    return pts_out, res_out, first_idx


def drop_columns_by_visibility(
    points: np.ndarray,
    *,
    min_visibility: float = SCREENING_VISIBILITY_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 2: Drop columns with valid data in less than min_visibility fraction of frames.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    min_visibility : in [0, 1]; drop columns with visibility < this

    Returns
    -------
    points_reduced : (n_frames, n_keep, 3)
    keep_indices : (n_keep,) int
    """
    vis = visibility_fraction(points)
    keep = np.where(vis >= min_visibility)[0]
    if len(keep) == 0:
        raise ScreeningError(
            2,
            f"no marker columns left with visibility >= {min_visibility:.0%}.",
        )
    return points[:, keep, :].copy(), keep


def apply_initial_screening_steps_1_and_2(
    points: np.ndarray,
    residual: np.ndarray | None = None,
    *,
    y_min_mm: float = SCREENING_Y_MIN_MM,
    y_max_mm: float = SCREENING_Y_MAX_MM,
    y_outside_fraction_threshold: float = SCREENING_Y_OUTSIDE_FRACTION_THRESHOLD,
    min_finite_y_frames: int = SCREENING_Y_MIN_FINITE_FRAMES,
    min_visibility: float = SCREENING_VISIBILITY_MIN,
    trim_first_frame: int | None = None,
    trim_last_frame: int | None = None,
    skip_visibility_step: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int]:
    """
    Optional Step 1.5 (trim to user first/last frame), then Step 1 (Y range),
    then Step 2 (visibility) unless skip_visibility_step.

    Trimming **before** Y screening (Plan B) ensures columns are dropped only if Y is
    out of range within the analysis window—spikes outside that window do not remove
    the marker column.

    Returns screened points, keep indices (into loaded ``points`` columns), and the
    0-based index of the first retained frame when trim is applied (else 0).

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    residual : (n_frames, n_points) or None; if provided, sliced to same columns/frames
    y_min_mm, y_max_mm : Step 1 Y band (mm)
    y_outside_fraction_threshold, min_finite_y_frames : Step 1 column-drop rule (see ``drop_columns_outside_y_range``)
    min_visibility : Step 2 threshold
    trim_first_frame, trim_last_frame : optional 1-based first/last frame to keep
        (Step 1.5). If both set, trim to this range **before** Step 1 (Y range); if either
        is None, no frame trimming is applied.
    skip_visibility_step : if True, do not apply Step 2 (keep all columns from Step 1/1.5).
    """
    pts = points
    residual_out = residual
    trim_start = 0
    if trim_first_frame is not None and trim_last_frame is not None:
        pts, residual_out, trim_start = trim_frames_by_range(
            pts,
            residual_out,
            first_frame_1based=trim_first_frame,
            last_frame_1based=trim_last_frame,
        )

    pts, keep1 = drop_columns_outside_y_range(
        pts,
        y_min_mm=y_min_mm,
        y_max_mm=y_max_mm,
        y_outside_fraction_threshold=y_outside_fraction_threshold,
        min_finite_y_frames=min_finite_y_frames,
    )
    if residual_out is not None:
        residual_out = residual_out[:, keep1]

    if skip_visibility_step:
        keep_indices = keep1
    else:
        pts, keep2 = drop_columns_by_visibility(pts, min_visibility=min_visibility)
        keep_indices = keep1[keep2]
        if residual_out is not None:
            residual_out = residual_out[:, keep2]

    return pts, residual_out, keep_indices, trim_start


def loaded_point_indices_for_body(
    screening_column_keep: np.ndarray,
    body_screened_column_indices: Sequence[int],
) -> list[int]:
    """
    Map each body trajectory column to the marker column index in ``load_c3d`` output
    (before initial screening).

    ``screening_column_keep[j]`` is the loaded-file column index for screened column ``j``.
    ``body_screened_column_indices[b]`` is the screened column index for body column ``b``
    (e.g. ``keep`` after removing obstacles from screened ``points_d``).

    Returns
    -------
    list of length len(body_screened_column_indices), 0-based indices into loaded ``points``.
    """
    sk = np.asarray(screening_column_keep, dtype=np.int64)
    return [int(sk[int(s)]) for s in body_screened_column_indices]


def compute_walking_direction_x(points: np.ndarray) -> float:
    """
    Step 3: Rate of change of centroid X over time (walking along x-axis).

    Uses linear trend of centroid X (mean over valid points per frame). The sign
    indicates whether the x value increases or decreases during walking:
    positive = x increases (subject walking toward +x), negative = x decreases
    (subject walking toward -x).

    Parameters
    ----------
    points : (n_frames, n_points, 3)

    Returns
    -------
    wdx : float; positive = x increases during walking, negative = x decreases.
          Zero or NaN if indeterminate (e.g. single frame).
    """
    n_frames = points.shape[0]
    if n_frames < 2:
        return 0.0
    centroid_x = np.nanmean(points[:, :, 0], axis=1)  # (n_frames,)
    valid = np.isfinite(centroid_x)
    if np.sum(valid) < 2:
        return 0.0
    t = np.arange(n_frames, dtype=np.float64)
    # Linear regression: centroid_x ~ t
    t_valid = t[valid]
    x_valid = centroid_x[valid]
    t_mean = np.mean(t_valid)
    x_mean = np.mean(x_valid)
    cov = np.mean((t_valid - t_mean) * (x_valid - x_mean))
    var_t = np.mean((t_valid - t_mean) ** 2)
    if var_t <= 0:
        return 0.0
    slope = cov / var_t
    return float(slope)


def get_lr_ap_axes_from_walking(wdx: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 3: L/R and A/P axes from whether x increases or decreases during walking.

    Subject walks forward along the x-axis. Convention:
    - When x value **increases** during walking: larger X = anterior, larger Y = left.
      So posterior = -x, subject's right = -y.
    - When x value **decreases** during walking: larger X = posterior, larger Y = right.
      So anterior = -x, subject's left = -y.

    Parameters
    ----------
    wdx : from compute_walking_direction_x; positive = x increases, negative = x decreases

    Returns
    -------
    d_back : (2,) xy unit vector pointing toward posterior (back)
    d_right : (2,) xy unit vector pointing toward subject's right
    """
    if wdx >= 0:
        # x increases during walking: larger x = anterior, larger y = left
        # → posterior = -x, right = -y
        d_back = np.array([-1.0, 0.0])
        d_right = np.array([0.0, -1.0])
    else:
        # x decreases during walking: larger x = posterior, larger y = right
        # → posterior = +x, right = +y
        d_back = np.array([1.0, 0.0])
        d_right = np.array([0.0, 1.0])
    return d_back, d_right
