"""Dynamic trial initial screening (Steps 1–5). See docs/DYNAMIC_TRIAL_INITIAL_SCREENING_PLAN.md."""

from __future__ import annotations

import numpy as np

from .constants import (
    SCREENING_Y_MIN_MM,
    SCREENING_Y_MAX_MM,
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 1: Drop columns where any frame has Y outside [y_min_mm, y_max_mm].

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    y_min_mm, y_max_mm : allowed Y range in mm

    Returns
    -------
    points_reduced : (n_frames, n_keep, 3)
    keep_indices : (n_keep,) int, indices into original columns
    """
    n_frames, n_points, _ = points.shape
    y = points[:, :, 1]  # (n_frames, n_points)
    valid = np.isfinite(y)
    # Column i is dropped if any frame has finite Y outside range
    out_of_range = valid & ((y < y_min_mm) | (y > y_max_mm))
    drop = np.any(out_of_range, axis=0)
    keep = np.where(~drop)[0]
    if len(keep) == 0:
        raise ScreeningError(
            1,
            "all marker columns were dropped (every column has at least one frame with Y outside "
            f"[{y_min_mm}, {y_max_mm}] mm, or no finite Y data).",
        )
    return points[:, keep, :].copy(), keep


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
    min_visibility: float = SCREENING_VISIBILITY_MIN,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Apply Step 1 (Y range) then Step 2 (visibility). Returns screened points and keep indices.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    residual : (n_frames, n_points) or None; if provided, sliced to same columns
    y_min_mm, y_max_mm, min_visibility : screening parameters

    Returns
    -------
    points_screened : (n_frames, n_keep, 3)
    residual_screened : (n_frames, n_keep) or None
    keep_indices : (n_keep,) int, indices into original columns
    """
    pts, keep1 = drop_columns_outside_y_range(
        points, y_min_mm=y_min_mm, y_max_mm=y_max_mm
    )
    if residual is not None:
        residual = residual[:, keep1]
    pts, keep2 = drop_columns_by_visibility(pts, min_visibility=min_visibility)
    # keep2 are indices into pts after Step 1; full keep = keep1[keep2]
    keep_indices = keep1[keep2]
    if residual is not None:
        residual = residual[:, keep2]
    return pts, residual, keep_indices


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
