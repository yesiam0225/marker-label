"""Gap filling for marker trajectories (frame-based thresholds TBD)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def fill_gaps_1d(x: np.ndarray, max_interp_frames: int = 10) -> np.ndarray:
    """
    Fill gaps in a 1D signal using linear/spline interpolation or propagation.

    Parameters
    ----------
    x : (n_frames,) float, NaN for missing
    max_interp_frames : gaps longer than this are filled by propagation only

    Returns
    -------
    (n_frames,) filled (copy)
    """
    out = x.copy()
    n = len(x)
    if n == 0:
        return out
    # Find runs of NaN
    valid = np.isfinite(x)
    if valid.all():
        return out
    # Propagate from edges
    if not valid[0]:
        first_valid = np.argmax(valid)
        if first_valid < n:
            out[:first_valid] = x[first_valid]
    if not valid[-1]:
        last_valid = n - 1 - np.argmax(valid[::-1])
        if last_valid >= 0:
            out[last_valid + 1 :] = x[last_valid]
    valid = np.isfinite(out)
    # Process each gap
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        start = i
        while i < n and not valid[i]:
            i += 1
        end = i  # first valid after gap
        gap_len = end - start
        if start == 0:
            out[:end] = out[end]
            continue
        if end == n:
            out[start:] = out[start - 1]
            continue
        left_val = out[start - 1]
        right_val = out[end]
        if gap_len <= 2:
            # Linear
            for k in range(gap_len):
                t = (k + 1) / (gap_len + 1)
                out[start + k] = (1 - t) * left_val + t * right_val
        elif gap_len <= max_interp_frames:
            # Cubic spline
            x_known = np.array([start - 1, end])
            y_known = np.array([left_val, right_val])
            cs = CubicSpline(x_known, y_known)
            out[start:end] = cs(np.arange(start, end))
        else:
            # Propagate left value
            out[start:end] = left_val
        valid = np.isfinite(out)
        i = end
    return out


def fill_gaps_trajectory(
    points: np.ndarray,
    max_interp_frames: int = 10,
) -> np.ndarray:
    """
    Fill gaps in (n_frames, 3) trajectory per axis.

    Parameters
    ----------
    points : (n_frames, 3), NaN where missing
    max_interp_frames : max gap length for spline interpolation

    Returns
    -------
    (n_frames, 3) filled
    """
    out = points.copy()
    for j in range(3):
        out[:, j] = fill_gaps_1d(points[:, j], max_interp_frames=max_interp_frames)
    return out


def fill_gaps_all_markers(
    points: np.ndarray,
    max_interp_frames: int = 10,
) -> np.ndarray:
    """
    Fill gaps for all markers. points shape (n_frames, n_points, 3).

    Returns
    -------
    (n_frames, n_points, 3) filled
    """
    n_frames, n_points, _ = points.shape
    out = points.copy()
    for m in range(n_points):
        out[:, m, :] = fill_gaps_trajectory(points[:, m, :], max_interp_frames=max_interp_frames)
    return out
