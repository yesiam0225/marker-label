"""Short-gap cubic spline interpolation (no extrapolation)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.interpolate import CubicSpline


def _entry(
    frame: int,
    marker: str,
    success: bool,
    confidence: str,
    px: float,
    py: float,
    pz: float,
    source_markers: str,
    gap_length: int,
    reason: str,
) -> dict:
    return {
        "frame": frame,
        "marker": marker,
        "method": "spline",
        "success": success,
        "confidence": confidence,
        "predicted_x": px,
        "predicted_y": py,
        "predicted_z": pz,
        "fit_residual_mm": np.nan,
        "source_markers": source_markers,
        "gap_length": gap_length,
        "reason": reason,
    }


def spline_fill(
    points: np.ndarray,
    marker: str,
    gap: tuple[int, int, int],
    label_to_idx: Mapping[str, int],
    config: Mapping,
    frame_column: np.ndarray,
) -> list[dict]:
    start, end, length = gap
    n_frames = points.shape[0]
    mi = label_to_idx[str(marker).strip()]
    pad = int(config.get("spline_padding_frames", 5))
    min_side = int(config.get("spline_min_anchor_frames", 2))

    out: list[dict] = []

    if start == 0 or end == n_frames - 1:
        for f in range(start, end + 1):
            out.append(
                _entry(
                    int(frame_column[f]),
                    marker,
                    False,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    "0",
                    length,
                    "gap_at_boundary",
                )
            )
        return out

    left_frames: list[int] = []
    f = start - 1
    while f >= 0 and len(left_frames) < pad:
        if np.isfinite(points[f, mi, :]).all():
            left_frames.append(f)
        f -= 1
    right_frames: list[int] = []
    f = end + 1
    while f < n_frames and len(right_frames) < pad:
        if np.isfinite(points[f, mi, :]).all():
            right_frames.append(f)
        f += 1
    left_frames.sort()
    right_frames.sort()

    if len(left_frames) < min_side or len(right_frames) < min_side:
        for ff in range(start, end + 1):
            out.append(
                _entry(
                    int(frame_column[ff]),
                    marker,
                    False,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    str(len(left_frames) + len(right_frames)),
                    length,
                    "insufficient_anchors",
                )
            )
        return out

    anchor_rows = left_frames + right_frames
    t_anchor = np.array([float(frame_column[r]) for r in anchor_rows], dtype=np.float64)
    p_anchor = np.stack([points[r, mi, :] for r in anchor_rows], axis=0)

    try:
        cs_x = CubicSpline(t_anchor, p_anchor[:, 0], extrapolate=False)
        cs_y = CubicSpline(t_anchor, p_anchor[:, 1], extrapolate=False)
        cs_z = CubicSpline(t_anchor, p_anchor[:, 2], extrapolate=False)
    except Exception as e:  # noqa: BLE001
        for ff in range(start, end + 1):
            out.append(
                _entry(
                    int(frame_column[ff]),
                    marker,
                    False,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    str(len(anchor_rows)),
                    length,
                    f"spline_fit_failed:{e!s}",
                )
            )
        return out

    if length <= 3:
        conf = "HIGH"
    elif length <= 7:
        conf = "MEDIUM"
    else:
        conf = "LOW"

    for ff in range(start, end + 1):
        tf = float(frame_column[ff])
        try:
            px = float(cs_x(tf))
            py = float(cs_y(tf))
            pz = float(cs_z(tf))
        except ValueError:
            out.append(
                _entry(
                    int(frame_column[ff]),
                    marker,
                    False,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    str(len(anchor_rows)),
                    length,
                    "extrapolation_forbidden",
                )
            )
            continue
        out.append(
            _entry(
                int(frame_column[ff]),
                marker,
                True,
                conf,
                px,
                py,
                pz,
                str(len(anchor_rows)),
                length,
                "",
            )
        )
    return out
