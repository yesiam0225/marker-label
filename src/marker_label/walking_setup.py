"""Auto-detect walking setup and lead-foot crossing from marker trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _finite_rows(traj: np.ndarray) -> np.ndarray:
    """Boolean mask for rows where XYZ are all finite."""
    return np.isfinite(traj).all(axis=1)


def determine_setup(
    markers: Mapping[str, np.ndarray],
    obstacle_pair: tuple[str, str] | None,
    *,
    warnings_list: list[str] | None = None,
) -> dict[str, Any]:
    """
    Auto-detect walking axis/direction, ML axis/sign, and obstacle position.

    Parameters
    ----------
    markers
        stem -> (n_frames, 3) trajectory arrays.
    obstacle_pair
        Required to compute ``obstacle_pos``. If ``None``, ``obstacle_pos`` is NaN.
    warnings_list
        Optional mutable list to append uncertainty warnings.
    """
    if "LASI" not in markers or "RASI" not in markers:
        raise ValueError("Setup detection requires LASI and RASI trajectories")
    lasi = np.asarray(markers["LASI"], dtype=np.float64)
    rasi = np.asarray(markers["RASI"], dtype=np.float64)
    if lasi.shape != rasi.shape or lasi.ndim != 2 or lasi.shape[1] != 3:
        raise ValueError("LASI and RASI must be matching (n_frames, 3) arrays")

    valid_lasi = _finite_rows(lasi)
    valid_rasi = _finite_rows(rasi)
    valid_both = valid_lasi & valid_rasi

    if int(np.sum(valid_both)) >= 20:
        pelvis_proxy = 0.5 * (lasi + rasi)
        valid = valid_both
    else:
        if int(np.sum(valid_lasi)) >= int(np.sum(valid_rasi)):
            pelvis_proxy = lasi
            valid = valid_lasi
        else:
            pelvis_proxy = rasi
            valid = valid_rasi

    if int(np.sum(valid)) < 20:
        raise ValueError("Insufficient pelvis marker data for setup detection")

    range_x = float(np.ptp(pelvis_proxy[valid, 0]))
    range_y = float(np.ptp(pelvis_proxy[valid, 1]))
    walking_axis = 0 if range_x > range_y else 1
    ml_axis = 1 - walking_axis

    valid_frames = np.flatnonzero(valid)
    n_window = max(10, int(len(valid_frames) // 4))
    early_mean = float(np.mean(pelvis_proxy[valid_frames[:n_window], walking_axis]))
    late_mean = float(np.mean(pelvis_proxy[valid_frames[-n_window:], walking_axis]))
    walking_direction = 1 if late_mean > early_mean else -1
    if warnings_list is not None:
        if abs(late_mean - early_mean) < 0.3 * max(range_x, range_y):
            warnings_list.append("Walking direction uncertain (small net position change)")

    lasi_ml = float(np.nanmean(lasi[:, ml_axis]))
    rasi_ml = float(np.nanmean(rasi[:, ml_axis]))
    left_ml_sign = 1 if lasi_ml > rasi_ml else -1

    obstacle_pos = float("nan")
    if obstacle_pair is not None:
        a, b = str(obstacle_pair[0]).strip(), str(obstacle_pair[1]).strip()
        if a not in markers or b not in markers:
            raise ValueError(f"Obstacle pair markers not found: {a!r}, {b!r}")
        ob_l = np.asarray(markers[a], dtype=np.float64)
        ob_r = np.asarray(markers[b], dtype=np.float64)
        obstacle_pos = float(
            0.5 * (np.nanmean(ob_l[:, walking_axis]) + np.nanmean(ob_r[:, walking_axis]))
        )

    return {
        "walking_axis": int(walking_axis),
        "walking_direction": int(walking_direction),
        "ml_axis": int(ml_axis),
        "left_ml_sign": int(left_ml_sign),
        "obstacle_pos": float(obstacle_pos),
    }


def find_lead_foot_crossing(
    markers: Mapping[str, np.ndarray],
    *,
    walking_axis: int,
    walking_direction: int,
    obstacle_pos: float,
) -> tuple[int, str]:
    """Return (crossing_frame, lead_foot_side) using LTOE/RTOE first obstacle crossing."""
    if "LTOE" not in markers or "RTOE" not in markers:
        raise ValueError("Lead-foot crossing requires LTOE and RTOE trajectories")
    ltoe = np.asarray(markers["LTOE"], dtype=np.float64)[:, int(walking_axis)]
    rtoe = np.asarray(markers["RTOE"], dtype=np.float64)[:, int(walking_axis)]

    def first_crossing(traj: np.ndarray, pos: float, direction: int) -> int | None:
        for f in range(len(traj) - 1):
            a = float(traj[f])
            b = float(traj[f + 1])
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if direction > 0:
                if a < pos and b >= pos:
                    return int(f + 1)
            else:
                if a > pos and b <= pos:
                    return int(f + 1)
        return None

    l_cross = first_crossing(ltoe, float(obstacle_pos), int(walking_direction))
    r_cross = first_crossing(rtoe, float(obstacle_pos), int(walking_direction))

    candidates: list[tuple[str, int]] = []
    if l_cross is not None:
        candidates.append(("left", int(l_cross)))
    if r_cross is not None:
        candidates.append(("right", int(r_cross)))
    if not candidates:
        raise ValueError("No foot crossing detected. Verify obstacle position and trajectories.")
    candidates.sort(key=lambda x: x[1])
    side, frame = candidates[0]
    return int(frame), str(side)
