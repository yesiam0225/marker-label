"""Velocity-based continuity validation for filled samples."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def validate_continuity(
    points: np.ndarray,
    marker: str,
    row: int,
    label_to_idx: Mapping[str, int],
    max_velocity_mm_per_frame: float,
) -> tuple[bool, str]:
    """Check filled row ``row`` against finite neighbors on the same marker."""
    n = points.shape[0]
    mi = label_to_idx[str(marker).strip()]
    p = points[row, mi, :]
    if not np.isfinite(p).all():
        return True, ""
    for nb, direction in ((row - 1, "prev"), (row + 1, "next")):
        if nb < 0 or nb >= n:
            continue
        q = points[nb, mi, :]
        if not np.isfinite(q).all():
            continue
        d = float(np.linalg.norm(p - q))
        if d > float(max_velocity_mm_per_frame):
            return False, f"velocity_violation_{direction}:{d:.2f}mm"
    return True, ""


def frame_column_to_row(frame_column: np.ndarray, frame_value: int) -> int | None:
    hits = np.flatnonzero(frame_column == int(frame_value))
    if hits.size == 0:
        return None
    return int(hits[0])


def apply_continuity_check_to_fills(
    points: np.ndarray,
    fills: list[dict],
    label_to_idx: Mapping[str, int],
    frame_column: np.ndarray,
    *,
    action: str = "downgrade",
    max_velocity_mm_per_frame: float = 50.0,
) -> tuple[int, int]:
    """
    Mutates ``points`` (revert) and ``fills`` entries (confidence / success / reason).

    Returns ``(continuity_warnings, reverted_fills)``.
    """
    warnings = 0
    reverted = 0
    for entry in fills:
        if not entry.get("success"):
            continue
        fc = int(entry["frame"])
        row = frame_column_to_row(frame_column, fc)
        if row is None:
            continue
        marker = str(entry["marker"]).strip()
        ok, reason = validate_continuity(
            points, marker, row, label_to_idx, max_velocity_mm_per_frame
        )
        if ok:
            continue
        if action == "revert":
            mi = label_to_idx[marker]
            points[row, mi, :] = np.nan
            entry["success"] = False
            entry["confidence"] = ""
            entry["reason"] = reason
            reverted += 1
        else:
            entry["confidence"] = "LOW"
            w = f"continuity_warning:{reason}"
            prev = entry.get("reason", "")
            entry["reason"] = f"{prev};{w}" if prev else w
            warnings += 1
    return warnings, reverted
