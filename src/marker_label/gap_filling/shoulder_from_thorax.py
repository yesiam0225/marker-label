"""Synthesize LSHO/RSHO from thorax markers using static shoulder-local offsets."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .reference import _local_coords_from_three


def _row(
    frame: int,
    marker: str,
    success: bool,
    px: float,
    py: float,
    pz: float,
    source: str,
    gap_length: int,
    reason: str,
) -> dict:
    return {
        "frame": int(frame),
        "marker": marker,
        "method": "shoulder_from_thorax",
        "success": success,
        "confidence": "LOW" if success else "",
        "predicted_x": px,
        "predicted_y": py,
        "predicted_z": pz,
        "fit_residual_mm": np.nan,
        "source_markers": source,
        "gap_length": int(gap_length),
        "reason": reason,
    }


def shoulder_from_thorax_fill(
    points: np.ndarray,
    target_marker: str,
    gap: tuple[int, int, int],
    label_to_idx: Mapping[str, int],
    frame_column: np.ndarray,
    shoulder_local: Mapping[str, np.ndarray],
    *,
    thorax_markers: tuple[str, str, str] = ("C7", "CLAV", "RBAK"),
) -> list[dict]:
    """
    Fill shoulder gaps when C7, CLAV, RBAK are visible using static-mean local offsets.
    """
    start, end, length = gap
    tgt = str(target_marker).strip()
    if tgt not in shoulder_local:
        return [
            _row(int(frame_column[f]), tgt, False, np.nan, np.nan, np.nan, "", length, "no_static_shoulder_local")
            for f in range(start, end + 1)
        ]
    t0, t1, t2 = (str(x).strip() for x in thorax_markers)
    for m in (t0, t1, t2):
        if m not in label_to_idx:
            return [
                _row(int(frame_column[f]), tgt, False, np.nan, np.nan, np.nan, "", length, "missing_thorax_marker")
                for f in range(start, end + 1)
            ]

    loc = np.asarray(shoulder_local[tgt], dtype=np.float64)
    out: list[dict] = []
    for f in range(start, end + 1):
        fc = int(frame_column[f])
        if not all(np.isfinite(points[f, label_to_idx[m], :]).all() for m in (t0, t1, t2)):
            out.append(
                _row(fc, tgt, False, np.nan, np.nan, np.nan, "", length, "thorax_not_visible")
            )
            continue
        p0 = points[f, label_to_idx[t0], :]
        p1 = points[f, label_to_idx[t1], :]
        p2 = points[f, label_to_idx[t2], :]
        try:
            origin, basis = _local_coords_from_three(p0, p1, p2)
        except ValueError:
            out.append(
                _row(fc, tgt, False, np.nan, np.nan, np.nan, "", length, "degenerate_thorax_frame")
            )
            continue
        pred = origin + basis @ loc
        out.append(
            _row(
                fc,
                tgt,
                True,
                float(pred[0]),
                float(pred[1]),
                float(pred[2]),
                f"{t0};{t1};{t2}",
                length,
                "",
            )
        )
    return out
