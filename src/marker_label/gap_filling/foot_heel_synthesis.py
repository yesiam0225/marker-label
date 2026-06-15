"""Synthesize missing LHEE/RHEE from static foot geometry when heel columns are absent."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .segment_utils import unique_segment_markers
from .two_marker_static import predict_from_two_anchors, unpack_two_marker_pack

_FOOT_HEEL_SEGMENTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("L_Foot", "LHEE", "LANK", "LTOE", "LTIB"),
    ("R_Foot", "RHEE", "RANK", "RTOE", "RTIB"),
)


def _header_cells_from_stems(stems: Sequence[str]) -> list[str]:
    cells = ["frame", "time"]
    for stem in stems:
        cells.extend([f"{stem}_x", f"{stem}_y", f"{stem}_z"])
    return cells


def _stem_to_triplet_from_stems(stems: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    i = 2
    for stem in stems:
        out[str(stem)] = (i, i + 1, i + 2)
        i += 3
    return out


def insert_marker_into_meta(meta: dict[str, Any], stem: str, *, after: str | None = None) -> bool:
    """
    Insert an empty marker triplet into parsed CSV metadata.

    Returns True when a new column was inserted.
    """
    stem = str(stem).strip()
    if stem in meta["stem_to_col_triplet"]:
        return False
    all_stems: list[str] = list(meta["all_stems"])
    if after is not None and after in all_stems:
        insert_at = all_stems.index(after) + 1
    else:
        insert_at = len(all_stems)
    all_stems_new = all_stems[:insert_at] + [stem] + all_stems[insert_at:]

    old_points = meta["points"]
    n_frames = int(old_points.shape[0])
    old_idx = {s: i for i, s in enumerate(meta["all_stems"])}
    new_points = np.full((n_frames, len(all_stems_new), 3), np.nan, dtype=np.float64)
    for ni, s in enumerate(all_stems_new):
        if s in old_idx:
            new_points[:, ni, :] = old_points[:, old_idx[s], :]

    if after is not None and after in meta["stem_to_col_triplet"]:
        _, _, iz = meta["stem_to_col_triplet"][after]
        col_insert = iz + 1
    else:
        col_insert = len(meta["data_rows"][0]) if meta["data_rows"] else 2

    new_rows: list[list[str]] = []
    for row in meta["data_rows"]:
        padded = list(row)
        while len(padded) < col_insert:
            padded.append("")
        new_rows.append(padded[:col_insert] + ["", "", ""] + padded[col_insert:])

    header_cells = _header_cells_from_stems(all_stems_new)
    buf = io.StringIO()
    csv.writer(buf).writerow(header_cells)
    meta["original_header_line"] = buf.getvalue()
    meta["all_stems"] = all_stems_new
    meta["points"] = new_points
    meta["data_rows"] = new_rows
    meta["stem_to_col_triplet"] = _stem_to_triplet_from_stems(all_stems_new)
    meta["label_to_marker_idx"] = {s: mi for mi, s in enumerate(all_stems_new)}
    meta["n_markers"] = len(all_stems_new)
    return True


def qc_stems_from_meta(meta: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in meta["all_stems"]:
        if str(s).strip().startswith("*"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(str(s))
    return out


def synthesize_missing_foot_heels(
    meta: dict[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    two_marker_offsets: Mapping[str, Mapping[str, tuple]],
    frames: np.ndarray,
    lab_vertical: Sequence[float],
) -> list[dict[str, Any]]:
    """
    Insert and fill LHEE/RHEE when anchors exist in dynamic data and static offsets are available.
    """
    vert = np.asarray(lab_vertical, dtype=np.float64)
    label_to_idx = meta["label_to_marker_idx"]
    points = meta["points"]
    fills: list[dict[str, Any]] = []

    for seg_name, heel, ank, toe, tib in _FOOT_HEEL_SEGMENTS:
        if seg_name not in segment_markers_dict:
            continue
        seg_markers = unique_segment_markers(segment_markers_dict[seg_name])
        if heel not in seg_markers:
            continue
        seg_offsets = two_marker_offsets.get(str(seg_name), {})
        pack = seg_offsets.get(heel)
        if pack is None:
            continue
        if ank not in label_to_idx or toe not in label_to_idx:
            continue

        if heel not in label_to_idx:
            insert_marker_into_meta(meta, heel, after=ank)
            label_to_idx = meta["label_to_marker_idx"]
            points = meta["points"]

        off, flip, shin_ref, o_m, x_m, kind = unpack_two_marker_pack(pack)
        ank_i = label_to_idx[ank]
        toe_i = label_to_idx[toe]
        tib_i = label_to_idx.get(tib)
        heel_i = label_to_idx[heel]
        o_i = label_to_idx.get(o_m)
        x_i = label_to_idx.get(x_m)
        if o_i is None or x_i is None:
            continue

        for f in range(points.shape[0]):
            if np.isfinite(points[f, heel_i, :]).all():
                continue
            if not (
                np.isfinite(points[f, ank_i, :]).all()
                and np.isfinite(points[f, toe_i, :]).all()
            ):
                continue
            if not (
                np.isfinite(points[f, o_i, :]).all()
                and np.isfinite(points[f, x_i, :]).all()
            ):
                continue
            tib_pt = None
            if tib_i is not None and np.isfinite(points[f, tib_i, :]).all():
                tib_pt = points[f, tib_i, :]
            pred = predict_from_two_anchors(
                points[f, o_i, :],
                points[f, x_i, :],
                off,
                vert,
                flip,
                tibia_pt=tib_pt,
                foot_ankle_pt=points[f, ank_i, :],
                foot_toe_pt=points[f, toe_i, :],
                shin_axis_reference=shin_ref,
                segment_kind=kind,
            )
            if not np.isfinite(pred).all():
                continue
            points[f, heel_i, :] = pred
            fills.append(
                {
                    "frame": int(frames[f]),
                    "marker": heel,
                    "method": "static_foot_heel",
                    "success": True,
                    "confidence": "HIGH",
                    "predicted_x": float(pred[0]),
                    "predicted_y": float(pred[1]),
                    "predicted_z": float(pred[2]),
                    "fit_residual_mm": 0.0,
                    "source_markers": f"{o_m};{x_m};static",
                    "gap_length": int(points.shape[0]),
                    "reason": "missing_heel_column",
                }
            )
    return fills
