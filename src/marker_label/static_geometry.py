"""
Z-band by rank and body segment geometry built from static trial (39 markers only).

Used for:
- Z-band by rank: assign dynamic points to bands by Z rank; used in dynamic labeling.
- Segment geometry: arm/hand labeling and later marker gap filling.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from .constants import (
    WHOLE_BODY_39,
    Z_BAND_SIZES,
    N_Z_BANDS,
    WHOLE_BODY_NO_ARM_HAND,
    Z_BAND_SIZES_NO_ARM_HAND,
    Z_BAND_GROUPS_NO_ARM_HAND,
)
from .segments import SEGMENTS


def assign_point_bands_by_z_rank(
    points_frame: np.ndarray,
    band_sizes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Assign each point to a Z-band by rank (descending Z). Used for dynamic trial.

    Valid points (finite) are sorted by Z descending. First band_sizes[11] get band 11,
    next band_sizes[10] get band 10, ... Invalid points get band -1.

    Parameters
    ----------
    points_frame : (n_points, 3)
    band_sizes : length N_Z_BANDS; default Z_BAND_SIZES

    Returns
    -------
    point_bands : (n_points,) int array; values in [0, N_Z_BANDS-1] or -1
    """
    if band_sizes is None:
        band_sizes = Z_BAND_SIZES
    n_pts = points_frame.shape[0]
    point_bands = np.full(n_pts, -1, dtype=np.int32)
    valid = np.isfinite(points_frame).all(axis=1)
    valid_indices = np.where(valid)[0]
    z_vals = points_frame[valid_indices, 2]
    order = np.argsort(-z_vals)  # descending Z: valid_indices[order[0]] is highest Z
    idx = 0
    for b in range(N_Z_BANDS - 1, -1, -1):
        size = band_sizes[b]
        for _ in range(size):
            if idx >= len(valid_indices):
                break
            pi = valid_indices[order[idx]]
            point_bands[pi] = b
            idx += 1
    return point_bands


def build_z_band_by_rank_from_static(
    template: dict[str, np.ndarray],
    whole_body_labels: tuple[str, ...] | list[str] | None = None,
    band_sizes: tuple[int, ...] | None = None,
) -> dict[str, int]:
    """
    Build label -> band index from static template using Z rank only (39 markers).

    Labels present in template and in whole_body_labels are sorted by Z descending.
    The first band_sizes[11] get band 11, next band_sizes[10] get band 10, ... down to band 0.

    Parameters
    ----------
    template : dict label -> (3,) position (e.g. from build_template_from_static)
    whole_body_labels : labels to include; default WHOLE_BODY_39
    band_sizes : length N_Z_BANDS; default Z_BAND_SIZES

    Returns
    -------
    label_to_band : dict str -> int (band index 0..N_Z_BANDS-1)
    """
    if whole_body_labels is None:
        whole_body_labels = WHOLE_BODY_39
    if band_sizes is None:
        band_sizes = Z_BAND_SIZES
    n_bands = len(band_sizes)
    label_to_band: dict[str, int] = {}

    # Collect (label, z) for labels in template and in whole_body set
    wb_set = {str(lab).strip().upper() for lab in whole_body_labels}
    candidates: list[tuple[str, float]] = []
    for lab, pos in template.items():
        key = str(lab).strip().upper()
        if key not in wb_set:
            continue
        if not np.isfinite(pos).all() or pos.shape[0] < 3:
            continue
        candidates.append((lab, float(pos[2])))

    # Sort by Z descending (highest Z first -> band 11)
    candidates.sort(key=lambda x: -x[1])

    # Assign bands: first band_sizes[11] -> 11, next band_sizes[10] -> 10, ...
    idx = 0
    for b in range(n_bands - 1, -1, -1):
        size = band_sizes[b]
        for _ in range(size):
            if idx >= len(candidates):
                break
            label_to_band[candidates[idx][0]] = b
            idx += 1

    return label_to_band


def build_z_band_no_arm_hand_from_static(
    template: dict[str, np.ndarray],
) -> dict[str, int]:
    """
    Build label -> band index from static using 27 markers (no arm/hand).
    Uses anatomical groups (Z_BAND_GROUPS_NO_ARM_HAND) so pelvis, leg, and foot
    are ranked by Z within their group and get the correct band.
    """
    label_to_key_upper = {str(lab).strip().upper(): lab for lab in template.keys()}
    label_to_band: dict[str, int] = {}

    for bands_sizes, group_labels in Z_BAND_GROUPS_NO_ARM_HAND:
        # Collect (template_key, z) for labels in this group that exist in template
        candidates: list[tuple[str, float]] = []
        for lab in group_labels:
            key = str(lab).strip().upper()
            tpl_key = label_to_key_upper.get(key)
            if tpl_key is None:
                continue
            pos = template.get(tpl_key)
            if pos is None or not np.isfinite(pos).all() or pos.shape[0] < 3:
                continue
            candidates.append((tpl_key, float(pos[2])))

        if not candidates:
            continue
        # Sort by Z descending within group
        candidates.sort(key=lambda x: -x[1])
        idx = 0
        for band, size in bands_sizes:
            for _ in range(size):
                if idx >= len(candidates):
                    break
                label_to_band[candidates[idx][0]] = band
                idx += 1

    return label_to_band


def save_z_band_to_json(label_to_band: dict[str, int], path: str) -> None:
    """Write label_to_band to a JSON file (label -> band index)."""
    import json
    # Keys may be strings with spaces from C3D; ensure serializable
    out = {str(k).strip(): int(v) for k, v in label_to_band.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def build_segment_geometry_from_static(
    template: dict[str, np.ndarray],
    segments: dict[str, list[str]] | None = None,
    whole_body_labels: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build body segment geometry from static template (39 markers only).

    For each segment (e.g. L_UpperArm: LSHO, LUPA, LELB), compute positions, axis (proximal to distal),
    length, and which markers belong to it. Used for arm/hand labeling and later gap filling.

    Parameters
    ----------
    template : dict label -> (3,) position
    segments : segment name -> list of marker names; default SEGMENTS (from segments.py)
    whole_body_labels : if set, only segments whose markers are in this set are included

    Returns
    -------
    segment_geometry : list of dicts with keys:
        name : str
        markers : list[str]  (ordered proximal to distal)
        positions : (n, 3) array
        axis : (3,) unit vector proximal -> distal (or None if segment has < 2 valid points)
        length_mm : float (sum of edge lengths)
        edges : list of (label_a, label_b) for each segment edge
    """
    if segments is None:
        segments = SEGMENTS
    if whole_body_labels is not None:
        wb_set = {str(lab).strip().upper() for lab in whole_body_labels}
    else:
        wb_set = None

    out: list[dict[str, Any]] = []
    label_to_key = {str(lab).strip().upper(): lab for lab in template.keys()}

    for seg_name, marker_chain in segments.items():
        # Skip obstacle segment
        if "OBSTACLE" in seg_name.upper():
            continue
        markers_in_seg: list[str] = []
        positions_list: list[np.ndarray] = []
        for m in marker_chain:
            key = str(m).strip().upper()
            if key not in label_to_key:
                continue
            if wb_set is not None and key not in wb_set:
                continue
            lab = label_to_key[key]
            pos = template.get(lab)
            if pos is None or not np.isfinite(pos).all():
                continue
            markers_in_seg.append(lab)
            positions_list.append(np.asarray(pos, dtype=np.float64))

        if len(markers_in_seg) < 2:
            continue

        positions = np.array(positions_list)
        # Axis: from first to last marker (proximal to distal)
        axis = positions[-1] - positions[0]
        length_total = np.linalg.norm(axis)
        if length_total > 1e-6:
            axis = axis / length_total
        else:
            axis = None

        # Sum of edge lengths (in case segment is a chain)
        length_mm = 0.0
        edges: list[tuple[str, str]] = []
        for k in range(len(markers_in_seg) - 1):
            a, b = markers_in_seg[k], markers_in_seg[k + 1]
            length_mm += float(np.linalg.norm(positions[k + 1] - positions[k]))
            edges.append((a, b))

        out.append({
            "name": seg_name,
            "markers": markers_in_seg,
            "positions": positions,
            "axis": axis,
            "length_mm": length_mm,
            "edges": edges,
        })

    return out
