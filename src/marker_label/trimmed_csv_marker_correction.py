"""
Post-trim marker correction for labeled motion-capture CSVs.

Uses reference geometry from the **original** trial (``original_best_frame`` argument, or the
original CSV's ``*.csv.bestframe`` when that argument is ``None``) and tier-based
rules so upper-body markers (not verified by leg-only trim) can be envelope-filtered, swapped,
rejected, or reassigned from unlabeled columns.

Staged pipeline: **envelope (1)** → **pelvis combinatorial swap (2)** → chain / same / cross-segment swaps
→ rejection → unlabeled assignment → continuity. Stage 5 assigns unlabeled points with
**priority 1** for markers invalidated in stages 1 or 4, then **priority 2** for originally
occluded (missing) markers, so recoverable points are not taken by unrelated gaps first.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .trial_trim import (
    DEFAULT_COLLINEARITY_EPS_MM,
    bestframe_sidecar_path,
    compute_reference_geometry,
    kabsch,
    parse_labeled_csv,
    segment_markers_dict_for_trim_preset,
)
from .trial_trim_combine import compute_subject_envelope
from .walking_setup import determine_setup

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "envelope_radius_mm": 1500.0,
    "envelope_min_visible_markers": 5,
    "walking_path_y_margin_mm": 500.0,
    "pelvis_swap_ratio": 0.5,
    "pelvis_swap_high_residual_mm": 10.0,
    "pelvis_swap_medium_residual_mm": 20.0,
    "swap_ratio": 0.5,
    "swap_absolute_max_mm": 50.0,
    "wrong_label_distance_mm": 50.0,
    "tier_suspicious_threshold_mm": {1: 25, 2: 20, 3: 30},
    "suspicious_neighbor_window": 5,
    "assignment_threshold_mm": 30.0,
    "velocity_threshold_per_frame_mm": 50.0,
    "collinearity_eps_mm": DEFAULT_COLLINEARITY_EPS_MM,
    # If the requested frame lacks finite markers for some segment, search this many
    # trial frames (by frame column) away before failing. Set to 0 to require an exact row only.
    "reference_frame_search_radius": 150,
    "chain_swap_y_separation_min_mm": 100.0,
    "chain_swap_side_offset_min_mm": 50.0,
    "chain_swap_min_consecutive_frames": 10,
    "chain_swap_min_visible_per_side": 2,
}

DEFAULT_BILATERAL_CHAINS: dict[str, tuple[list[str], list[str]]] = {
    "arm": (
        ["LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN"],
        ["RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN"],
    ),
}

DEFAULT_SAME_SEGMENT_SWAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("T10", "STRN"),
    ("C7", "CLAV"),
    ("LFHD", "RFHD"),
    ("LBHD", "RBHD"),
    ("LFHD", "LBHD"),
    ("RFHD", "RBHD"),
    ("LSHO", "RSHO"),
    ("LWRA", "LWRB"),
    ("RWRA", "RWRB"),
)

def filter_same_segment_swap_pairs(
    base: Sequence[tuple[str, str]],
    exclude_pairs: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    """Return ``base`` without any pair whose unordered marker set matches an entry in ``exclude_pairs``."""
    if not exclude_pairs:
        return tuple(base)
    banned = {frozenset({str(a).strip(), str(b).strip()}) for a, b in exclude_pairs}
    out: list[tuple[str, str]] = []
    for m1, m2 in base:
        if frozenset({str(m1).strip(), str(m2).strip()}) in banned:
            continue
        out.append((str(m1).strip(), str(m2).strip()))
    return tuple(out)


DEFAULT_CROSS_SEGMENT_SWAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("LASI", "LWRA"),
    ("LASI", "LWRB"),
    ("LASI", "LFIN"),
    ("RASI", "RWRA"),
    ("RASI", "RWRB"),
    ("RASI", "RFIN"),
    ("LPSI", "LWRA"),
    ("LPSI", "LWRB"),
    ("RPSI", "RWRA"),
    ("RPSI", "RWRB"),
    ("LASI", "STRN"),
    ("RASI", "STRN"),
    ("LPSI", "T10"),
    ("RPSI", "T10"),
    ("LWRA", "LTHI"),
    ("LWRB", "LTHI"),
    ("LFIN", "LTHI"),
    ("RWRA", "RTHI"),
    ("RWRB", "RTHI"),
    ("RFIN", "RTHI"),
    ("LWRA", "LKNE"),
    ("RWRA", "RKNE"),
)


def load_best_frame(bestframe_path: str | Path) -> int:
    """Read the integer stored in a ``*.csv.bestframe`` sidecar (same convention as :mod:`marker_label.trial_trim`)."""
    p = Path(bestframe_path)
    if not p.is_file():
        raise FileNotFoundError(f"Best-frame sidecar not found: {p}")
    text = p.read_text().strip().split()
    if not text:
        raise ValueError(f"Empty bestframe file: {p}")
    return int(text[0])


def _meta_markers_dict(meta: dict[str, Any]) -> dict[str, np.ndarray]:
    points: np.ndarray = meta["points"]
    stems: list[str] = meta["all_stems"]
    return {str(st): points[:, i, :] for i, st in enumerate(stems)}


def determine_chain_swap_setup(
    meta_orig: dict[str, Any],
) -> dict[str, Any]:
    """
    Auto-detect ML axis and left-side sign for chain swap detection.

    Uses the same setup detection logic as trim/combine (without obstacle requirement).
    """
    return determine_setup(_meta_markers_dict(meta_orig), obstacle_pair=None, warnings_list=[])


def find_consecutive_intervals(bool_array: Sequence[bool], min_length: int) -> list[tuple[int, int]]:
    """Return ``(start_row, end_row)`` inclusive for each run of True with length >= ``min_length`` (row indices)."""
    intervals: list[tuple[int, int]] = []
    in_run = False
    run_start = 0
    arr = list(bool_array)
    n = len(arr)
    k = int(min_length)
    if k < 1:
        return intervals
    for i, val in enumerate(arr):
        if val and not in_run:
            in_run = True
            run_start = i
        elif not val and in_run:
            if i - run_start >= k:
                intervals.append((run_start, i - 1))
            in_run = False
    if in_run and n - run_start >= k:
        intervals.append((run_start, n - 1))
    return intervals


def detect_chain_swap_at_frame(
    points: np.ndarray,
    row_f: int,
    label_to_idx: Mapping[str, int],
    chain_left: Sequence[str],
    chain_right: Sequence[str],
    ml_axis: int,
    left_ml_sign: int,
    cfg: Mapping[str, Any],
) -> bool:
    """
    True if both bilateral chains appear on the wrong ML side of the pelvis midline.

    Requires finite ``LASI`` and ``RASI`` at ``row_f``. Markers missing from the CSV are skipped.
    """
    if "LASI" not in label_to_idx or "RASI" not in label_to_idx:
        return False
    i_la, i_ra = label_to_idx["LASI"], label_to_idx["RASI"]
    p_la = points[row_f, i_la, :]
    p_ra = points[row_f, i_ra, :]
    if not (np.isfinite(p_la).all() and np.isfinite(p_ra).all()):
        return False

    axis = int(ml_axis)
    pelvis_y = float((p_la[axis] + p_ra[axis]) / 2.0)

    left_present = [str(m).strip() for m in chain_left if str(m).strip() in label_to_idx]
    right_present = [str(m).strip() for m in chain_right if str(m).strip() in label_to_idx]

    min_vis = int(cfg["chain_swap_min_visible_per_side"])
    left_y: list[float] = []
    for m in left_present:
        mi = label_to_idx[m]
        q = points[row_f, mi, :]
        if np.isfinite(q).all():
            left_y.append(float(q[axis]))
    right_y: list[float] = []
    for m in right_present:
        mi = label_to_idx[m]
        q = points[row_f, mi, :]
        if np.isfinite(q).all():
            right_y.append(float(q[axis]))

    if len(left_y) < min_vis or len(right_y) < min_vis:
        return False

    left_mean_y = float(np.mean(left_y))
    right_mean_y = float(np.mean(right_y))
    y_sep_min = float(cfg["chain_swap_y_separation_min_mm"])
    if abs(left_mean_y - right_mean_y) < y_sep_min:
        return False

    off_min = float(cfg["chain_swap_side_offset_min_mm"])
    left_offset = left_mean_y - pelvis_y
    right_offset = right_mean_y - pelvis_y
    if abs(left_offset) < off_min or abs(right_offset) < off_min:
        return False

    left_actual_sign = int(np.sign(left_offset))
    right_actual_sign = int(np.sign(right_offset))
    if left_actual_sign == 0 or right_actual_sign == 0:
        return False

    expected_left = int(left_ml_sign)
    expected_right = -int(left_ml_sign)
    if left_actual_sign != expected_left and right_actual_sign != expected_right:
        return True
    return False


def _aligned_chain_pairs(
    chain_left: Sequence[str],
    chain_right: Sequence[str],
    label_to_idx: Mapping[str, int],
) -> list[tuple[str, str]]:
    """(left, right) pairs in list order where both stems exist in the CSV."""
    pairs: list[tuple[str, str]] = []
    for ml, mr in zip(chain_left, chain_right, strict=False):
        a, b = str(ml).strip(), str(mr).strip()
        if a in label_to_idx and b in label_to_idx:
            pairs.append((a, b))
    return pairs


def _log_chain_interval_summary(
    frames: np.ndarray,
    start_row: int,
    end_row: int,
    chain_name: str,
    left_y_sign: int,
    pairs_swapped: Sequence[tuple[str, str]],
    reason: str,
) -> dict[str, Any]:
    """One correction-log row summarizing a full bilateral chain swap interval."""
    return {
        "frame": int(frames[start_row]),
        "marker": f"{chain_name}_chain",
        "stage": 6,
        "action": "full_chain_swap_interval",
        "tier": 3,
        "confidence": "HIGH",
        "original_x": "",
        "original_y": "",
        "original_z": "",
        "new_x": "",
        "new_y": "",
        "new_z": "",
        "expected_x": "",
        "expected_y": "",
        "expected_z": "",
        "reason": reason,
        "interval_end_frame": int(frames[end_row]),
        "interval_duration_frames": int(end_row - start_row + 1),
        "markers_swapped": ",".join(f"{a}<->{b}" for a, b in pairs_swapped),
        "left_y_sign_used": int(left_y_sign),
    }


def apply_full_chain_swap_detection(
    points: np.ndarray,
    meta: dict[str, Any],
    meta_orig: dict[str, Any],
    bilateral_chains: Mapping[str, tuple[Sequence[str], Sequence[str]]],
    original_frame_for_sign: int,
    *,
    fallback_original_frame_for_sign: int | None,
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """
    Detect extended left-right chain swaps using lab-Y vs pelvis midline (post-pelvis correction).

    Returns ``(log_entries, interval_summaries, n_intervals, n_frames_affected)``.
    """
    if not bilateral_chains:
        return [], [], 0, 0

    setup = determine_chain_swap_setup(meta_orig)
    left_y_sign = int(setup["left_ml_sign"])
    ml_axis = int(setup["ml_axis"])
    frames: np.ndarray = meta["frames"]
    n_frames = int(meta["n_frames"])

    log_entries: list[dict[str, Any]] = []
    interval_summaries: list[dict[str, Any]] = []
    min_run = int(cfg["chain_swap_min_consecutive_frames"])
    frames_affected = 0

    l2i: dict[str, int] = meta["label_to_marker_idx"]

    for chain_name, (chain_left, chain_right) in bilateral_chains.items():
        pairs = _aligned_chain_pairs(chain_left, chain_right, l2i)
        if not pairs:
            continue

        flags = np.zeros(n_frames, dtype=bool)
        for f in range(n_frames):
            flags[f] = detect_chain_swap_at_frame(
                points, f, l2i, chain_left, chain_right, ml_axis, left_y_sign, cfg
            )

        for start_row, end_row in find_consecutive_intervals(flags.tolist(), min_run):
            for f in range(start_row, end_row + 1):
                for m_left, m_right in pairs:
                    il, ir = l2i[m_left], l2i[m_right]
                    left_pos = points[f, il, :].copy()
                    right_pos = points[f, ir, :].copy()
                    points[f, il, :] = right_pos
                    points[f, ir, :] = left_pos
            frames_affected += int(end_row - start_row + 1)

            reason = (
                f"Both chains wrong Y side vs pelvis; interval rows [{start_row},{end_row}] "
                f"frames [{int(frames[start_row])},{int(frames[end_row])}]"
            )
            log_entries.append(
                _log_chain_interval_summary(
                    frames, start_row, end_row, chain_name, left_y_sign, pairs, reason
                )
            )
            interval_summaries.append(
                {
                    "chain_name": chain_name,
                    "start_frame": int(frames[start_row]),
                    "end_frame": int(frames[end_row]),
                    "duration_frames": int(end_row - start_row + 1),
                    "left_y_sign_used": int(left_y_sign),
                }
            )

    return log_entries, interval_summaries, len(interval_summaries), frames_affected


def _is_unlabeled_stem(stem: str) -> bool:
    s = str(stem).strip()
    if s.startswith("*"):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    return False


def _segment_key_ci(segment_markers_dict: Mapping[str, Sequence[str]], name: str) -> str:
    want = str(name).strip().lower()
    for k in segment_markers_dict:
        if str(k).strip().lower() == want:
            return str(k)
    raise KeyError(f"No segment named {name!r} in segment_markers_dict")


def _pelvis_four_markers(
    segment_markers_dict: Mapping[str, Sequence[str]],
    pelvis_segment_name: str,
) -> list[str] | None:
    try:
        key = _segment_key_ci(segment_markers_dict, pelvis_segment_name)
    except KeyError:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for raw in segment_markers_dict[key]:
        s = str(raw).strip()
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) == 4:
            return out
    return None


def classify_tier_from_segment_name(segment_name: str) -> int:
    """Return tier 1 (verified-style leg), 2 (CoM-critical), or 3 (supportive)."""
    seg_lower = str(segment_name).lower()
    verified_keywords = ("thigh", "shank", "tibia", "foot", "leg")
    tier2_keywords = ("pelvis", "trunk", "torso")
    if any(kw in seg_lower for kw in verified_keywords):
        return 1
    if any(kw in seg_lower for kw in tier2_keywords):
        return 2
    return 3


def auto_classify_tiers(
    segment_markers_dict: Mapping[str, Sequence[str]],
    verified_segments: Sequence[str] | None,
) -> dict[str, int]:
    """Map each marker stem to tier 1, 2, or 3."""
    vs_raw = list(verified_segments) if verified_segments is not None else []
    vs = {str(s).strip().lower() for s in vs_raw}
    out: dict[str, int] = {}
    for seg_name, names in segment_markers_dict.items():
        seg_l = str(seg_name).strip().lower()
        if vs:
            tier = 1 if seg_l in vs else classify_tier_from_segment_name(seg_name)
        else:
            tier = classify_tier_from_segment_name(seg_name)
        for raw in names:
            stem = str(raw).strip()
            if stem not in out:
                out[stem] = tier
    return out


def _stem_segment_map(segment_markers_dict: Mapping[str, Sequence[str]]) -> dict[str, str]:
    m: dict[str, str] = {}
    for seg, names in segment_markers_dict.items():
        for raw in names:
            m[str(raw).strip()] = str(seg)
    return m


def _ref_stem_row_index(ref_seg: Mapping[str, Any]) -> dict[str, int]:
    names: list[str] = list(ref_seg["marker_names"])
    return {str(n).strip(): i for i, n in enumerate(names)}


def compute_expected_for_stem(
    points: np.ndarray,
    f: int,
    stem: str,
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    label_to_idx: Mapping[str, int],
    exclude_markers: frozenset[str],
) -> np.ndarray | None:
    """Expected global (3,) for ``stem`` using its segment Kabsch, excluding ``exclude_markers``."""
    smap = _stem_segment_map(segment_markers_dict)
    if stem not in smap:
        return None
    seg_name = smap[stem]
    if seg_name not in reference_geometry:
        return None
    ref = reference_geometry[seg_name]
    stem_row = _ref_stem_row_index(ref)
    if stem not in stem_row:
        return None
    P_loc = ref["P_local"]
    seg_names = [str(x).strip() for x in segment_markers_dict[seg_name]]
    rows_loc: list[np.ndarray] = []
    rows_glob: list[np.ndarray] = []
    used: list[str] = []
    for s in seg_names:
        if s in exclude_markers or s == stem or s not in label_to_idx:
            continue
        if s not in stem_row:
            continue
        mi = label_to_idx[s]
        g = points[f, mi, :]
        if not np.isfinite(g).all():
            continue
        rows_loc.append(np.asarray(P_loc[stem_row[s]], dtype=np.float64))
        rows_glob.append(np.asarray(g, dtype=np.float64))
        used.append(s)
    if len(rows_loc) < 3:
        return None
    a_loc = np.stack(rows_loc, axis=0)
    a_glob = np.stack(rows_glob, axis=0)
    r, t = kabsch(a_loc, a_glob)
    pl = np.asarray(P_loc[stem_row[stem]], dtype=np.float64)
    return pl @ r.T + t


def _pairwise_swap_error(
    points: np.ndarray,
    f: int,
    m1: str,
    m2: str,
    seg_name: str,
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    label_to_idx: Mapping[str, int],
) -> tuple[float, float] | None:
    """Return (error_normal, error_swap) in mm, or None if not computable."""
    excl_n = frozenset({m1, m2})
    e1n = compute_expected_for_stem(points, f, m1, segment_markers_dict, reference_geometry, label_to_idx, excl_n)
    e2n = compute_expected_for_stem(points, f, m2, segment_markers_dict, reference_geometry, label_to_idx, excl_n)
    if e1n is None or e2n is None:
        return None
    i1, i2 = label_to_idx[m1], label_to_idx[m2]
    p1, p2 = points[f, i1, :], points[f, i2, :]
    if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
        return None
    err_n = float(np.linalg.norm(p1 - e1n) + np.linalg.norm(p2 - e2n))
    err_s = float(np.linalg.norm(p1 - e2n) + np.linalg.norm(p2 - e1n))
    return err_n, err_s


def _confidence_from_residual_drop(high_mm: float, med_mm: float, best: float, identity: float) -> str:
    if best < high_mm and identity > 50.0:
        return "HIGH"
    if best < med_mm and identity > 30.0:
        return "MEDIUM"
    return "LOW"


def _envelope_invalidation_reason(
    p: np.ndarray,
    centroid: np.ndarray,
    cfg: Mapping[str, Any],
    env_info: Mapping[str, Any],
) -> str:
    """Return ``envelope_outside`` or ``walking_path_y_outside`` for a flagged envelope violation."""
    env_r = float(cfg["envelope_radius_mm"])
    viol_sphere = bool(np.isfinite(centroid).all() and float(np.linalg.norm(p - centroid)) > env_r)
    y_band = env_info.get("walking_y_band_mm")
    viol_y = False
    if y_band is not None and len(y_band) == 2:
        y_lo, y_hi = float(y_band[0]), float(y_band[1])
        viol_y = bool(p[1] < y_lo or p[1] > y_hi)
    if viol_sphere:
        return "envelope_outside"
    if viol_y:
        return "walking_path_y_outside"
    return "envelope_outside"


def apply_envelope_filter(
    points: np.ndarray,
    meta: dict[str, Any],
    marker_tiers: Mapping[str, int],
    body_stems: Sequence[str],
    envelope_outside: np.ndarray,
    centroids: np.ndarray,
    env_info: Mapping[str, Any],
    cfg: Mapping[str, Any],
    invalidated_marker_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    frames: np.ndarray = meta["frames"]
    n_frames = int(meta["n_frames"])
    for f in range(n_frames):
        c = centroids[f]
        for j, stem in enumerate(body_stems):
            if stem not in label_to_idx:
                continue
            tier = marker_tiers.get(stem, 3)
            if tier == 1:
                continue
            mi = label_to_idx[stem]
            p = points[f, mi, :]
            if not np.isfinite(p).all():
                continue
            if j < envelope_outside.shape[1] and envelope_outside[f, j]:
                prev = p.copy()
                inv_reason = _envelope_invalidation_reason(p, c, cfg, env_info)
                points[f, mi, :] = np.nan
                invalidated_marker_frames.append(
                    {
                        "frame": int(frames[f]),
                        "marker": stem,
                        "original_position": prev.copy(),
                        "invalidation_reason": inv_reason,
                        "invalidation_stage": 1,
                        "tier": int(tier),
                    }
                )
                log.append(
                    _log_row(
                        int(frames[f]),
                        stem,
                        1,
                        "envelope_invalid",
                        tier,
                        "MEDIUM",
                        prev,
                        points[f, mi, :],
                        np.full(3, np.nan),
                        "outside_envelope_or_y_band",
                    )
                )
    return log


def _tier_priority_sort_key(tier: int) -> tuple[int, int]:
    """Lower tuple sorts earlier: tier 2, then 3, then 1 (tier-1 markers processed last)."""
    t = int(tier)
    group = {2: 0, 3: 1, 1: 2}.get(t, 1)
    return (group, t)


def _log_row(
    frame: int,
    marker: str,
    stage: int,
    action: str,
    tier: int,
    confidence: str,
    orig: np.ndarray,
    new: np.ndarray,
    expected: np.ndarray,
    reason: str,
    *,
    priority: str | int = "",
    invalidation_reason: str = "",
    source_unlabeled: str = "",
    distance_to_expected: str | float = "",
) -> dict[str, Any]:
    dte = (
        float(distance_to_expected)
        if isinstance(distance_to_expected, (int, float)) and np.isfinite(float(distance_to_expected))
        else (str(distance_to_expected) if distance_to_expected != "" else "")
    )
    return {
        "frame": frame,
        "marker": marker,
        "stage": stage,
        "action": action,
        "tier": tier,
        "confidence": confidence,
        "original_x": float(orig[0]) if np.isfinite(orig[0]) else "",
        "original_y": float(orig[1]) if np.isfinite(orig[1]) else "",
        "original_z": float(orig[2]) if np.isfinite(orig[2]) else "",
        "new_x": float(new[0]) if np.isfinite(new[0]) else "",
        "new_y": float(new[1]) if np.isfinite(new[1]) else "",
        "new_z": float(new[2]) if np.isfinite(new[2]) else "",
        "expected_x": float(expected[0]) if np.isfinite(expected[0]) else "",
        "expected_y": float(expected[1]) if np.isfinite(expected[1]) else "",
        "expected_z": float(expected[2]) if np.isfinite(expected[2]) else "",
        "reason": reason,
        "priority": priority if priority != "" else "",
        "invalidation_reason": invalidation_reason,
        "source_unlabeled": source_unlabeled,
        "distance_to_expected": dte,
    }


def apply_pelvis_combinatorial_swap(
    points: np.ndarray,
    meta: dict[str, Any],
    pelvis_markers: list[str],
    reference_geometry: Mapping[str, Any],
    pelvis_seg_key: str,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    frames: np.ndarray = meta["frames"]
    n_frames = int(meta["n_frames"])
    ref = reference_geometry[pelvis_seg_key]
    stem_row = _ref_stem_row_index(ref)
    P_loc = ref["P_local"]
    loc_by_phys = [np.asarray(P_loc[stem_row[m]], dtype=np.float64) for m in pelvis_markers]

    ratio = float(cfg["pelvis_swap_ratio"])
    high_mm = float(cfg["pelvis_swap_high_residual_mm"])
    med_mm = float(cfg["pelvis_swap_medium_residual_mm"])

    for f in range(n_frames):
        cur: list[np.ndarray] = []
        for m in pelvis_markers:
            mi = label_to_idx[m]
            q = points[f, mi, :]
            if not np.isfinite(q).all():
                cur = []
                break
            cur.append(np.asarray(q, dtype=np.float64))
        if len(cur) != 4:
            continue
        P_cur = np.stack(cur, axis=0)

        best_perm: tuple[int, ...] | None = None
        best_total = np.inf
        identity_total: float | None = None
        for perm in itertools.permutations((0, 1, 2, 3), 4):
            P_ref = np.stack([loc_by_phys[perm[j]] for j in range(4)], axis=0)
            r, t = kabsch(P_ref, P_cur)
            pred = P_ref @ r.T + t
            total = float(np.sum(np.linalg.norm(P_cur - pred, axis=1)))
            if perm == (0, 1, 2, 3):
                identity_total = total
            if total < best_total:
                best_total = total
                best_perm = perm
        if best_perm is None or identity_total is None or best_perm == (0, 1, 2, 3):
            continue
        if not (best_total < identity_total * ratio and best_total < float(cfg["swap_absolute_max_mm"]) * 4):
            continue
        inv = [0] * 4
        for slot in range(4):
            inv[best_perm[slot]] = slot
        conf = _confidence_from_residual_drop(high_mm, med_mm, best_total, identity_total)
        for j in range(4):
            mj = pelvis_markers[j]
            mi = label_to_idx[mj]
            prev = points[f, mi, :].copy()
            src = inv[j]
            points[f, mi, :] = cur[src]
            log.append(
                _log_row(
                    int(frames[f]),
                    mj,
                    2,
                    "pelvis_combinatorial_swap",
                    2,
                    conf,
                    prev,
                    points[f, mi, :],
                    cur[j],
                    f"perm={best_perm}",
                )
            )
    return log


def apply_same_segment_swap(
    points: np.ndarray,
    meta: dict[str, Any],
    swap_pairs: Sequence[tuple[str, str]],
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    cfg: Mapping[str, Any],
    marker_tiers: Mapping[str, int],
) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    frames: np.ndarray = meta["frames"]
    smap = _stem_segment_map(segment_markers_dict)
    ratio = float(cfg["swap_ratio"])
    abs_max = float(cfg["swap_absolute_max_mm"])
    n_frames = int(meta["n_frames"])
    for m1, m2 in swap_pairs:
        if m1 not in label_to_idx or m2 not in label_to_idx:
            continue
        if m1 not in smap or m2 not in smap or smap[m1] != smap[m2]:
            continue
        seg_name = smap[m1]
        if seg_name not in reference_geometry:
            continue
        for f in range(n_frames):
            pe = _pairwise_swap_error(
                points, f, m1, m2, seg_name, segment_markers_dict, reference_geometry, label_to_idx
            )
            if pe is None:
                continue
            err_n, err_s = pe
            if err_s >= ratio * err_n or err_s >= abs_max * 2:
                continue
            i1, i2 = label_to_idx[m1], label_to_idx[m2]
            p1, p2 = points[f, i1, :].copy(), points[f, i2, :].copy()
            e1 = compute_expected_for_stem(
                points, f, m1, segment_markers_dict, reference_geometry, label_to_idx, frozenset({m1, m2})
            )
            e2 = compute_expected_for_stem(
                points, f, m2, segment_markers_dict, reference_geometry, label_to_idx, frozenset({m1, m2})
            )
            points[f, i1, :], points[f, i2, :] = p2.copy(), p1.copy()
            tier = max(marker_tiers.get(m1, 3), marker_tiers.get(m2, 3))
            conf = "HIGH" if err_s < 5.0 else "MEDIUM" if err_s < 15.0 else "LOW"
            log.append(
                _log_row(
                    int(frames[f]),
                    m1,
                    2,
                    "same_segment_swap",
                    tier,
                    conf,
                    p1,
                    points[f, i1, :],
                    e1 if e1 is not None else np.full(3, np.nan),
                    f"swap_with={m2}",
                )
            )
            log.append(
                _log_row(
                    int(frames[f]),
                    m2,
                    2,
                    "same_segment_swap",
                    tier,
                    conf,
                    p2,
                    points[f, i2, :],
                    e2 if e2 is not None else np.full(3, np.nan),
                    f"swap_with={m1}",
                )
            )
    return log


def apply_cross_segment_swap(
    points: np.ndarray,
    meta: dict[str, Any],
    swap_pairs: Sequence[tuple[str, str]],
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    marker_tiers: Mapping[str, int],
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    frames: np.ndarray = meta["frames"]
    smap = _stem_segment_map(segment_markers_dict)
    ratio = float(cfg["swap_ratio"])
    abs_max = float(cfg["swap_absolute_max_mm"])
    wl_mm = float(cfg["wrong_label_distance_mm"])
    n_frames = int(meta["n_frames"])
    for m1, m2 in swap_pairs:
        if m1 not in label_to_idx or m2 not in label_to_idx:
            continue
        if m1 not in smap or m2 not in smap:
            continue
        for f in range(n_frames):
            t1, t2 = marker_tiers.get(m1, 3), marker_tiers.get(m2, 3)
            i1, i2 = label_to_idx[m1], label_to_idx[m2]
            p1, p2 = points[f, i1, :], points[f, i2, :]
            if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                continue
            if {t1, t2} == {1, 1}:
                continue
            if t1 == 1 and t2 != 1:
                e1 = compute_expected_for_stem(
                    points, f, m1, segment_markers_dict, reference_geometry, label_to_idx, frozenset()
                )
                e2 = compute_expected_for_stem(
                    points, f, m2, segment_markers_dict, reference_geometry, label_to_idx, frozenset()
                )
                if e1 is None:
                    continue
                d_tier1 = float(np.linalg.norm(p1 - e1))
                d_m2_to_e1 = float(np.linalg.norm(p2 - e1))
                if d_m2_to_e1 < wl_mm and d_tier1 < wl_mm * 0.5:
                    prev = p2.copy()
                    points[f, i2, :] = np.nan
                    log.append(
                        _log_row(
                            int(frames[f]),
                            m2,
                            3,
                            "wrong_label_rejection",
                            t2,
                            "MEDIUM",
                            prev,
                            points[f, i2, :],
                            e1,
                            f"near_tier1_expected_{m1}",
                        )
                    )
                continue
            if t2 == 1 and t1 != 1:
                e2 = compute_expected_for_stem(
                    points, f, m2, segment_markers_dict, reference_geometry, label_to_idx, frozenset()
                )
                if e2 is None:
                    continue
                d_p1_e2 = float(np.linalg.norm(p1 - e2))
                d_p2_e2 = float(np.linalg.norm(p2 - e2))
                if d_p1_e2 < wl_mm and d_p2_e2 < wl_mm * 0.5:
                    prev = p1.copy()
                    points[f, i1, :] = np.nan
                    log.append(
                        _log_row(
                            int(frames[f]),
                            m1,
                            3,
                            "wrong_label_rejection",
                            t1,
                            "MEDIUM",
                            prev,
                            points[f, i1, :],
                            e2,
                            f"near_tier1_expected_{m2}",
                        )
                    )
                continue
            excl = frozenset({m1, m2})
            e1 = compute_expected_for_stem(
                points, f, m1, segment_markers_dict, reference_geometry, label_to_idx, excl
            )
            e2 = compute_expected_for_stem(
                points, f, m2, segment_markers_dict, reference_geometry, label_to_idx, excl
            )
            if e1 is None or e2 is None:
                continue
            err_n = float(np.linalg.norm(p1 - e1) + np.linalg.norm(p2 - e2))
            err_s = float(np.linalg.norm(p1 - e2) + np.linalg.norm(p2 - e1))
            if err_s >= ratio * err_n or err_s >= abs_max * 2:
                continue
            op1, op2 = p1.copy(), p2.copy()
            points[f, i1, :], points[f, i2, :] = op2.copy(), op1.copy()
            tier = max(t1, t2)
            conf = "MEDIUM"
            log.append(
                _log_row(
                    int(frames[f]),
                    m1,
                    3,
                    "cross_segment_swap",
                    tier,
                    conf,
                    op1,
                    points[f, i1, :],
                    e1 if e1 is not None else np.full(3, np.nan),
                    f"swap_with={m2}",
                )
            )
            log.append(
                _log_row(
                    int(frames[f]),
                    m2,
                    3,
                    "cross_segment_swap",
                    tier,
                    conf,
                    op2,
                    points[f, i2, :],
                    e2 if e2 is not None else np.full(3, np.nan),
                    f"swap_with={m1}",
                )
            )
    return log


def apply_single_frame_rejection(
    points: np.ndarray,
    meta: dict[str, Any],
    marker_tiers: Mapping[str, int],
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    cfg: Mapping[str, Any],
    sustained_warnings: list[str],
    invalidated_marker_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    frames: np.ndarray = meta["frames"]
    n_frames = int(meta["n_frames"])
    th = {int(k): float(v) for k, v in dict(cfg["tier_suspicious_threshold_mm"]).items()}
    win = int(cfg["suspicious_neighbor_window"])
    for stem, mi in label_to_idx.items():
        if _is_unlabeled_stem(stem) or stem.startswith("OBSTACLE"):
            continue
        tier = marker_tiers.get(stem, 3)
        thr = th.get(tier, 30.0)
        for f in range(n_frames):
            p = points[f, mi, :]
            if not np.isfinite(p).all():
                continue
            excl = frozenset({stem})
            exp = compute_expected_for_stem(
                points, f, stem, segment_markers_dict, reference_geometry, label_to_idx, excl
            )
            if exp is None:
                continue
            err = float(np.linalg.norm(p - exp))
            if err <= thr:
                continue
            neigh: list[float] = []
            for d in range(-win, win + 1):
                if d == 0:
                    continue
                fn = f + d
                if fn < 0 or fn >= n_frames:
                    continue
                p2 = points[fn, mi, :]
                if not np.isfinite(p2).all():
                    continue
                e2 = compute_expected_for_stem(
                    points, fn, stem, segment_markers_dict, reference_geometry, label_to_idx, excl
                )
                if e2 is None:
                    continue
                neigh.append(float(np.linalg.norm(p2 - e2)))
            if neigh:
                med = float(np.median(neigh))
            else:
                med = 0.0
            if med < thr:
                prev = p.copy()
                points[f, mi, :] = np.nan
                invalidated_marker_frames.append(
                    {
                        "frame": int(frames[f]),
                        "marker": stem,
                        "original_position": prev.copy(),
                        "invalidation_reason": "single_frame_outlier",
                        "invalidation_stage": 4,
                        "tier": int(tier),
                    }
                )
                log.append(
                    _log_row(
                        int(frames[f]),
                        stem,
                        4,
                        "single_frame_outlier_rejection",
                        tier,
                        "MEDIUM",
                        prev,
                        points[f, mi, :],
                        exp,
                        f"err={err:.1f}_median_neighbor={med:.1f}",
                    )
                )
            else:
                sustained_warnings.append(
                    f"Sustained high error {stem} frame={int(frames[f])} err={err:.1f} median_neighbor={med:.1f}"
                )
    return log


def try_unlabeled_replacement(
    entry: Mapping[str, Any],
    priority: int,
    points: np.ndarray,
    meta: dict[str, Any],
    unlabeled_indices: Sequence[int],
    unlabeled_stems: Sequence[str],
    marker_tiers: Mapping[str, int],
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    centroids: np.ndarray,
    consumed_unlabeled_per_frame: dict[int, set[str]],
    frame_to_row: Mapping[int, int],
    idx_to_stem: Mapping[int, str],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Try to fill one missing marker row from an unlabeled column; returns one log dict (always)."""
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    frame_col = int(entry["frame"])
    stem = str(entry["marker"])
    tier = int(entry.get("tier", marker_tiers.get(stem, 3)))
    row_f = frame_to_row.get(frame_col)
    if row_f is None or stem not in label_to_idx:
        return {
            "frame": frame_col,
            "marker": stem,
            "stage": 5,
            "action": "no_expected_skip",
            "tier": tier,
            "confidence": "LOW",
            "original_x": "",
            "original_y": "",
            "original_z": "",
            "new_x": "",
            "new_y": "",
            "new_z": "",
            "expected_x": "",
            "expected_y": "",
            "expected_z": "",
            "reason": "missing_frame_row_or_marker",
            "priority": priority,
            "invalidation_reason": entry.get("invalidation_reason") if priority == 1 else "",
            "source_unlabeled": "",
            "distance_to_expected": "",
        }
    mi = label_to_idx[stem]
    prev = points[row_f, mi, :].copy()
    if np.isfinite(prev).all():
        return {
            "frame": frame_col,
            "marker": stem,
            "stage": 5,
            "action": "no_longer_missing_skip",
            "tier": tier,
            "confidence": "LOW",
            "original_x": float(prev[0]),
            "original_y": float(prev[1]),
            "original_z": float(prev[2]),
            "new_x": float(prev[0]),
            "new_y": float(prev[1]),
            "new_z": float(prev[2]),
            "expected_x": "",
            "expected_y": "",
            "expected_z": "",
            "reason": "marker_already_finite_before_stage5",
            "priority": priority,
            "invalidation_reason": entry.get("invalidation_reason") if priority == 1 else "",
            "source_unlabeled": "",
            "distance_to_expected": "",
        }

    exp = compute_expected_for_stem(
        points,
        row_f,
        stem,
        segment_markers_dict,
        reference_geometry,
        label_to_idx,
        frozenset({stem}),
    )
    if exp is None or not np.isfinite(exp).all():
        return {
            "frame": frame_col,
            "marker": stem,
            "stage": 5,
            "action": "no_expected_skip",
            "tier": tier,
            "confidence": "LOW",
            "original_x": "",
            "original_y": "",
            "original_z": "",
            "new_x": "",
            "new_y": "",
            "new_z": "",
            "expected_x": float(exp[0]) if exp is not None and np.isfinite(exp[0]) else "",
            "expected_y": float(exp[1]) if exp is not None and np.isfinite(exp[1]) else "",
            "expected_z": float(exp[2]) if exp is not None and np.isfinite(exp[2]) else "",
            "reason": "expected_position_not_computable",
            "priority": priority,
            "invalidation_reason": entry.get("invalidation_reason") if priority == 1 else "",
            "source_unlabeled": "",
            "distance_to_expected": "",
        }

    env_r = float(cfg["envelope_radius_mm"])
    assign_max = float(cfg["assignment_threshold_mm"])
    c = centroids[row_f]
    consumed = consumed_unlabeled_per_frame.setdefault(row_f, set())

    candidates: list[tuple[float, int, np.ndarray]] = []
    for ui in unlabeled_indices:
        ust = idx_to_stem[int(ui)]
        if ust in consumed:
            continue
        pu = points[row_f, ui, :]
        if not np.isfinite(pu).all():
            continue
        if np.isfinite(c).all():
            if float(np.linalg.norm(pu - c)) > env_r:
                continue
        dist = float(np.linalg.norm(pu - exp))
        if dist < assign_max:
            candidates.append((dist, int(ui), pu.copy()))

    inv_reason = str(entry.get("invalidation_reason", "") if priority == 1 else "")

    if not candidates:
        return {
            "frame": frame_col,
            "marker": stem,
            "stage": 5,
            "action": "no_candidate",
            "tier": tier,
            "confidence": "LOW",
            "original_x": "",
            "original_y": "",
            "original_z": "",
            "new_x": "",
            "new_y": "",
            "new_z": "",
            "expected_x": float(exp[0]),
            "expected_y": float(exp[1]),
            "expected_z": float(exp[2]),
            "reason": "no_unlabeled_within_envelope_and_threshold",
            "priority": priority,
            "invalidation_reason": inv_reason,
            "source_unlabeled": "",
            "distance_to_expected": "",
        }

    candidates.sort(key=lambda x: x[0])
    best_dist, best_ui, _best_pos = candidates[0]
    best_name = str(idx_to_stem.get(best_ui, best_ui))
    donor = points[row_f, best_ui, :].copy()
    points[row_f, mi, :] = donor
    points[row_f, best_ui, :] = np.nan
    consumed.add(best_name)

    if best_dist < 10.0:
        conf = "HIGH"
    elif best_dist < 20.0:
        conf = "MEDIUM"
    else:
        conf = "LOW"
    if len(candidates) == 1 and conf == "MEDIUM":
        conf = "HIGH"

    return {
        "frame": frame_col,
        "marker": stem,
        "stage": 5,
        "action": "unlabeled_replacement",
        "tier": tier,
        "confidence": conf,
        "original_x": float(prev[0]) if np.isfinite(prev[0]) else "",
        "original_y": float(prev[1]) if np.isfinite(prev[1]) else "",
        "original_z": float(prev[2]) if np.isfinite(prev[2]) else "",
        "new_x": float(donor[0]),
        "new_y": float(donor[1]),
        "new_z": float(donor[2]),
        "expected_x": float(exp[0]),
        "expected_y": float(exp[1]),
        "expected_z": float(exp[2]),
        "reason": f"from={best_name}",
        "priority": priority,
        "invalidation_reason": inv_reason,
        "source_unlabeled": best_name,
        "distance_to_expected": float(best_dist),
    }


def apply_unlabeled_assignment_with_priority(
    points: np.ndarray,
    meta: dict[str, Any],
    unlabeled_indices: Sequence[int],
    unlabeled_stems: Sequence[str],
    invalidated_marker_frames: Sequence[Mapping[str, Any]],
    marker_tiers: Mapping[str, int],
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    centroids: np.ndarray,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Stage 5: assign unlabeled points to missing labeled markers.

    Priority 1: markers invalidated in stages 1 or 4 (sorted tier 2→3→1, then earlier stage).
    Priority 2: originally missing (NaN) markers not in the invalidation list (tier order).
    """
    log: list[dict[str, Any]] = []
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    frames: np.ndarray = meta["frames"]
    n_frames = int(meta["n_frames"])
    idx_to_stem = {int(ui): str(ust) for ui, ust in zip(unlabeled_indices, unlabeled_stems, strict=True)}
    frame_to_row = {int(frames[i]): i for i in range(n_frames)}

    invalidated_set = {(int(e["frame"]), str(e["marker"])) for e in invalidated_marker_frames}
    priority_1 = sorted(
        invalidated_marker_frames,
        key=lambda e: (
            _tier_priority_sort_key(int(e["tier"])),
            int(e["invalidation_stage"]),
            int(e["frame"]),
            str(e["marker"]),
        ),
    )

    seg_markers = [
        s
        for s in label_to_idx
        if not _is_unlabeled_stem(s) and not str(s).startswith("OBSTACLE")
    ]
    priority_2: list[dict[str, Any]] = []
    for row_f in range(n_frames):
        fc = int(frames[row_f])
        for stem in seg_markers:
            if stem not in label_to_idx:
                continue
            mi = label_to_idx[stem]
            if np.isfinite(points[row_f, mi, :]).all():
                continue
            if (fc, stem) in invalidated_set:
                continue
            priority_2.append(
                {
                    "frame": fc,
                    "marker": stem,
                    "tier": int(marker_tiers.get(stem, 3)),
                }
            )
    priority_2.sort(
        key=lambda e: (_tier_priority_sort_key(int(e["tier"])), int(e["frame"]), str(e["marker"]))
    )

    consumed_unlabeled_per_frame: dict[int, set[str]] = {}

    for entry in priority_1:
        log.append(
            try_unlabeled_replacement(
                entry,
                1,
                points,
                meta,
                unlabeled_indices,
                unlabeled_stems,
                marker_tiers,
                segment_markers_dict,
                reference_geometry,
                centroids,
                consumed_unlabeled_per_frame,
                frame_to_row,
                idx_to_stem,
                cfg,
            )
        )

    for entry in priority_2:
        row = try_unlabeled_replacement(
            entry,
            2,
            points,
            meta,
            unlabeled_indices,
            unlabeled_stems,
            marker_tiers,
            segment_markers_dict,
            reference_geometry,
            centroids,
            consumed_unlabeled_per_frame,
            frame_to_row,
            idx_to_stem,
            cfg,
        )
        # Avoid huge logs: priority-2 occluded slots usually lack geometry; skip routine skips.
        if str(row.get("action", "")) not in ("no_expected_skip", "no_longer_missing_skip"):
            log.append(row)

    return log


def validate_continuity(
    points: np.ndarray,
    meta: dict[str, Any],
    log_entries: list[dict[str, Any]],
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Downgrade confidence or revert LOW corrections that break per-frame velocity vs neighbors."""
    v_thr = float(cfg["velocity_threshold_per_frame_mm"])
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    n_frames = int(meta["n_frames"])
    frames: np.ndarray = meta["frames"]
    frame_to_row = {int(frames[i]): i for i in range(n_frames)}
    mod_actions = {
        "same_segment_swap",
        "cross_segment_swap",
        "pelvis_combinatorial_swap",
        "single_frame_outlier_rejection",
        "unlabeled_assigned",
        "unlabeled_replacement",
    }
    for row in log_entries:
        act = str(row.get("action", ""))
        if act not in mod_actions:
            continue
        if act.startswith("wrong_label"):
            continue
        marker = str(row["marker"])
        if marker not in label_to_idx:
            continue
        frv = int(row["frame"])
        if frv not in frame_to_row:
            continue
        f = frame_to_row[frv]
        mi = label_to_idx[marker]
        new_pos = np.array(
            [
                float(row["new_x"]) if row["new_x"] != "" else np.nan,
                float(row["new_y"]) if row["new_y"] != "" else np.nan,
                float(row["new_z"]) if row["new_z"] != "" else np.nan,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(new_pos).all():
            continue
        viol = 0
        for delta in (-1, 1):
            fn = f + delta
            if fn < 0 or fn >= n_frames:
                continue
            nb = points[fn, mi, :]
            if not np.isfinite(nb).all():
                continue
            if float(np.linalg.norm(new_pos - nb)) > v_thr:
                viol += 1
        if viol == 0:
            continue
        conf = str(row.get("confidence", "LOW"))
        if conf == "HIGH":
            row["confidence"] = "MEDIUM"
        elif conf == "MEDIUM":
            row["confidence"] = "LOW"
        else:
            ox = row["original_x"]
            oy = row["original_y"]
            oz = row["original_z"]
            rev = np.array(
                [
                    float(ox) if ox != "" else np.nan,
                    float(oy) if oy != "" else np.nan,
                    float(oz) if oz != "" else np.nan,
                ],
                dtype=np.float64,
            )
            if np.isfinite(rev).all():
                points[f, mi, :] = rev
            row["action"] = str(row["action"]) + "_reverted"
    return log_entries


def _sync_data_rows_from_points(meta: dict[str, Any]) -> None:
    stem_to_triplet: dict[str, tuple[int, int, int]] = meta["stem_to_col_triplet"]
    all_stems: list[str] = meta["all_stems"]
    points: np.ndarray = meta["points"]
    rows: list[list[str]] = meta["data_rows"]
    for r in range(int(meta["n_frames"])):
        row = rows[r]
        for si, stem in enumerate(all_stems):
            ix, iy, iz = stem_to_triplet[stem]
            for k, j in enumerate((ix, iy, iz)):
                v = points[r, si, k]
                row[j] = "" if not np.isfinite(v) else str(float(v))


def save_corrected_csv(meta: dict[str, Any], output_csv_path: str | Path) -> None:
    """Write labeled CSV using ``meta`` header and ``data_rows`` (call :func:`_sync_data_rows_from_points` first)."""
    out = Path(output_csv_path)
    with open(out, "w", newline="") as f:
        f.write(meta["original_header_line"])
        w = csv.writer(f)
        for row in meta["data_rows"]:
            w.writerow(row)


def save_correction_log(log_entries: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    df = pd.DataFrame(list(log_entries))
    df.to_csv(output_path, index=False)


def save_quality_metrics(metrics: Mapping[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(metrics, indent=2))


def plot_correction_summary(metrics: Mapping[str, Any], save_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(str(metrics.get("trial_name", "correction_summary")))

    vb = metrics.get("visibility_before", {})
    va = metrics.get("visibility_after", {})
    markers = sorted(set(vb) | set(va))
    x = np.arange(len(markers))
    before = [vb.get(m, 0) for m in markers]
    after = [va.get(m, 0) for m in markers]
    axes[0].bar(x - 0.2, before, 0.4, label="before")
    axes[0].bar(x + 0.2, after, 0.4, label="after")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(markers, rotation=90, fontsize=6)
    axes[0].set_title("Visibility count")
    axes[0].legend()

    cbs = metrics.get("corrections_by_stage", {})
    axes[1].bar(range(len(cbs)), list(cbs.values()))
    axes[1].set_xticks(range(len(cbs)))
    axes[1].set_xticklabels(list(cbs.keys()), rotation=45, ha="right", fontsize=7)
    axes[1].set_title("Corrections by stage")

    m_before = metrics.get("mean_residual_per_segment_before", {})
    m_after = metrics.get("mean_residual_per_segment_after", {})
    segs = sorted(set(m_before) | set(m_after))
    xx = np.arange(len(segs))
    axes[2].bar(xx - 0.2, [m_before.get(s, 0.0) for s in segs], 0.4, label="before")
    axes[2].bar(xx + 0.2, [m_after.get(s, 0.0) for s in segs], 0.4, label="after")
    axes[2].set_xticks(xx)
    axes[2].set_xticklabels(segs, rotation=45, ha="right", fontsize=7)
    axes[2].set_title("Mean residual mm / segment")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def _visibility_counts(
    points: np.ndarray,
    meta: dict[str, Any],
    stems: Sequence[str],
) -> dict[str, int]:
    label_to_idx = meta["label_to_marker_idx"]
    n = int(meta["n_frames"])
    out: dict[str, int] = {}
    for s in stems:
        if s not in label_to_idx:
            continue
        mi = label_to_idx[s]
        c = 0
        for f in range(n):
            if np.isfinite(points[f, mi, :]).all():
                c += 1
        out[s] = c
    return out


def _mean_segment_residual(
    points: np.ndarray,
    meta: dict[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
) -> dict[str, float]:
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    n_frames = int(meta["n_frames"])
    out: dict[str, float] = {}
    for seg_name, ref in reference_geometry.items():
        names_ref: list[str] = [str(x).strip() for x in ref["marker_names"]]
        P_loc = ref["P_local"]
        stem_row = _ref_stem_row_index(ref)
        acc: list[float] = []
        for f in range(n_frames):
            rows_loc: list[np.ndarray] = []
            rows_glob: list[np.ndarray] = []
            for stem in names_ref:
                if stem not in label_to_idx:
                    continue
                mi = label_to_idx[stem]
                g = points[f, mi, :]
                if not np.isfinite(g).all():
                    continue
                rows_loc.append(np.asarray(P_loc[stem_row[stem]], dtype=np.float64))
                rows_glob.append(np.asarray(g, dtype=np.float64))
            if len(rows_loc) < 3:
                continue
            a_loc = np.stack(rows_loc, axis=0)
            a_glob = np.stack(rows_glob, axis=0)
            r, t = kabsch(a_loc, a_glob)
            pred = a_loc @ r.T + t
            err = float(np.mean(np.linalg.norm(a_glob - pred, axis=1)))
            acc.append(err)
        if acc:
            out[seg_name] = float(np.mean(acc))
    return out


def _find_reference_geometry_at_best_row(
    meta_orig: dict[str, Any],
    seg_dict: Mapping[str, list[str]],
    preferred_frame: int,
    collinearity_eps_mm: float,
    search_radius: int,
) -> tuple[int, int, dict[str, Any]]:
    """
    Pick a row in ``meta_orig`` where :func:`compute_reference_geometry` succeeds.

    Tries ``preferred_frame`` first, then rows ordered by increasing ``|frame - preferred|``.
    Returns ``(row_index, frame_column_value, reference_geometry)``.
    """
    frames_arr: np.ndarray = meta_orig["frames"]
    n = int(meta_orig["n_frames"])
    preferred_frame = int(preferred_frame)
    if search_radius <= 0:
        hits = np.flatnonzero(frames_arr == preferred_frame)
        if hits.size == 0:
            raise ValueError(
                f"original_best_frame={preferred_frame} not found in original CSV frame column"
            )
        ri = int(hits[0])
        rg = compute_reference_geometry(
            meta_orig, seg_dict, ri, collinearity_eps_mm=collinearity_eps_mm
        )
        return ri, int(frames_arr[ri]), rg

    order = sorted(range(n), key=lambda ri: abs(int(frames_arr[ri]) - preferred_frame))
    last_err: ValueError | None = None
    for ri in order:
        if abs(int(frames_arr[ri]) - preferred_frame) > int(search_radius):
            continue
        try:
            rg = compute_reference_geometry(
                meta_orig, seg_dict, ri, collinearity_eps_mm=collinearity_eps_mm
            )
            return ri, int(frames_arr[ri]), rg
        except ValueError as e:
            last_err = e
            continue
    msg = (
        f"No suitable reference row within ±{search_radius} frames of "
        f"original_best_frame={preferred_frame} for all segments. "
        "Pick a frame with full marker visibility or increase reference_frame_search_radius in config."
    )
    if last_err is not None:
        msg = f"{msg} Last error: {last_err}"
    raise ValueError(msg)


def _count_frames_complete_pelvis(
    points_after: np.ndarray,
    meta: dict[str, Any],
    expected: tuple[str, ...],
) -> int:
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]
    stems = [m for m in expected if m in label_to_idx]
    if len(stems) != 4:
        return 0
    n = int(meta["n_frames"])
    c = 0
    for f in range(n):
        if all(np.isfinite(points_after[f, label_to_idx[m], :]).all() for m in stems):
            c += 1
    return int(c)


def _log_priority_int(row: Mapping[str, Any]) -> int:
    p = row.get("priority", "")
    if p == "" or p is None:
        return 0
    try:
        return int(p)
    except (TypeError, ValueError):
        return 0


def compute_quality_metrics(
    *,
    meta: dict[str, Any],
    points_before: np.ndarray,
    points_after: np.ndarray,
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference_geometry: Mapping[str, Any],
    marker_tiers: Mapping[str, int],
    log_entries: Sequence[Mapping[str, Any]],
    sustained_warnings: Sequence[str],
    original_csv_path: str,
    trimmed_best_frame_sidecar: int,
    trial_name: str,
    trimmed_only_marker_columns: Sequence[str] | None = None,
    reference_frame_requested: int | None = None,
    reference_frame_used: int | None = None,
    original_best_frame_source: str | None = None,
    full_chain_swap_intervals: Sequence[Mapping[str, Any]] | None = None,
    stage5_5_full_chain_swap_frames_affected: int = 0,
    invalidated_marker_frames: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    seg_stems = [str(x).strip() for names in segment_markers_dict.values() for x in names]
    seg_stems = list(dict.fromkeys(seg_stems))
    vb = _visibility_counts(points_before, meta, seg_stems)
    va = _visibility_counts(points_after, meta, seg_stems)
    by_tier: dict[int, dict[str, Any]] = {1: {}, 2: {}, 3: {}}
    for t in (1, 2, 3):
        ms = [m for m, ti in marker_tiers.items() if ti == t and m in vb]
        if ms:
            b = np.mean([vb[m] for m in ms]) / max(int(meta["n_frames"]), 1)
            a = np.mean([va[m] for m in ms]) / max(int(meta["n_frames"]), 1)
            by_tier[t] = {"before_mean": float(b), "after_mean": float(a), "markers": {m: (vb[m], va[m]) for m in ms}}
    cbs: dict[str, int] = {}
    for row in log_entries:
        act = str(row.get("action", ""))
        if act == "full_chain_swap_interval":
            continue
        cbs[act] = cbs.get(act, 0) + 1
    n_chain_iv = len(list(full_chain_swap_intervals or []))
    cbs["stage5_5_full_chain_swap_intervals"] = int(n_chain_iv)
    cbs["stage5_5_full_chain_swap_frames_affected"] = int(stage5_5_full_chain_swap_frames_affected)

    cbs["stage5_priority1_replacements"] = sum(
        1
        for row in log_entries
        if str(row.get("action", "")) == "unlabeled_replacement" and _log_priority_int(row) == 1
    )
    cbs["stage5_priority2_replacements"] = sum(
        1
        for row in log_entries
        if str(row.get("action", "")) == "unlabeled_replacement" and _log_priority_int(row) == 2
    )
    cbs["stage5_priority1_no_candidate"] = sum(
        1
        for row in log_entries
        if str(row.get("action", "")) == "no_candidate" and _log_priority_int(row) == 1
    )
    cbs["stage5_priority2_no_candidate"] = sum(
        1
        for row in log_entries
        if str(row.get("action", "")) == "no_candidate" and _log_priority_int(row) == 2
    )

    inv_list = [dict(x) for x in (invalidated_marker_frames or [])]
    total_inv = len(inv_list)
    recovered_pri1 = 0
    by_marker_inv: dict[str, int] = {}
    by_marker_rec: dict[str, int] = {}
    by_reason_inv: dict[str, int] = {}
    by_reason_rec: dict[str, int] = {}
    for e in inv_list:
        m = str(e.get("marker", ""))
        rsn = str(e.get("invalidation_reason", ""))
        by_marker_inv[m] = by_marker_inv.get(m, 0) + 1
        by_reason_inv[rsn] = by_reason_inv.get(rsn, 0) + 1
    for e in inv_list:
        fc = int(e["frame"])
        m = str(e["marker"])
        rsn = str(e.get("invalidation_reason", ""))
        if any(
            str(r.get("action", "")) == "unlabeled_replacement"
            and _log_priority_int(r) == 1
            and int(r.get("frame", -1)) == fc
            and str(r.get("marker", "")) == m
            for r in log_entries
        ):
            recovered_pri1 += 1
            by_marker_rec[m] = by_marker_rec.get(m, 0) + 1
            by_reason_rec[rsn] = by_reason_rec.get(rsn, 0) + 1

    def _rate(num: int, den: int) -> float:
        return float(100.0 * num / den) if den else 0.0

    recovery_rate_by_marker = {
        m: _rate(by_marker_rec.get(m, 0), by_marker_inv.get(m, 0))
        for m in sorted(set(by_marker_inv) | set(by_marker_rec))
    }
    recovery_rate_by_reason = {
        r: _rate(by_reason_rec.get(r, 0), by_reason_inv.get(r, 0))
        for r in sorted(set(by_reason_inv) | set(by_reason_rec))
    }
    invalidation_recovery_summary: dict[str, Any] = {
        "total_invalidations": int(total_inv),
        "recovered_via_unlabeled": int(recovered_pri1),
        "recovery_rate": _rate(recovered_pri1, total_inv),
        "recovery_rate_by_marker": recovery_rate_by_marker,
        "recovery_rate_by_invalidation_reason": recovery_rate_by_reason,
    }

    cbt = {1: 0, 2: 0, 3: 0}
    for row in log_entries:
        cbt[int(row.get("tier", 3))] = cbt.get(int(row.get("tier", 3)), 0) + 1
    cbc = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for row in log_entries:
        cbc[str(row.get("confidence", "LOW"))] = cbc.get(str(row.get("confidence", "LOW")), 0) + 1
    m_before = _mean_segment_residual(points_before, meta, segment_markers_dict, reference_geometry)
    m_after = _mean_segment_residual(points_after, meta, segment_markers_dict, reference_geometry)
    pelvis_stems = {"LASI", "RASI", "LPSI", "RPSI"}
    trunk_stems = {"C7", "T10", "CLAV", "STRN"}
    com_crit = {m: va.get(m, 0) / max(int(meta["n_frames"]), 1) for m in pelvis_stems | trunk_stems if m in va}

    return {
        "trial_name": trial_name,
        "trimmed_only_marker_columns": list(trimmed_only_marker_columns or []),
        "n_frames": int(meta["n_frames"]),
        "reference_source": {
            "original_csv": original_csv_path,
            "original_best_frame_requested": reference_frame_requested,
            "original_best_frame_source": original_best_frame_source,
            "reference_frame_used": reference_frame_used,
            "trimmed_best_frame_sidecar": trimmed_best_frame_sidecar,
        },
        "visibility_before": vb,
        "visibility_after": va,
        "visibility_change": {m: va.get(m, 0) - vb.get(m, 0) for m in seg_stems if m in vb or m in va},
        "visibility_by_tier": by_tier,
        "corrections_by_stage": dict(sorted(cbs.items())),
        "corrections_by_tier": cbt,
        "corrections_by_confidence": cbc,
        "mean_residual_per_segment_before": m_before,
        "mean_residual_per_segment_after": m_after,
        "com_critical_marker_visibility_after": com_crit,
        "sustained_drift_warnings": list(sustained_warnings),
        "frames_with_complete_pelvis": _count_frames_complete_pelvis(
            points_after, meta, ("LASI", "RASI", "LPSI", "RPSI")
        ),
        "max_residual_per_segment_after": {
            k: float(v) for k, v in m_after.items()
        },
        "full_chain_swap_intervals": list(full_chain_swap_intervals or []),
        "invalidation_recovery_summary": invalidation_recovery_summary,
    }


def correct_markers(
    trimmed_csv_path: str | Path,
    original_csv_path: str | Path,
    original_best_frame: int | None,
    output_csv_path: str | Path,
    segment_markers_dict: Mapping[str, Sequence[str]],
    *,
    marker_tiers: Mapping[str, int] | None = None,
    same_segment_swap_pairs: Sequence[tuple[str, str]] | None = None,
    cross_segment_swap_pairs: Sequence[tuple[str, str]] | None = None,
    bilateral_chains: Mapping[str, tuple[Sequence[str], Sequence[str]]] | None = None,
    pelvis_segment_name: str = "Pelvis",
    verified_segments: Sequence[str] | None = None,
    obstacle_marker_pair: tuple[str, str] | None = None,
    body_marker_names: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Correct labeling on a trimmed CSV using reference geometry from the original trial.

    ``original_best_frame`` may be ``None``: then the integer in
    ``bestframe_sidecar_path(original_csv_path)`` is used (same convention as
    :func:`marker_label.trial_trim.load_best_frame_1based` / pipeline ``*.csv.bestframe``).

    See module docstring for the staged pipeline (envelope → pelvis → other swaps →
    rejection → unlabeled assignment → continuity).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    seg_dict = {str(k): [str(x).strip() for x in v] for k, v in segment_markers_dict.items()}
    for seg, names in seg_dict.items():
        if len(names) < 3:
            raise ValueError(f"Segment {seg!r} needs ≥3 markers")

    _, meta_trim = parse_labeled_csv(trimmed_csv_path)
    _, meta_orig = parse_labeled_csv(original_csv_path)

    t_stems = set(meta_trim["all_stems"])
    o_stems = set(meta_orig["all_stems"])
    if not o_stems.issubset(t_stems):
        only_orig = sorted(o_stems - t_stems)[:15]
        raise ValueError(
            "Original CSV has marker columns not present in the trimmed CSV. "
            "Re-export or use an original whose column set is a subset of the trimmed file. "
            f"only_in_original_not_in_trimmed={only_orig!r}"
        )
    trimmed_only = sorted(t_stems - o_stems)

    for names in seg_dict.values():
        for raw in names:
            st = str(raw).strip()
            if st not in meta_trim["label_to_marker_idx"]:
                raise ValueError(f"Segment marker {st!r} missing from trimmed CSV")

    if original_best_frame is None:
        orig_bf_path = bestframe_sidecar_path(Path(original_csv_path))
        if not orig_bf_path.is_file():
            raise FileNotFoundError(
                "original_best_frame was not provided and original CSV has no best-frame sidecar: "
                f"{orig_bf_path}"
            )
        obf_requested = int(load_best_frame(orig_bf_path))
        obf_source = "original_csv_bestframe_sidecar"
    else:
        obf_requested = int(original_best_frame)
        obf_source = "argument"

    hits = np.flatnonzero(meta_orig["frames"] == obf_requested)
    if hits.size == 0:
        raise ValueError(
            f"original_best_frame={obf_requested} not found in original CSV frame column"
        )
    search_r = int(cfg.get("reference_frame_search_radius", 150))
    orig_row, ref_frame_used, ref_geom = _find_reference_geometry_at_best_row(
        meta_orig,
        seg_dict,
        obf_requested,
        float(cfg["collinearity_eps_mm"]),
        search_r,
    )
    if ref_frame_used != obf_requested:
        logger.info(
            "Using reference frame %s (requested %s) for geometry; "
            "requested row had missing/invalid markers for at least one segment.",
            ref_frame_used,
            obf_requested,
        )

    bf_path = bestframe_sidecar_path(trimmed_csv_path)
    trimmed_bf = load_best_frame(bf_path)

    points = np.array(meta_trim["points"], copy=True)
    points_before = points.copy()
    meta_work = dict(meta_trim)
    meta_work["points"] = points

    marker_tier_map: dict[str, int] = (
        {str(k): int(v) for k, v in marker_tiers.items()} if marker_tiers is not None else auto_classify_tiers(seg_dict, verified_segments)
    )

    body_union = (
        [str(s).strip() for s in body_marker_names]
        if body_marker_names is not None
        else sorted({str(x).strip() for names in seg_dict.values() for x in names})
    )
    log_all: list[dict[str, Any]] = []
    sustained: list[str] = []
    invalidated_marker_frames: list[dict[str, Any]] = []

    centroids, envelope_outside, env_info = compute_subject_envelope(
        meta_work,
        body_union,
        obstacle_marker_pair,
        envelope_radius_mm=float(cfg["envelope_radius_mm"]),
        envelope_min_visible_markers=int(cfg["envelope_min_visible_markers"]),
        walking_path_y_margin_mm=float(cfg["walking_path_y_margin_mm"]),
    )

    log_all.extend(
        apply_envelope_filter(
            points,
            meta_work,
            marker_tier_map,
            body_union,
            envelope_outside,
            centroids,
            env_info,
            cfg,
            invalidated_marker_frames,
        )
    )

    pelvis_four = _pelvis_four_markers(seg_dict, pelvis_segment_name)
    pelvis_key: str | None = None
    try:
        pelvis_key = _segment_key_ci(seg_dict, pelvis_segment_name)
    except KeyError:
        pelvis_four = None
    if pelvis_four is not None and pelvis_key is not None and pelvis_key in ref_geom:
        log_all.extend(
            apply_pelvis_combinatorial_swap(
                points, meta_work, pelvis_four, ref_geom, pelvis_key, cfg
            )
        )
    elif pelvis_four is not None:
        logger.warning(
            "Pelvis markers listed but segment %r missing from reference_geometry.",
            pelvis_segment_name,
        )

    if bilateral_chains is not None:
        chains_resolved: dict[str, tuple[list[str], list[str]]] = {
            str(k): ([str(x).strip() for x in v[0]], [str(x).strip() for x in v[1]])
            for k, v in bilateral_chains.items()
        }
    else:
        chains_resolved = {k: (list(v[0]), list(v[1])) for k, v in DEFAULT_BILATERAL_CHAINS.items()}

    chain_interval_meta: list[dict[str, Any]] = []
    chain_frames_aff = 0
    if chains_resolved:
        fb_sign = int(ref_frame_used) if int(ref_frame_used) != int(obf_requested) else None
        clogs, chain_interval_meta, _n_iv, chain_frames_aff = apply_full_chain_swap_detection(
            points,
            meta_work,
            meta_orig,
            chains_resolved,
            int(obf_requested),
            fallback_original_frame_for_sign=fb_sign,
            cfg=cfg,
        )
        log_all.extend(clogs)

    s_pairs = tuple(same_segment_swap_pairs) if same_segment_swap_pairs is not None else DEFAULT_SAME_SEGMENT_SWAP_PAIRS
    c_pairs = tuple(cross_segment_swap_pairs) if cross_segment_swap_pairs is not None else DEFAULT_CROSS_SEGMENT_SWAP_PAIRS
    log_all.extend(
        apply_same_segment_swap(points, meta_work, s_pairs, seg_dict, ref_geom, cfg, marker_tier_map)
    )
    log_all.extend(
        apply_cross_segment_swap(
            points, meta_work, c_pairs, seg_dict, ref_geom, marker_tier_map, cfg
        )
    )
    log_all.extend(
        apply_single_frame_rejection(
            points,
            meta_work,
            marker_tier_map,
            seg_dict,
            ref_geom,
            cfg,
            sustained,
            invalidated_marker_frames,
        )
    )

    unlabeled_idx: list[int] = []
    unlabeled_stem: list[str] = []
    for si, stem in enumerate(meta_work["all_stems"]):
        if _is_unlabeled_stem(stem):
            unlabeled_idx.append(si)
            unlabeled_stem.append(stem)
    log_all.extend(
        apply_unlabeled_assignment_with_priority(
            points,
            meta_work,
            unlabeled_idx,
            unlabeled_stem,
            invalidated_marker_frames,
            marker_tier_map,
            seg_dict,
            ref_geom,
            centroids,
            cfg,
        )
    )

    validate_continuity(points, meta_work, log_all, cfg)
    _sync_data_rows_from_points(meta_work)

    out_p = Path(output_csv_path)
    save_corrected_csv(meta_work, out_p)
    shutil.copy2(bf_path, bestframe_sidecar_path(out_p))
    save_correction_log(log_all, out_p.with_suffix(out_p.suffix + ".corrections.csv"))
    qm = compute_quality_metrics(
        meta=meta_work,
        points_before=points_before,
        points_after=points,
        segment_markers_dict=seg_dict,
        reference_geometry=ref_geom,
        marker_tiers=marker_tier_map,
        log_entries=log_all,
        sustained_warnings=sustained,
        original_csv_path=str(Path(original_csv_path).resolve()),
        trimmed_best_frame_sidecar=int(trimmed_bf),
        trial_name=out_p.stem,
        trimmed_only_marker_columns=trimmed_only,
        reference_frame_requested=obf_requested,
        reference_frame_used=int(ref_frame_used),
        original_best_frame_source=obf_source,
        full_chain_swap_intervals=chain_interval_meta,
        stage5_5_full_chain_swap_frames_affected=int(chain_frames_aff),
        invalidated_marker_frames=invalidated_marker_frames,
    )
    save_quality_metrics(qm, out_p.with_suffix(out_p.suffix + ".quality.json"))
    try:
        plot_correction_summary(qm, out_p.with_suffix(out_p.suffix + ".summary.png"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not write summary plot: %s", e)

    corrected_markers = {
        meta_work["all_stems"][mi]: points[:, mi, :].copy()
        for mi in range(len(meta_work["all_stems"]))
        if not _is_unlabeled_stem(meta_work["all_stems"][mi])
    }
    df_log = pd.DataFrame(log_all)
    return {
        "corrected_markers": corrected_markers,
        "quality_metrics": qm,
        "correction_log": df_log,
    }


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Correct marker labels on a trimmed labeled CSV using original-trial reference geometry."
    )
    parser.add_argument("trimmed_csv", type=Path, help="Trimmed labeled CSV (e.g. from marker-label-trim-combine)")
    parser.add_argument("original_csv", type=Path, help="Original untrimmed labeled CSV")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output corrected CSV path")
    parser.add_argument(
        "--original-best-frame",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Preferred frame column value in the original CSV for reference geometry. "
            "If omitted, reads the integer from the original CSV's *.csv.bestframe sidecar "
            "(same convention as marker-label-trial-trim). "
            "If the row has missing markers for a segment, the nearest frame within "
            "--reference-frame-search-radius is used (see quality JSON)."
        ),
    )
    parser.add_argument(
        "--reference-frame-search-radius",
        type=int,
        default=150,
        help=(
            "Search trial frames within ±this distance of the requested reference frame when the "
            "preferred row cannot build all segment references. Use 0 to require the exact frame only."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted({"full-body", "lower-body"}),
        default="full-body",
        help="Segment dictionary preset (markers must exist in CSV)",
    )
    parser.add_argument(
        "--obstacle-pair",
        nargs=2,
        metavar=("L", "R"),
        default=None,
        help="Obstacle markers for walking-path Y filter",
    )
    parser.add_argument(
        "--exclude-same-segment-swap",
        nargs=2,
        metavar=("M1", "M2"),
        action="append",
        default=None,
        help=(
            "Exclude one marker pair from the built-in same-segment swap list (repeatable). "
            "Example: --exclude-same-segment-swap T10 STRN"
        ),
    )
    parser.add_argument(
        "--no-same-segment-swap",
        action="store_true",
        help="Disable all default same-segment pairwise swaps (overrides --exclude-same-segment-swap).",
    )
    args = parser.parse_args()
    seg = segment_markers_dict_for_trim_preset(args.preset)
    pair = tuple(args.obstacle_pair) if args.obstacle_pair else None
    cfg_cli = {"reference_frame_search_radius": int(args.reference_frame_search_radius)}
    if args.no_same_segment_swap:
        s_pairs: tuple[tuple[str, str], ...] | None = ()
    else:
        s_pairs = filter_same_segment_swap_pairs(
            DEFAULT_SAME_SEGMENT_SWAP_PAIRS,
            [tuple(x) for x in (args.exclude_same_segment_swap or [])],
        )
    try:
        res = correct_markers(
            args.trimmed_csv,
            args.original_csv,
            args.original_best_frame,
            args.output,
            seg,
            obstacle_marker_pair=pair,
            pelvis_segment_name="Pelvis",
            same_segment_swap_pairs=s_pairs,
            config=cfg_cli,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    print(json.dumps({"n_corrections": len(res["correction_log"])}, indent=2))


if __name__ == "__main__":
    main()
