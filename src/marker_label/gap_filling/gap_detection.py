"""Detect NaN runs in marker trajectories and categorize fill strategy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .segment_utils import min_visible_others_for_segment


def _marker_missing_row(points_row: np.ndarray) -> bool:
    return not np.isfinite(points_row).all()


def find_gaps(trajectory: np.ndarray) -> list[tuple[int, int, int]]:
    """
    ``trajectory``: (n_frames, 3). Missing if any coordinate is non-finite.

    Returns list of ``(start, end, length)`` inclusive ``start``/``end`` indices.
    """
    t = np.asarray(trajectory, dtype=np.float64)
    n = t.shape[0]
    miss = np.array([_marker_missing_row(t[f]) for f in range(n)], dtype=bool)
    out: list[tuple[int, int, int]] = []
    i = 0
    while i < n:
        if not miss[i]:
            i += 1
            continue
        j = i
        while j < n and miss[j]:
            j += 1
        out.append((i, j - 1, j - i))
        i = j
    return out


def _segment_for_marker(
    marker: str, segment_markers_dict: Mapping[str, Sequence[str]]
) -> str | None:
    m = str(marker).strip()
    for seg, names in segment_markers_dict.items():
        if m in [str(x).strip() for x in names]:
            return str(seg)
    return None


def _count_visible_segment_others(
    points: np.ndarray,
    label_to_idx: Mapping[str, int],
    seg_names: Sequence[str],
    target: str,
    frame: int,
) -> int:
    c = 0
    for m in seg_names:
        if str(m).strip() == target:
            continue
        mi = label_to_idx[str(m).strip()]
        if np.isfinite(points[frame, mi, :]).all():
            c += 1
    return c


def categorize_gap(
    marker: str,
    gap: tuple[int, int, int],
    points: np.ndarray,
    label_to_idx: Mapping[str, int],
    segment_markers_dict: Mapping[str, Sequence[str]],
    bilateral_pelvis_markers: tuple[str, ...],
    asis_markers: tuple[str, ...],
    config: Mapping,
) -> tuple[str, str | None]:
    """
    Returns ``('rigid_body', segment)`` | ``('asis_only', None)`` | ``('spline', None)``
    | ``('unfillable', None)``.
    """
    start, end, length = gap
    n_frames = points.shape[0]
    marker = str(marker).strip()
    seg = _segment_for_marker(marker, segment_markers_dict)
    long_thr = int(config.get("long_gap_threshold", 50))
    attempt_long = bool(config.get("attempt_long_gaps", True))

    if seg is not None and (attempt_long or length <= long_thr):
        seg_names = list(dict.fromkeys(str(x).strip() for x in segment_markers_dict[seg]))
        min_others = min_visible_others_for_segment(seg_names, config)
        if all(
            _count_visible_segment_others(points, label_to_idx, seg_names, marker, f) >= min_others
            for f in range(start, end + 1)
        ):
            return "rigid_body", seg

    bilateral_set = {str(x).strip() for x in bilateral_pelvis_markers}
    if marker in bilateral_set and bool(config.get("asis_only_enabled", True)):
        ok_asis = True
        for f in range(start, end + 1):
            for a in asis_markers:
                if not np.isfinite(points[f, label_to_idx[str(a).strip()], :]).all():
                    ok_asis = False
                    break
            if not ok_asis:
                break
        if ok_asis:
            return "asis_only", None

    spline_max = int(config.get("spline_max_length", 10))
    if length <= spline_max and start > 0 and end < n_frames - 1:
        left_ok = np.isfinite(points[start - 1, label_to_idx[marker], :]).all()
        right_ok = np.isfinite(points[end + 1, label_to_idx[marker], :]).all()
        if left_ok and right_ok:
            return "spline", None

    return "unfillable", None
