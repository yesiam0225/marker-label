"""Segment marker counts and rigid-fill visibility thresholds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def unique_segment_markers(seg_names: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(x).strip() for x in seg_names))


def unique_marker_count(seg_names: Sequence[str]) -> int:
    return len(unique_segment_markers(seg_names))


def min_visible_others_for_segment(
    seg_names: Sequence[str],
    config: Mapping,
) -> int:
    """
    Minimum same-segment markers (excluding target) visible at each gap frame for rigid.

    Default: 3 if segment has 4+ unique markers, else 2 when
    ``allow_two_marker_rigid`` is true (foot/hand triangles).
    """
    n_unique = unique_marker_count(seg_names)
    if n_unique >= 4:
        return 3
    if n_unique == 3 and bool(config.get("allow_two_marker_rigid", True)):
        return 2
    return 3


def segment_uses_two_marker_rigid(
    segment_name: str,
    seg_names: Sequence[str],
    config: Mapping,
) -> bool:
    if not bool(config.get("allow_two_marker_rigid", True)):
        return False
    if unique_marker_count(seg_names) != 3:
        return False
    allowed = config.get("two_marker_rigid_segment_names")
    if allowed is None:
        return True
    return str(segment_name) in {str(x) for x in allowed}
