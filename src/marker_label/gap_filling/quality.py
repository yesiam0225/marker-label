"""Quality metrics after gap filling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .gap_detection import find_gaps


def _visibility_counts(points: np.ndarray, stems: Sequence[str], label_to_idx: Mapping[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    n = points.shape[0]
    for m in stems:
        mi = label_to_idx[m]
        c = 0
        for f in range(n):
            if np.isfinite(points[f, mi, :]).all():
                c += 1
        out[m] = c
    return out


def _total_gaps_and_lengths(points: np.ndarray, stems: Sequence[str], label_to_idx: Mapping[str, int]) -> tuple[int, float, int]:
    total = 0
    lengths: list[int] = []
    for m in stems:
        mi = label_to_idx[m]
        for s, e, ln in find_gaps(points[:, mi, :]):
            total += 1
            lengths.append(ln)
    if not lengths:
        return 0, 0.0, 0
    return total, float(np.mean(lengths)), int(max(lengths))


def compute_quality_metrics(
    points_before: np.ndarray,
    points_after: np.ndarray,
    stems: Sequence[str],
    label_to_idx: Mapping[str, int],
    fills_log: Sequence[Mapping],
    segment_markers_dict: Mapping[str, Sequence[str]],
    *,
    continuity_warnings: int = 0,
    reverted_fills: int = 0,
) -> dict:
    vb = _visibility_counts(points_before, stems, label_to_idx)
    va = _visibility_counts(points_after, stems, label_to_idx)
    vc = {m: int(va.get(m, 0) - vb.get(m, 0)) for m in stems}

    fills_by_method = {"rigid_body": 0, "asis_only": 0, "spline": 0, "unfillable": 0}
    fills_by_marker: dict[str, int] = {m: 0 for m in stems}
    fills_by_conf = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for row in fills_log:
        meth = str(row.get("method", ""))
        if meth in fills_by_method:
            fills_by_method[meth] += 1
        mk = str(row.get("marker", ""))
        if mk in fills_by_marker:
            fills_by_marker[mk] += 1
        if row.get("success"):
            conf = str(row.get("confidence", ""))
            if conf in fills_by_conf:
                fills_by_conf[conf] += 1

    tg_b, mean_b, max_b = _total_gaps_and_lengths(points_before, stems, label_to_idx)
    tg_a, _, _ = _total_gaps_and_lengths(points_after, stems, label_to_idx)

    stacked: dict[str, dict[str, int]] = {
        "rigid_body": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "asis_only": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "spline": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
    }
    for row in fills_log:
        if not row.get("success"):
            continue
        meth = str(row.get("method", ""))
        conf = str(row.get("confidence", ""))
        if meth in stacked and conf in stacked[meth]:
            stacked[meth][conf] += 1

    unf_frames: set[int] = set()
    unf_by_marker: dict[str, int] = {m: 0 for m in stems}
    for row in fills_log:
        if str(row.get("method")) != "unfillable":
            continue
        fc = int(row["frame"])
        unf_frames.add(fc)
        mk = str(row["marker"])
        if mk in unf_by_marker:
            unf_by_marker[mk] += 1

    seg_incomplete: dict[str, int] = {}
    seg_complete_frames: dict[str, int] = {}
    n_frames = points_after.shape[0]
    for seg, names in segment_markers_dict.items():
        ns = [str(x).strip() for x in names if str(x).strip() in label_to_idx]
        inc = 0
        complete = 0
        for f in range(n_frames):
            ok = all(np.isfinite(points_after[f, label_to_idx[nm], :]).all() for nm in ns)
            if ok:
                complete += 1
            else:
                inc += 1
        seg_incomplete[str(seg)] = inc
        seg_complete_frames[str(seg)] = complete

    return {
        "visibility_before": vb,
        "visibility_after": va,
        "visibility_change": vc,
        "fills_by_method": fills_by_method,
        "fills_by_marker": fills_by_marker,
        "fills_by_confidence": fills_by_conf,
        "gap_statistics": {
            "total_gaps_before": tg_b,
            "total_gaps_after": tg_a,
            "mean_gap_length_before": mean_b,
            "max_gap_length_before": max_b,
        },
        "fills_by_method_and_confidence": stacked,
        "unfillable_summary": {
            "frames_with_unfillable_gaps": len(unf_frames),
            "markers_with_unfillable_gaps": {k: v for k, v in unf_by_marker.items() if v > 0},
            "segments_incomplete_after_fill": seg_incomplete,
        },
        "segments_complete_per_frame_after_fill": seg_complete_frames,
        "continuity_warnings": int(continuity_warnings),
        "reverted_fills": int(reverted_fills),
    }
