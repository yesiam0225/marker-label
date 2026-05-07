"""ASIS-only pelvis fallback for PSI markers (LPSI / RPSI)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / n


def ap_axis_at_anchor(
    points: np.ndarray,
    row: int,
    label_to_idx: Mapping[str, int],
    lasi: str,
    rasi: str,
    lpsi: str,
    rpsi: str,
) -> np.ndarray:
    i_l, i_r = label_to_idx[lasi], label_to_idx[rasi]
    i_lp, i_rp = label_to_idx[lpsi], label_to_idx[rpsi]
    lasi_p = points[row, i_l, :]
    rasi_p = points[row, i_r, :]
    lpsi_p = points[row, i_lp, :]
    rpsi_p = points[row, i_rp, :]
    center = 0.5 * (lasi_p + rasi_p)
    ml = _unit(rasi_p - lasi_p)
    psi_mid = 0.5 * (lpsi_p + rpsi_p)
    ap_dir = psi_mid - center
    ap_dir = ap_dir - float(np.dot(ap_dir, ml)) * ml
    return _unit(ap_dir)


def interpolate_ap_axis(
    points: np.ndarray,
    anchor_rows: Sequence[int],
    target_row: int,
    label_to_idx: Mapping[str, int],
    frame_column: np.ndarray,
    lasi: str,
    rasi: str,
    lpsi: str,
    rpsi: str,
) -> np.ndarray:
    """Linearly interpolate AP unit vector in frame-number space between anchors."""
    if not anchor_rows:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    rows = sorted(int(r) for r in anchor_rows)
    pairs: list[tuple[float, np.ndarray]] = []
    for r in rows:
        fn = float(frame_column[r])
        v = ap_axis_at_anchor(points, r, label_to_idx, lasi, rasi, lpsi, rpsi)
        pairs.append((fn, v))
    pairs.sort(key=lambda x: x[0])
    t0 = float(frame_column[target_row])
    if t0 <= pairs[0][0]:
        return _unit(pairs[0][1])
    if t0 >= pairs[-1][0]:
        return _unit(pairs[-1][1])
    for i in range(len(pairs) - 1):
        t1, v1 = pairs[i]
        t2, v2 = pairs[i + 1]
        if t1 <= t0 <= t2:
            if abs(t2 - t1) < 1e-9:
                return _unit(v1)
            w = (t0 - t1) / (t2 - t1)
            return _unit((1.0 - w) * v1 + w * v2)
    return _unit(pairs[-1][1])


def _pelvis_segment_key(
    segment_markers_dict: Mapping[str, Sequence[str]],
    lasi: str,
    rasi: str,
    lpsi: str,
    rpsi: str,
) -> str | None:
    need = {lasi, rasi, lpsi, rpsi}
    for seg, names in segment_markers_dict.items():
        ns = {str(x).strip() for x in names}
        if need <= ns:
            return str(seg)
    return None


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
        "method": "asis_only",
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


def asis_only_fill(
    points: np.ndarray,
    target_marker: str,
    gap: tuple[int, int, int],
    asis_markers: tuple[str, str],
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference: Mapping[str, Mapping[str, np.ndarray]],
    label_to_idx: Mapping[str, int],
    config: Mapping,
    frame_column: np.ndarray,
    *,
    lpsi_name: str = "LPSI",
    rpsi_name: str = "RPSI",
) -> list[dict]:
    _ = reference
    start, end, length = gap
    lasi, rasi = str(asis_markers[0]).strip(), str(asis_markers[1]).strip()
    pelvis_seg = _pelvis_segment_key(segment_markers_dict, lasi, rasi, lpsi_name, rpsi_name)
    out: list[dict] = []

    if pelvis_seg is None:
        for f in range(start, end + 1):
            out.append(
                _entry(
                    int(frame_column[f]),
                    target_marker,
                    False,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    "0",
                    length,
                    "no_pelvis_segment_in_dict",
                )
            )
        return out

    # Body-fixed offset from full-pelvis anchors (mean of R^T @ (p_target - center) per anchor).
    # This matches rigid pelvis motion; segment-local reference offsets are not used here.
    radius = int(config.get("asis_only_search_radius", 20))
    min_anchors = int(config.get("asis_only_min_anchor_frames", 2))

    n_all = int(points.shape[0])

    def _full_pelvis_rows(lo: int, hi: int) -> list[int]:
        acc: list[int] = []
        for r in range(lo, hi + 1):
            ok = True
            for m in (lasi, rasi, lpsi_name, rpsi_name):
                if not np.isfinite(points[r, label_to_idx[m], :]).all():
                    ok = False
                    break
            if ok:
                acc.append(r)
        return acc

    def collect_anchors() -> list[int]:
        lo = max(0, start - radius)
        hi = min(n_all - 1, end + radius)
        acc = _full_pelvis_rows(lo, hi)
        if len(acc) < min_anchors:
            acc = _full_pelvis_rows(0, n_all - 1)
        return acc

    anchor_rows = collect_anchors()
    if len(anchor_rows) < min_anchors:
        for f in range(start, end + 1):
            out.append(
                _entry(
                    int(frame_column[f]),
                    target_marker,
                    False,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    str(len(anchor_rows)),
                    length,
                    "insufficient_full_pelvis_anchors",
                )
            )
        return out

    mean_body_offset = np.zeros(3, dtype=np.float64)
    for r in anchor_rows:
        lasi_p = points[r, label_to_idx[lasi], :]
        rasi_p = points[r, label_to_idx[rasi], :]
        center_a = 0.5 * (lasi_p + rasi_p)
        ml_a = _unit(rasi_p - lasi_p)
        ap_a = ap_axis_at_anchor(points, r, label_to_idx, lasi, rasi, lpsi_name, rpsi_name)
        si_a = _unit(np.cross(ap_a, ml_a))
        ap_o_a = _unit(np.cross(ml_a, si_a))
        r_a = np.stack([ml_a, si_a, ap_o_a], axis=1)
        pt_a = points[r, label_to_idx[target_marker], :]
        mean_body_offset += r_a.T @ (pt_a - center_a)
    mean_body_offset /= float(len(anchor_rows))

    for f in range(start, end + 1):
        fc = int(frame_column[f])
        if not np.isfinite(points[f, label_to_idx[lasi], :]).all() or not np.isfinite(
            points[f, label_to_idx[rasi], :]
        ).all():
            out.append(
                _entry(
                    fc,
                    target_marker,
                    False,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    str(len(anchor_rows)),
                    length,
                    "asis_not_visible",
                )
            )
            continue
        lasi_p = points[f, label_to_idx[lasi], :]
        rasi_p = points[f, label_to_idx[rasi], :]
        pelvis_center = 0.5 * (lasi_p + rasi_p)
        ml = _unit(rasi_p - lasi_p)
        ap = interpolate_ap_axis(
            points,
            anchor_rows,
            f,
            label_to_idx,
            frame_column,
            lasi,
            rasi,
            lpsi_name,
            rpsi_name,
        )
        si = _unit(np.cross(ap, ml))
        ap_orth = _unit(np.cross(ml, si))
        r_mat = np.stack([ml, si, ap_orth], axis=1)
        pred = pelvis_center + r_mat @ mean_body_offset
        out.append(
            _entry(
                fc,
                target_marker,
                True,
                "LOW",
                float(pred[0]),
                float(pred[1]),
                float(pred[2]),
                str(len(anchor_rows)),
                length,
                "",
            )
        )
    return out
