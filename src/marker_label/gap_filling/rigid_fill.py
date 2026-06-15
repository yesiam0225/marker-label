"""Rigid-body gap fill via Kabsch between reference local and current global."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from marker_label.trial_trim import kabsch

from .segment_utils import min_visible_others_for_segment, segment_uses_two_marker_rigid
from .two_marker_static import pick_two_anchor_markers, predict_from_two_anchors, unpack_two_marker_pack, _foot_tibia_for_marker


def _confidence_from_residual(max_res: float, cfg: Mapping) -> str:
    hi = float(cfg.get("rigid_fill_high_residual_mm", 5))
    med = float(cfg.get("rigid_fill_medium_residual_mm", 15))
    if max_res <= hi:
        return "HIGH"
    if max_res <= med:
        return "MEDIUM"
    return "LOW"


def rigid_body_fill(
    points: np.ndarray,
    marker: str,
    gap: tuple[int, int, int],
    segment: str,
    segment_markers_dict: Mapping[str, Sequence[str]],
    reference: Mapping[str, Mapping[str, np.ndarray]],
    label_to_idx: Mapping[str, int],
    config: Mapping,
    frame_column: np.ndarray,
    *,
    two_marker_offsets: Mapping[str, Mapping[str, tuple]] | None = None,
    lab_vertical: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> list[dict]:
    """
    Frame-wise rigid fill. Returns fill log rows (does not mutate ``points``).
    """
    start, end, length = gap
    seg_names = list(dict.fromkeys(str(x).strip() for x in segment_markers_dict[segment]))
    if segment not in reference:
        return [
            _row(
                int(frame_column[f]),
                marker,
                "rigid_body",
                False,
                "",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "",
                length,
                "no_reference_segment",
            )
            for f in range(start, end + 1)
        ]

    ref_seg = reference[segment]
    max_res_thr = float(config.get("rigid_fill_max_residual_mm", 30))
    min_others = min_visible_others_for_segment(seg_names, config)
    use_two = segment_uses_two_marker_rigid(segment, seg_names, config)
    vert = np.array(lab_vertical, dtype=np.float64)
    seg_two = (two_marker_offsets or {}).get(segment, {}) if two_marker_offsets else {}
    mk = str(marker).strip()
    two_pack = seg_two.get(mk) if use_two else None
    out: list[dict] = []

    for f in range(start, end + 1):
        fc = int(frame_column[f])
        used: list[str] = []
        p_ref_list: list[np.ndarray] = []
        p_cur_list: list[np.ndarray] = []
        for m in seg_names:
            if str(m).strip() == mk:
                continue
            mi = label_to_idx[m]
            if not np.isfinite(points[f, mi, :]).all():
                continue
            if m not in ref_seg:
                continue
            used.append(m)
            p_ref_list.append(np.asarray(ref_seg[m], dtype=np.float64))
            p_cur_list.append(np.asarray(points[f, mi, :], dtype=np.float64))
        if len(used) < min_others:
            out.append(
                _row(
                    fc,
                    marker,
                    "rigid_body",
                    False,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    ";".join(used),
                    length,
                    "insufficient_visible_markers",
                )
            )
            continue

        if len(used) >= 3:
            p_ref = np.stack(p_ref_list, axis=0)
            p_cur = np.stack(p_cur_list, axis=0)
            r, t = kabsch(p_ref, p_cur)
            pred_local = np.asarray(ref_seg[mk], dtype=np.float64)
            pred = pred_local @ r.T + t
            recon = p_ref @ r.T + t
            res = np.linalg.norm(recon - p_cur, axis=1)
            max_res = float(np.max(res))
        elif len(used) == 2 and two_pack is not None:
            off, flip, shin_ref, o_m, x_m, kind = unpack_two_marker_pack(two_pack)
            pair = pick_two_anchor_markers(seg_names, mk, used)
            if pair is None:
                max_res = np.nan
                pred = np.full(3, np.nan)
            else:
                a_m, b_m = pair
                mi_a, mi_b = label_to_idx[a_m], label_to_idx[b_m]
                ank_pt, toe_pt = _foot_ankle_toe_pts(points, f, label_to_idx, mk)
                pred = predict_from_two_anchors(
                    points[f, mi_a, :],
                    points[f, mi_b, :],
                    off,
                    vert,
                    flip,
                    tibia_pt=_tibia_pt_for_foot_marker(points, f, label_to_idx, mk),
                    foot_ankle_pt=ank_pt,
                    foot_toe_pt=toe_pt,
                    shin_axis_reference=shin_ref,
                    segment_kind=kind,
                )
                max_res = 0.0 if np.isfinite(pred).all() else np.nan
            used = list(used)
            used.append("two_marker_static")
        else:
            out.append(
                _row(
                    fc,
                    marker,
                    "rigid_body",
                    False,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    ";".join(used),
                    length,
                    "need_three_markers_or_static_two_marker",
                )
            )
            continue
        if max_res > max_res_thr:
            out.append(
                _row(
                    fc,
                    marker,
                    "rigid_body",
                    False,
                    "",
                    float(pred[0]),
                    float(pred[1]),
                    float(pred[2]),
                    max_res,
                    ";".join(used),
                    length,
                    "residual_above_max",
                )
            )
            continue
        conf = _confidence_from_residual(max_res, config)
        out.append(
            _row(
                fc,
                marker,
                "rigid_body",
                True,
                conf,
                float(pred[0]),
                float(pred[1]),
                float(pred[2]),
                max_res,
                ";".join(used),
                length,
                "",
            )
        )
    return out


def _tibia_pt_for_foot_marker(
    points: np.ndarray,
    frame: int,
    label_to_idx: Mapping[str, int],
    marker: str,
) -> np.ndarray | None:
    tib_name = _foot_tibia_for_marker(marker)
    if not tib_name or tib_name not in label_to_idx:
        return None
    pt = points[frame, label_to_idx[tib_name], :]
    if not np.isfinite(pt).all():
        return None
    return np.asarray(pt, dtype=np.float64)


def _foot_ankle_toe_pts(
    points: np.ndarray,
    frame: int,
    label_to_idx: Mapping[str, int],
    marker: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    m = str(marker).strip()
    if m.startswith("L"):
        ank_name, toe_name = "LANK", "LTOE"
    elif m.startswith("R"):
        ank_name, toe_name = "RANK", "RTOE"
    else:
        return None, None
    if ank_name not in label_to_idx or toe_name not in label_to_idx:
        return None, None
    ank = points[frame, label_to_idx[ank_name], :]
    toe = points[frame, label_to_idx[toe_name], :]
    if not (np.isfinite(ank).all() and np.isfinite(toe).all()):
        return None, None
    return np.asarray(ank, dtype=np.float64), np.asarray(toe, dtype=np.float64)


def _row(
    frame: int,
    marker: str,
    method: str,
    success: bool,
    confidence: str,
    px: float,
    py: float,
    pz: float,
    fit_residual_mm: float,
    source_markers: str,
    gap_length: int,
    reason: str,
) -> dict:
    return {
        "frame": frame,
        "marker": marker,
        "method": method,
        "success": success,
        "confidence": confidence,
        "predicted_x": px,
        "predicted_y": py,
        "predicted_z": pz,
        "fit_residual_mm": fit_residual_mm,
        "source_markers": source_markers,
        "gap_length": gap_length,
        "reason": reason,
    }
