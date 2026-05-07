"""Rigid-body gap fill via Kabsch between reference local and current global."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from marker_label.trial_trim import kabsch


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
    out: list[dict] = []

    for f in range(start, end + 1):
        fc = int(frame_column[f])
        used: list[str] = []
        p_ref_list: list[np.ndarray] = []
        p_cur_list: list[np.ndarray] = []
        mk = str(marker).strip()
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
        if len(used) < 3:
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
        p_ref = np.stack(p_ref_list, axis=0)
        p_cur = np.stack(p_cur_list, axis=0)
        r, t = kabsch(p_ref, p_cur)
        pred_local = np.asarray(ref_seg[marker], dtype=np.float64)
        pred = pred_local @ r.T + t
        recon = p_ref @ r.T + t
        res = np.linalg.norm(recon - p_cur, axis=1)
        max_res = float(np.max(res))
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
