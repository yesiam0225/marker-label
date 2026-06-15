"""Synthesize missing hand markers from static forearm–wrist geometry (FRM, WRA, WRB, FIN)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from marker_label.gap_filling.reference import (
    _global_to_local,
    _local_coords_from_three,
    _local_to_global,
)
from marker_label.synthesize_fin_from_static import FOREARM_FRAME, HAND_MARKERS

from .foot_heel_synthesis import insert_marker_into_meta, qc_stems_from_meta
from .segment_utils import unique_segment_markers
from .static_reference import _load_static_points_and_label_idx

_HAND_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("L_Hand", "L"),
    ("R_Hand", "R"),
)

# Frame anchor order (p0, p1, p2) for each target marker.
_FRAME_ANCHORS: dict[str, tuple[str, str, str]] = {
    "FIN": ("FRM", "WRA", "WRB"),
    "WRB": ("FRM", "WRA", "FIN"),
    "WRA": ("FRM", "WRB", "FIN"),
    "FRM": ("WRA", "WRB", "FIN"),
}

_INSERT_AFTER: dict[str, tuple[str, ...]] = {
    "LFRM": ("LELB", "LUPA", "LSHO"),
    "LWRA": ("LFRM", "LELB", "LUPA"),
    "LWRB": ("LWRA", "LFRM"),
    "LFIN": ("LWRB", "LWRA"),
    "RFRM": ("RELB", "RUPA", "RSHO"),
    "RWRA": ("RFRM", "RELB", "RUPA"),
    "RWRB": ("RWRA", "RFRM"),
    "RFIN": ("RWRB", "RWRA"),
}


@dataclass(frozen=True)
class HandSideStaticContext:
    models: dict[str, tuple[tuple[str, str, str], np.ndarray]]
    frm: str
    wra: str
    wrb: str
    wrb_plane_sign: float
    frame_y_ref: np.ndarray | None


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return np.asarray(v, dtype=np.float64) / n


def _mean_finite_rows(
    points: np.ndarray,
    label_to_idx: Mapping[str, int],
    names: Sequence[str],
) -> dict[str, np.ndarray] | None:
    acc: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for f in range(points.shape[0]):
        vecs: dict[str, np.ndarray] = {}
        for n in names:
            p = points[f, label_to_idx[n], :]
            if not np.isfinite(p).all():
                break
            vecs[n] = p
        else:
            for n in names:
                acc[n].append(vecs[n])
    if not acc[names[0]]:
        return None
    return {n: np.mean(np.stack(v, axis=0), axis=0) for n, v in acc.items() if v}


def _side_marker_names(side: str) -> tuple[str, str, str, str]:
    wra, wrb, fin = HAND_MARKERS[side]
    frm, _, _ = FOREARM_FRAME[side]
    return frm, wra, wrb, fin


def _sagittal_ap_axis_from_row(
    points_row: np.ndarray,
    label_to_idx: Mapping[str, int],
    lab_vertical: np.ndarray,
) -> np.ndarray | None:
    """Walking A/P from pelvis ML (RASI-LASI) × lab-up; matches body-labeling arm A/P."""
    lasi_i = label_to_idx.get("LASI")
    rasi_i = label_to_idx.get("RASI")
    if lasi_i is None or rasi_i is None:
        return None
    lasi = points_row[lasi_i, :]
    rasi = points_row[rasi_i, :]
    if not (np.isfinite(lasi).all() and np.isfinite(rasi).all()):
        return None
    ml = _unit(rasi - lasi)
    g = _unit(lab_vertical)
    if ml is None or g is None:
        return None
    return _unit(np.cross(ml, g))


def _lock_wrb_sagittal_to_wra(
    pred: np.ndarray,
    wra: np.ndarray,
    expected_sagittal_sign: float,
    ap_axis: np.ndarray,
) -> np.ndarray:
    """Keep WRB on the same sagittal (A/P) side of WRA as the reference sign."""
    if abs(expected_sagittal_sign) < 1e-12:
        return pred
    d = float(np.dot(pred - wra, ap_axis))
    if d * expected_sagittal_sign < 0.0:
        return np.asarray(wra, dtype=np.float64) + (pred - wra) - 2.0 * d * ap_axis
    return pred


def _posterior_axis_in_wrist_plane(
    frm: np.ndarray,
    wra: np.ndarray,
    lab_vertical: np.ndarray,
) -> np.ndarray | None:
    """Unit axis in the wrist plane (perpendicular to forearm), from forearm × lab-up."""
    x_ax = _unit(wra - frm)
    g = _unit(lab_vertical)
    if x_ax is None or g is None:
        return None
    post = np.cross(x_ax, g)
    post = post - float(np.dot(post, x_ax)) * x_ax
    post_u = _unit(post)
    if post_u is not None:
        return post_u
    post = np.cross(g, x_ax)
    post = post - float(np.dot(post, x_ax)) * x_ax
    return _unit(post)


def _wrb_wra_plane_dot(
    frm: np.ndarray,
    wra: np.ndarray,
    wrb: np.ndarray,
    lab_vertical: np.ndarray,
) -> float:
    post = _posterior_axis_in_wrist_plane(frm, wra, lab_vertical)
    if post is None:
        return 0.0
    return float(np.dot(wrb - wra, post))


def _lock_wrb_posterior_to_wra(
    pred: np.ndarray,
    wra: np.ndarray,
    frm: np.ndarray,
    expected_plane_sign: float,
    lab_vertical: np.ndarray,
) -> np.ndarray:
    """Keep WRB on the same side of WRA (posterior vs anterior) as the reference sign."""
    if abs(expected_plane_sign) < 1e-12:
        return pred
    post = _posterior_axis_in_wrist_plane(frm, wra, lab_vertical)
    if post is None:
        return pred
    d = float(np.dot(pred - wra, post))
    if d * expected_plane_sign < 0.0:
        return np.asarray(wra, dtype=np.float64) + (pred - wra) - 2.0 * d * post
    return pred


def _build_side_context(
    static_points: np.ndarray,
    s_idx: Mapping[str, int],
    side: str,
    lab_vertical: np.ndarray,
) -> HandSideStaticContext | None:
    frm, wra, wrb, fin = _side_marker_names(side)
    names = (frm, wra, wrb, fin)
    if not all(n in s_idx for n in names):
        return None
    means = _mean_finite_rows(static_points, s_idx, names)
    if means is None:
        return None

    short = {"FRM": frm, "WRA": wra, "WRB": wrb, "FIN": fin}
    out: dict[str, tuple[tuple[str, str, str], np.ndarray]] = {}
    frame_y_ref: np.ndarray | None = None
    wrb_roles = _FRAME_ANCHORS["WRB"]
    for key, (a, b, c) in _FRAME_ANCHORS.items():
        tgt = short[key]
        p0, p1, p2 = means[short[a]], means[short[b]], means[short[c]]
        try:
            origin, basis = _local_coords_from_three(p0, p1, p2)
            local = _global_to_local(means[tgt], origin, basis)
        except ValueError:
            continue
        out[tgt] = ((short[a], short[b], short[c]), local)
        if key == "WRB":
            frame_y_ref = np.asarray(basis[:, 1], dtype=np.float64).copy()
    if not out:
        return None
    wrb_plane_sign = _wrb_wra_plane_dot(
        means[frm], means[wra], means[wrb], lab_vertical
    )
    return HandSideStaticContext(
        models=out,
        frm=frm,
        wra=wra,
        wrb=wrb,
        wrb_plane_sign=wrb_plane_sign,
        frame_y_ref=frame_y_ref,
    )


def _insert_after_for(marker: str, all_stems: Sequence[str]) -> str | None:
    for cand in _INSERT_AFTER.get(marker, ()):
        if cand in all_stems:
            return cand
    return None


def _predict_row(
    points_row: np.ndarray,
    label_to_idx: Mapping[str, int],
    anchors: tuple[str, str, str],
    local: np.ndarray,
    *,
    frame_y_reference: np.ndarray | None = None,
) -> np.ndarray | None:
    pts = []
    for name in anchors:
        mi = label_to_idx.get(name)
        if mi is None:
            return None
        p = points_row[mi, :]
        if not np.isfinite(p).all():
            return None
        pts.append(p)
    try:
        origin, basis = _local_coords_from_three(pts[0], pts[1], pts[2])
    except ValueError:
        return None
    if frame_y_reference is not None:
        ref = _unit(frame_y_reference)
        if ref is not None:
            y_ax = np.asarray(basis[:, 1], dtype=np.float64)
            x_ax = np.asarray(basis[:, 0], dtype=np.float64)
            if float(np.dot(y_ax, ref)) < 0.0:
                y_ax = -y_ax
                z_ax = np.cross(x_ax, y_ax)
                nz = float(np.linalg.norm(z_ax))
                if nz >= 1e-12:
                    z_ax = z_ax / nz
                    y_ax = np.cross(z_ax, x_ax)
                    y_ax = y_ax / float(np.linalg.norm(y_ax))
                    basis = np.stack([x_ax, y_ax, z_ax], axis=1)
    pred = _local_to_global(local, origin, basis)
    return pred if np.isfinite(pred).all() else None


def _predict_wrb_row(
    points_row: np.ndarray,
    label_to_idx: Mapping[str, int],
    anchors: tuple[str, str, str],
    local: np.ndarray,
    *,
    expected_plane_sign: float,
    lab_vertical: np.ndarray,
    frame_y_reference: np.ndarray | None = None,
) -> np.ndarray | None:
    pred = _predict_row(
        points_row,
        label_to_idx,
        anchors,
        local,
        frame_y_reference=frame_y_reference,
    )
    if pred is None:
        return None
    frm_i = label_to_idx.get(anchors[0])
    wra_i = label_to_idx.get(anchors[1])
    if frm_i is None or wra_i is None:
        return pred
    frm = points_row[frm_i, :]
    wra = points_row[wra_i, :]
    if not (np.isfinite(frm).all() and np.isfinite(wra).all()):
        return pred
    return _lock_wrb_posterior_to_wra(pred, wra, frm, expected_plane_sign, lab_vertical)


def _local_of_marker(
    points_row: np.ndarray,
    label_to_idx: Mapping[str, int],
    marker: str,
    anchors: tuple[str, str, str],
) -> np.ndarray | None:
    mi = label_to_idx.get(marker)
    if mi is None or not np.isfinite(points_row[mi, :]).all():
        return None
    pts = []
    for name in anchors:
        ai = label_to_idx.get(name)
        if ai is None:
            return None
        p = points_row[ai, :]
        if not np.isfinite(p).all():
            return None
        pts.append(p)
    try:
        origin, basis = _local_coords_from_three(pts[0], pts[1], pts[2])
    except ValueError:
        return None
    return _global_to_local(points_row[mi, :], origin, basis)


def _opposite_side(side: str) -> str:
    return "R" if side == "L" else "L"


def _role_names(side: str) -> dict[str, str]:
    frm, wra, wrb, fin = _side_marker_names(side)
    return {"FRM": frm, "WRA": wra, "WRB": wrb, "FIN": fin}


def _anchor_names(side: str, roles: tuple[str, str, str]) -> tuple[str, str, str]:
    names = _role_names(side)
    return names[roles[0]], names[roles[1]], names[roles[2]]


def _mirror_contralateral_local(local: np.ndarray, role_key: str) -> np.ndarray:
    """Mirror mediolateral local component; WRB uses z to preserve anterior/posterior."""
    out = np.asarray(local, dtype=np.float64).copy()
    if role_key == "WRB":
        out[2] *= -1.0
    else:
        out[1] *= -1.0
    return out


def synthesize_hand_from_contralateral(
    meta: dict[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    frames: np.ndarray,
    lab_vertical: Sequence[float],
    *,
    static_csv_path: str | None = None,
    dynamic_label_to_idx: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Fill missing hand markers using the opposite side's rigid hand geometry.

    Runs only for sides whose hand geometry (FRM/WRA/WRB/FIN) is unavailable in
    the static trial. When ``static_csv_path`` is set and that side's static
    reference builds successfully, contralateral synthesis is skipped for that side.

    WRB uses sagittal A/P lock from the source side when pelvis markers exist.
    """
    vert = np.asarray(lab_vertical, dtype=np.float64)
    label_to_idx = meta["label_to_marker_idx"]
    points = meta["points"]
    fills: list[dict[str, Any]] = []

    static_side_ctx: dict[str, HandSideStaticContext | None] = {}
    if static_csv_path and dynamic_label_to_idx is not None:
        static_points, _, s_idx_all, _, _ = _load_static_points_and_label_idx(
            static_csv_path, dynamic_label_to_idx
        )
        for side_key in ("L", "R"):
            static_side_ctx[side_key] = _build_side_context(
                static_points, s_idx_all, side_key, vert
            )

    for seg_name, side in _HAND_SEGMENTS:
        if seg_name not in segment_markers_dict:
            continue
        if static_side_ctx.get(side) is not None:
            continue
        hand_markers = set(unique_segment_markers(segment_markers_dict[seg_name]))
        roles = _role_names(side)
        opp = _opposite_side(side)
        opp_roles = _role_names(opp)
        if not hand_markers.intersection({roles["WRA"], roles["WRB"], roles["FIN"]}):
            continue

        for role_key, anchor_roles in _FRAME_ANCHORS.items():
            tgt = roles[role_key]
            src = opp_roles[role_key]
            tgt_anchors = _anchor_names(side, anchor_roles)
            src_anchors = _anchor_names(opp, anchor_roles)

            if tgt not in label_to_idx:
                after = _insert_after_for(tgt, meta["all_stems"])
                insert_marker_into_meta(meta, tgt, after=after)
                label_to_idx = meta["label_to_marker_idx"]
                points = meta["points"]

            if src not in label_to_idx:
                continue
            mi = label_to_idx[tgt]

            for f in range(points.shape[0]):
                if np.isfinite(points[f, mi, :]).all():
                    continue
                local = _local_of_marker(points[f], label_to_idx, src, src_anchors)
                if local is None:
                    continue
                mirrored = _mirror_contralateral_local(local, role_key)
                if role_key == "WRB":
                    src_wra_i = label_to_idx.get(opp_roles["WRA"])
                    src_wrb_i = label_to_idx.get(opp_roles["WRB"])
                    tgt_wra_i = label_to_idx.get(roles["WRA"])
                    pred = _predict_row(points[f], label_to_idx, tgt_anchors, mirrored)
                    if pred is None:
                        continue
                    ap = _sagittal_ap_axis_from_row(points[f], label_to_idx, vert)
                    if (
                        ap is not None
                        and src_wra_i is not None
                        and src_wrb_i is not None
                        and tgt_wra_i is not None
                        and np.isfinite(points[f, src_wrb_i, :]).all()
                    ):
                        expected_sign = float(
                            np.dot(
                                points[f, src_wrb_i, :] - points[f, src_wra_i, :],
                                ap,
                            )
                        )
                        pred = _lock_wrb_sagittal_to_wra(
                            pred,
                            points[f, tgt_wra_i, :],
                            expected_sign,
                            ap,
                        )
                else:
                    pred = _predict_row(points[f], label_to_idx, tgt_anchors, mirrored)
                if pred is None:
                    continue
                points[f, mi, :] = pred
                fills.append(
                    {
                        "frame": int(frames[f]),
                        "marker": tgt,
                        "method": "contralateral_hand",
                        "success": True,
                        "confidence": "MEDIUM",
                        "predicted_x": float(pred[0]),
                        "predicted_y": float(pred[1]),
                        "predicted_z": float(pred[2]),
                        "fit_residual_mm": 0.0,
                        "source_markers": ";".join(src_anchors) + ";" + src,
                        "gap_length": int(points.shape[0]),
                        "reason": "contralateral_hand_mirror",
                    }
                )
    return fills


def synthesize_missing_hand_markers(
    meta: dict[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    static_csv_path: str,
    frames: np.ndarray,
    dynamic_label_to_idx: Mapping[str, int],
    lab_vertical: Sequence[float],
) -> list[dict[str, Any]]:
    """
    Insert and fill missing FRM/WRA/WRB/FIN using static rigid hand geometry.

    Each target is predicted in the local frame of the other three markers (forearm frame).
    """
    vert = np.asarray(lab_vertical, dtype=np.float64)
    static_points, _, s_idx_all, _, _ = _load_static_points_and_label_idx(
        static_csv_path, dynamic_label_to_idx
    )
    label_to_idx = meta["label_to_marker_idx"]
    points = meta["points"]
    fills: list[dict[str, Any]] = []

    for seg_name, side in _HAND_SEGMENTS:
        if seg_name not in segment_markers_dict:
            continue
        hand_markers = set(unique_segment_markers(segment_markers_dict[seg_name]))
        frm, wra, wrb, fin = _side_marker_names(side)
        if not hand_markers.intersection({wra, wrb, fin}):
            continue
        ctx = _build_side_context(static_points, s_idx_all, side, vert)
        if ctx is None:
            continue

        for marker in (frm, wra, wrb, fin):
            if marker not in ctx.models:
                continue

            if marker not in label_to_idx:
                after = _insert_after_for(marker, meta["all_stems"])
                insert_marker_into_meta(meta, marker, after=after)
                label_to_idx = meta["label_to_marker_idx"]
                points = meta["points"]

            anchors, local = ctx.models[marker]
            mi = label_to_idx[marker]

            for f in range(points.shape[0]):
                if np.isfinite(points[f, mi, :]).all():
                    continue
                if marker == ctx.wrb:
                    pred = _predict_wrb_row(
                        points[f],
                        label_to_idx,
                        anchors,
                        local,
                        expected_plane_sign=ctx.wrb_plane_sign,
                        lab_vertical=vert,
                        frame_y_reference=ctx.frame_y_ref,
                    )
                else:
                    pred = _predict_row(points[f], label_to_idx, anchors, local)
                if pred is None:
                    continue
                points[f, mi, :] = pred
                fills.append(
                    {
                        "frame": int(frames[f]),
                        "marker": marker,
                        "method": "static_hand_forearm",
                        "success": True,
                        "confidence": "HIGH",
                        "predicted_x": float(pred[0]),
                        "predicted_y": float(pred[1]),
                        "predicted_z": float(pred[2]),
                        "fit_residual_mm": 0.0,
                        "source_markers": ";".join(anchors) + ";static",
                        "gap_length": int(points.shape[0]),
                        "reason": "missing_hand_marker",
                    }
                )
    return fills


__all__ = [
    "qc_stems_from_meta",
    "synthesize_hand_from_contralateral",
    "synthesize_missing_hand_markers",
]
