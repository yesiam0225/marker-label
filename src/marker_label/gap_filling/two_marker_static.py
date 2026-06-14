"""Rigid fill for 3-marker segments when only two peers are visible (uses static/local offset)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from marker_label.synthesize_rhee_from_static import (
    ankle_tibia_name,
    foot_basis_from_ankle_toe,
    medial_reference_ankle_toe_tibia,
    shin_axis_ankle_toe_tibia,
)

from .segment_utils import unique_segment_markers

_FOOT_MARKERS = frozenset(
    {"LANK", "LTOE", "LHEE", "RANK", "RTOE", "RHEE"}
)


def _foot_tibia_for_marker(marker: str) -> str | None:
    m = str(marker).strip()
    if m not in _FOOT_MARKERS:
        return None
    return ankle_tibia_name(m[:1] + "ANK")


def _mean_finite_positions(
    points: np.ndarray,
    label_to_idx: Mapping[str, int],
    names: Sequence[str],
) -> dict[str, np.ndarray] | None:
    acc: dict[str, list[np.ndarray]] = {str(n).strip(): [] for n in names}
    for f in range(points.shape[0]):
        for n in names:
            nn = str(n).strip()
            if nn not in label_to_idx:
                continue
            p = points[f, label_to_idx[nn], :]
            if np.isfinite(p).all():
                acc[nn].append(p)
    out: dict[str, np.ndarray] = {}
    for nn, lst in acc.items():
        if not lst:
            return None
        out[nn] = np.mean(np.stack(lst, axis=0), axis=0)
    return out


def _foot_ankle_toe_in_means(means: Mapping[str, np.ndarray]) -> tuple[str, str] | None:
    for ank, toe in (("LANK", "LTOE"), ("RANK", "RTOE")):
        if ank in means and toe in means:
            return ank, toe
    return None


def unpack_two_marker_pack(
    pack: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str, str]:
    """``(offset, flip_ref, shin_ref?, origin, x)``; legacy 4-tuples omit shin_ref."""
    if len(pack) == 4:
        off, flip, o_m, x_m = pack
        return (
            np.asarray(off, dtype=np.float64),
            np.asarray(flip, dtype=np.float64),
            None,
            str(o_m),
            str(x_m),
        )
    off, flip, shin_ref, o_m, x_m = pack
    shin = None
    if shin_ref is not None:
        shin = np.asarray(shin_ref, dtype=np.float64)
    return (
        np.asarray(off, dtype=np.float64),
        np.asarray(flip, dtype=np.float64),
        shin,
        str(o_m),
        str(x_m),
    )


def body_offset_from_static_means(
    means: Mapping[str, np.ndarray],
    origin_marker: str,
    x_marker: str,
    target_marker: str,
    lab_vertical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(offset in ankle-toe basis, flip_reference, shin_axis_ref)``."""
    foot_ank_toe = _foot_ankle_toe_in_means(means)
    if foot_ank_toe is not None:
        ank_name, toe_name = foot_ank_toe
        o = np.asarray(means[ank_name], dtype=np.float64)
        toe_pt = np.asarray(means[toe_name], dtype=np.float64)
        t = np.asarray(means[target_marker], dtype=np.float64)
    else:
        o = np.asarray(means[origin_marker], dtype=np.float64)
        toe_pt = np.asarray(means[x_marker], dtype=np.float64)
        t = np.asarray(means[target_marker], dtype=np.float64)
    tib_name = _foot_tibia_for_marker(origin_marker) or _foot_tibia_for_marker(x_marker)
    if foot_ank_toe is not None:
        tib_name = ankle_tibia_name(foot_ank_toe[0])
    tib = means.get(tib_name) if tib_name and tib_name in means else None
    basis_kw: dict[str, object] = {"z_flip_reference": None}
    shin_ref: np.ndarray | None = None
    if tib is not None:
        tib_v = np.asarray(tib, dtype=np.float64)
        shin_ref = shin_axis_ankle_toe_tibia(o, toe_pt, tib_v)
        med_ref = medial_reference_ankle_toe_tibia(o, toe_pt, tib_v)
        if med_ref is not None:
            basis_kw["tibia"] = tib_v
            basis_kw["z_flip_reference"] = med_ref
            if shin_ref is not None:
                basis_kw["shin_axis_reference"] = shin_ref
    x_ax, y_ax, z_ax = foot_basis_from_ankle_toe(o, toe_pt, lab_vertical, **basis_kw)
    r = np.stack([x_ax, y_ax, z_ax], axis=1)
    flip_ref = basis_kw["z_flip_reference"] if basis_kw["z_flip_reference"] is not None else t - o
    return r.T @ (t - o), np.asarray(flip_ref, dtype=np.float64), shin_ref


def pick_two_anchor_markers(
    seg_names: Sequence[str],
    target: str,
    visible: Sequence[str],
) -> tuple[str, str] | None:
    """Prefer segment list order: first two visible markers that are not ``target``."""
    tgt = str(target).strip()
    vis = {str(v).strip() for v in visible}
    picked: list[str] = []
    for m in seg_names:
        ms = str(m).strip()
        if ms == tgt or ms not in vis:
            continue
        picked.append(ms)
        if len(picked) == 2:
            return picked[0], picked[1]
    return None


def predict_from_two_anchors(
    origin_pt: np.ndarray,
    x_pt: np.ndarray,
    offset: np.ndarray,
    lab_vertical: np.ndarray,
    flip_reference: np.ndarray,
    *,
    tibia_pt: np.ndarray | None = None,
    foot_ankle_pt: np.ndarray | None = None,
    foot_toe_pt: np.ndarray | None = None,
    shin_axis_reference: np.ndarray | None = None,
) -> np.ndarray:
    basis_o = (
        np.asarray(foot_ankle_pt, dtype=np.float64)
        if foot_ankle_pt is not None and np.isfinite(foot_ankle_pt).all()
        else origin_pt
    )
    basis_x = (
        np.asarray(foot_toe_pt, dtype=np.float64)
        if foot_toe_pt is not None and np.isfinite(foot_toe_pt).all()
        else x_pt
    )
    basis_kw: dict[str, object] = {"z_flip_reference": flip_reference}
    if shin_axis_reference is not None:
        basis_kw["shin_axis_reference"] = shin_axis_reference
    if tibia_pt is not None and np.isfinite(tibia_pt).all():
        basis_kw["tibia"] = tibia_pt
    x_ax, y_ax, z_ax = foot_basis_from_ankle_toe(
        basis_o, basis_x, lab_vertical, **basis_kw
    )
    r = np.stack([x_ax, y_ax, z_ax], axis=1)
    return np.asarray(basis_o, dtype=np.float64) + r @ offset


def _mean_foot_positions_with_tibia(
    points: np.ndarray,
    label_to_idx: Mapping[str, int],
    foot_names: Sequence[str],
    tib_name: str,
) -> dict[str, np.ndarray] | None:
    """Mean foot marker positions on frames where all foot markers and tibia are finite."""
    names = [str(n).strip() for n in foot_names]
    tib = str(tib_name).strip()
    if tib not in label_to_idx:
        return None
    acc: dict[str, list[np.ndarray]] = {n: [] for n in names}
    acc[tib] = []
    for f in range(points.shape[0]):
        vecs: dict[str, np.ndarray] = {}
        for n in names + [tib]:
            p = points[f, label_to_idx[n], :]
            if not np.isfinite(p).all():
                break
            vecs[n] = p
        else:
            for n in names + [tib]:
                acc[n].append(vecs[n])
    if not acc[names[0]]:
        return None
    return {n: np.mean(np.stack(lst, axis=0), axis=0) for n, lst in acc.items() if lst}


def static_offsets_for_three_marker_segment(
    static_points: np.ndarray,
    label_to_idx: Mapping[str, int],
    seg_names: Sequence[str],
    lab_vertical: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, str, str]]:
    """
    For each marker in a 3-marker segment, offsets when using the other two as anchors.

    Returns ``{target: (offset, flip_ref, shin_ref, origin_marker, x_marker)}``.
    """
    names = unique_segment_markers(seg_names)
    if len(names) != 3:
        return {}
    means = _mean_finite_positions(static_points, label_to_idx, names)
    if means is None or len(means) != 3:
        return {}
    tib_names = {_foot_tibia_for_marker(n) for n in names}
    tib_names.discard(None)
    for tib in tib_names:
        if not tib or tib in means:
            continue
        co_means = _mean_foot_positions_with_tibia(static_points, label_to_idx, names, tib)
        if co_means is not None:
            means.update(co_means)
        elif tib in label_to_idx:
            extra = _mean_finite_positions(static_points, label_to_idx, [tib])
            if extra:
                means.update(extra)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, str, str]] = {}
    for tgt in names:
        others = [n for n in names if n != tgt]
        o_m, x_m = others[0], others[1]
        off, flip, shin_ref = body_offset_from_static_means(means, o_m, x_m, tgt, lab_vertical)
        out[tgt] = (off, flip, shin_ref, o_m, x_m)
    return out
