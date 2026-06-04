"""Rigid fill for 3-marker segments when only two peers are visible (uses static/local offset)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from marker_label.synthesize_rhee_from_static import foot_basis_from_ankle_toe

from .segment_utils import unique_segment_markers


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


def body_offset_from_static_means(
    means: Mapping[str, np.ndarray],
    origin_marker: str,
    x_marker: str,
    target_marker: str,
    lab_vertical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(offset in ankle-toe basis, flip_reference vector)``."""
    o = np.asarray(means[origin_marker], dtype=np.float64)
    x = np.asarray(means[x_marker], dtype=np.float64)
    t = np.asarray(means[target_marker], dtype=np.float64)
    x_ax, y_ax, z_ax = foot_basis_from_ankle_toe(o, x, lab_vertical, z_flip_reference=None)
    r = np.stack([x_ax, y_ax, z_ax], axis=1)
    flip_ref = t - o
    return r.T @ (t - o), flip_ref


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
) -> np.ndarray:
    x_ax, y_ax, z_ax = foot_basis_from_ankle_toe(
        origin_pt, x_pt, lab_vertical, z_flip_reference=flip_reference
    )
    r = np.stack([x_ax, y_ax, z_ax], axis=1)
    return np.asarray(origin_pt, dtype=np.float64) + r @ offset


def static_offsets_for_three_marker_segment(
    static_points: np.ndarray,
    label_to_idx: Mapping[str, int],
    seg_names: Sequence[str],
    lab_vertical: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, str, str]]:
    """
    For each marker in a 3-marker segment, offsets when using the other two as anchors.

    Returns ``{target: (offset, flip_ref, origin_marker, x_marker)}``.
    """
    names = unique_segment_markers(seg_names)
    if len(names) != 3:
        return {}
    means = _mean_finite_positions(static_points, label_to_idx, names)
    if means is None or len(means) != 3:
        return {}
    out: dict[str, tuple[np.ndarray, np.ndarray, str, str]] = {}
    for tgt in names:
        others = [n for n in names if n != tgt]
        o_m, x_m = others[0], others[1]
        off, flip = body_offset_from_static_means(means, o_m, x_m, tgt, lab_vertical)
        out[tgt] = (off, flip, o_m, x_m)
    return out
