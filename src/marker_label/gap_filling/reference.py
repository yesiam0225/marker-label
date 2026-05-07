"""Robust per-segment local reference from multiple clean frames."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from marker_label.trial_trim import bestframe_sidecar_path, parse_labeled_csv

logger = logging.getLogger(__name__)


def _frame_row_for_column_value(frames: np.ndarray, frame_col: int) -> int | None:
    hits = np.flatnonzero(frames == int(frame_col))
    if hits.size == 0:
        return None
    return int(hits[0])


def _local_coords_from_three(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Origin at p0; x along p1-p0; z = x ×(p2-p0); y = z×x. Returns (origin, basis 3×3 columns x,y,z)."""
    origin = np.asarray(p0, dtype=np.float64)
    x_ax = np.asarray(p1, dtype=np.float64) - origin
    n = float(np.linalg.norm(x_ax))
    if n < 1e-9:
        raise ValueError("Degenerate segment axis (first two markers coincident)")
    x_ax = x_ax / n
    v2 = np.asarray(p2, dtype=np.float64) - origin
    z_ax = np.cross(x_ax, v2)
    nz = float(np.linalg.norm(z_ax))
    if nz < 1e-9:
        raise ValueError("Degenerate segment plane (third marker collinear)")
    z_ax = z_ax / nz
    y_ax = np.cross(z_ax, x_ax)
    b = np.stack([x_ax, y_ax, z_ax], axis=1)
    return origin, b


def _global_to_local(p: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return basis.T @ (np.asarray(p, dtype=np.float64) - origin)


def _local_to_global(local: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return origin + basis @ np.asarray(local, dtype=np.float64)


def build_robust_reference(
    points: np.ndarray,
    label_to_idx: Mapping[str, int],
    segment_markers_dict: Mapping[str, Sequence[str]],
    *,
    n_samples: int = 30,
    best_frame: int | None = None,
    csv_path: str | Path | None = None,
    frames: np.ndarray | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """
    For each segment, average marker positions in a segment-local frame over up to ``n_samples``
    fully-visible frames. Falls back to a single frame (``best_frame`` column value, or sidecar).

    Returns
    -------
    reference
        ``{segment_name: {marker_name: (3,) local mean}}``
    warnings
        Human-readable warnings (e.g. insufficient clean frames).
    """
    n_frames = int(points.shape[0])
    if frames is None:
        frame_ids = np.arange(n_frames, dtype=np.int64)
    else:
        frame_ids = np.asarray(frames, dtype=np.int64).reshape(-1)
    warnings: list[str] = []

    bf_row: int | None = None
    if best_frame is not None:
        bf_row = _frame_row_for_column_value(frame_ids, int(best_frame))
    elif csv_path is not None:
        p = Path(csv_path)
        bf_path = bestframe_sidecar_path(p)
        if bf_path.is_file():
            try:
                fc = int(bf_path.read_text().strip().split()[0])
                bf_row = _frame_row_for_column_value(frame_ids, fc)
            except (ValueError, OSError):
                pass

    out: dict[str, dict[str, np.ndarray]] = {}
    seg_dict = {str(k): [str(x).strip() for x in v] for k, v in segment_markers_dict.items()}

    for seg_name, names in seg_dict.items():
        names = list(dict.fromkeys(m for m in names if m in label_to_idx))
        if len(names) < 3:
            warnings.append(f"Segment {seg_name!r}: fewer than 3 markers in CSV; skipped.")
            continue
        m0, m1, m2 = names[0], names[1], names[2]
        clean_rows: list[int] = []
        for f in range(n_frames):
            ok = True
            for m in names:
                idx = label_to_idx[m]
                if not np.isfinite(points[f, idx, :]).all():
                    ok = False
                    break
            if ok:
                clean_rows.append(f)
        if len(clean_rows) < 5:
            fb = bf_row if bf_row is not None and bf_row in clean_rows else (clean_rows[0] if clean_rows else None)
            if fb is None:
                warnings.append(f"Segment {seg_name!r}: no fully-visible frame; skipped.")
                continue
            warnings.append(
                f"Segment {seg_name!r}: only {len(clean_rows)} clean frames (<5); using single frame row {fb}."
            )
            sample_rows = [fb]
        else:
            idxs = np.linspace(0, len(clean_rows) - 1, num=min(int(n_samples), len(clean_rows)))
            sample_rows = [clean_rows[int(round(float(i)))] for i in idxs]

        locals_acc: dict[str, list[np.ndarray]] = {m: [] for m in names}
        for f in sample_rows:
            i0, i1, i2 = label_to_idx[m0], label_to_idx[m1], label_to_idx[m2]
            p0, p1, p2 = points[f, i0, :], points[f, i1, :], points[f, i2, :]
            try:
                origin, basis = _local_coords_from_three(p0, p1, p2)
            except ValueError:
                continue
            for m in names:
                idx = label_to_idx[m]
                loc = _global_to_local(points[f, idx, :], origin, basis)
                locals_acc[m].append(loc)
        seg_local: dict[str, np.ndarray] = {}
        for m in names:
            if not locals_acc[m]:
                warnings.append(f"Segment {seg_name!r}: no valid samples for {m}; skipped segment.")
                seg_local = {}
                break
            seg_local[m] = np.mean(np.stack(locals_acc[m], axis=0), axis=0)
        if seg_local:
            out[seg_name] = seg_local
    for w in warnings:
        logger.warning("%s", w)
    return out, warnings


def load_points_and_meta(csv_path: str | Path) -> tuple[np.ndarray, dict]:
    """Parse labeled CSV; returns ``points`` (n_frames, n_markers, 3) and metadata dict."""
    _, meta = parse_labeled_csv(csv_path)
    return np.asarray(meta["points"], dtype=np.float64), meta
