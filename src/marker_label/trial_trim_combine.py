"""
Trial trimming and two-pass combining for labeled motion-capture CSVs.

Single-pass: trusted frame range around one best frame (Kabsch residuals + visibility + envelope).
Two-pass: trim each labeling pass, pick a switch frame, merge rows so frames before switch use pass 1
and frames from switch onward use pass 2.

**Trim vs combine:** ``segment_markers_dict`` (CLI ``--preset`` / JSON) lists markers that must exist in the CSV.
Trusted-range trimming, overlap switch scoring, and envelope (unless ``body_marker_names`` is set) use a separate
**trim-QC** segment set: config ``trim_qc_preset`` (default ``"legs-feet"``) or ``trim_qc_preset: None`` to use the
same dict as ``segment_markers_dict``. Combining (row merge) does not depend on ``segment_markers_dict`` geometry.

Sidecar convention matches :mod:`marker_label.trial_trim`: ``<csv>.csv.bestframe`` with the same integer
as the ``frame`` column (pipeline / viewer convention).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .trial_trim import (
    DEFAULT_COLLINEARITY_EPS_MM,
    DEFAULT_CONSECUTIVE_BAD_FRAMES,
    DEFAULT_MIN_TRIAL_LENGTH,
    DEFAULT_RESIDUAL_THRESHOLD_MM,
    DEFAULT_RESIDUAL_THRESHOLD_THIGH_MM,
    DEFAULT_THIGH_SEGMENT_KEYWORDS,
    DEFAULT_VISIBLE_RATIO_THRESHOLD,
    TRIM_SEGMENT_PRESET_CHOICES,
    bestframe_sidecar_path,
    build_visibility_union_indices,
    compute_reference_geometry,
    find_trim_boundary,
    kabsch,
    parse_labeled_csv,
    resolve_visibility_ratio_marker_indices,
    save_bestframe_sidecar,
    segment_markers_dict_for_trim_preset,
    _segment_residual_threshold_mm,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "residual_threshold_mm": DEFAULT_RESIDUAL_THRESHOLD_MM,
    "residual_threshold_thigh_mm": DEFAULT_RESIDUAL_THRESHOLD_THIGH_MM,
    "thigh_segment_keywords": list(DEFAULT_THIGH_SEGMENT_KEYWORDS),
    "visible_ratio_threshold": DEFAULT_VISIBLE_RATIO_THRESHOLD,
    "consecutive_bad_frames": DEFAULT_CONSECUTIVE_BAD_FRAMES,
    "min_trial_length": DEFAULT_MIN_TRIAL_LENGTH,
    "envelope_radius_mm": 1500.0,
    "envelope_min_visible_markers": 3,
    "envelope_outlier_threshold": 0,
    "walking_path_y_margin_mm": 500.0,
    "gap_warning_threshold": 50,
    "collinearity_eps_mm": DEFAULT_COLLINEARITY_EPS_MM,
    "exclude_obstacles_not_in_segments": True,
    "visibility_ratio_marker_subset": "legs-feet",
    # Preset name for trim-only QC (legs-feet = thigh–shank–foot chains). None = use segment_markers_dict.
    "trim_qc_preset": "legs-feet",
}


def _resolve_trim_qc_segment_dict(
    cfg: Mapping[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    raw = cfg.get("trim_qc_preset", "legs-feet")
    if raw is None:
        return {str(k): list(v) for k, v in segment_markers_dict.items()}
    return segment_markers_dict_for_trim_preset(str(raw))


def _markers_union(segment_markers_dict: Mapping[str, Sequence[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for names in segment_markers_dict.values():
        for raw in names:
            st = str(raw).strip()
            if st and st not in seen:
                seen.add(st)
                out.append(st)
    return out


def _validate_two_pass_frames_time(meta1: dict[str, Any], meta2: dict[str, Any]) -> None:
    if not np.array_equal(meta1["frames"], meta2["frames"]):
        raise ValueError("Two-pass mode requires identical `frame` columns in both CSVs")
    t1, t2 = meta1["times"], meta2["times"]
    if t1.shape != t2.shape or not np.allclose(t1, t2, rtol=0.0, atol=0.0, equal_nan=True):
        raise ValueError("Two-pass mode requires identical `time` columns in both CSVs")


def _canonical_stems_pass1_then_pass2_only(stems1: list[str], stems2: list[str]) -> list[str]:
    """Pass-1 column order, then any stems present only in pass 2 (append)."""
    seen = set(stems1)
    out = list(stems1)
    for s in stems2:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _remap_metadata_to_canonical(meta: dict[str, Any], canonical_stems: list[str]) -> dict[str, Any]:
    """
    Rebuild metadata so ``all_stems`` / ``points`` / ``data_rows`` / triple indices match ``canonical_stems``.

    Missing markers in the source CSV become all-NaN trajectories and empty CSV cells.
    """
    old_l2i: dict[str, int] = meta["label_to_marker_idx"]
    points_old: np.ndarray = meta["points"]
    n_frames = int(meta["n_frames"])
    n_can = len(canonical_stems)
    points_new = np.full((n_frames, n_can, 3), np.nan, dtype=np.float64)
    for j, stem in enumerate(canonical_stems):
        if stem in old_l2i:
            oi = old_l2i[stem]
            points_new[:, j, :] = points_old[:, oi, :]

    header_cells = ["frame", "time"]
    for stem in canonical_stems:
        header_cells.extend([f"{stem}_x", f"{stem}_y", f"{stem}_z"])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header_cells)
    original_header_line = buf.getvalue()

    stem_to_triplet: dict[str, tuple[int, int, int]] = {}
    for j, stem in enumerate(canonical_stems):
        base = 2 + j * 3
        stem_to_triplet[stem] = (base, base + 1, base + 2)

    data_rows: list[list[str]] = []
    for r in range(n_frames):
        old_row = meta["data_rows"][r]
        fr = old_row[0].strip() if len(old_row) > 0 else str(int(meta["frames"][r]))
        tm = old_row[1].strip() if len(old_row) > 1 else ""
        row = [fr, tm]
        for j in range(n_can):
            for k in range(3):
                v = points_new[r, j, k]
                row.append("" if not np.isfinite(v) else str(float(v)))
        data_rows.append(row)

    label_to_marker_idx = {stem: j for j, stem in enumerate(canonical_stems)}
    ignored_star = [s for s in canonical_stems if str(s).strip().startswith("*")]

    out_meta = dict(meta)
    out_meta["all_stems"] = list(canonical_stems)
    out_meta["n_markers"] = n_can
    out_meta["points"] = points_new
    out_meta["stem_to_col_triplet"] = stem_to_triplet
    out_meta["label_to_marker_idx"] = label_to_marker_idx
    out_meta["original_header_line"] = original_header_line
    out_meta["data_rows"] = data_rows
    out_meta["ignored_star_stems"] = ignored_star
    return out_meta


def compute_subject_envelope(
    metadata: dict[str, Any],
    body_marker_names: Sequence[str],
    obstacle_marker_pair: tuple[str, str] | None,
    *,
    envelope_radius_mm: float,
    envelope_min_visible_markers: int,
    walking_path_y_margin_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Per-frame centroid of visible body markers and per-(frame, marker) envelope violation flags.

    Returns
    -------
    centroids : (n_frames, 3), NaN when insufficient visible markers
    envelope_outside : (n_frames, n_body) bool — True if marker visible and violates sphere or Y-band
    info : obstacle Y min/max used (if pair set)
    """
    points: np.ndarray = metadata["points"]
    label_to_idx: dict[str, int] = metadata["label_to_marker_idx"]
    n_frames = int(metadata["n_frames"])

    body_stems = [str(s).strip() for s in body_marker_names]
    body_idx = [label_to_idx[s] for s in body_stems if s in label_to_idx]
    if len(body_idx) < int(envelope_min_visible_markers):
        raise ValueError(
            f"Not enough body markers present in CSV for envelope: need at least "
            f"{envelope_min_visible_markers}, got {len(body_idx)}"
        )

    y_lo = y_hi = None
    if obstacle_marker_pair is not None:
        a, b = str(obstacle_marker_pair[0]).strip(), str(obstacle_marker_pair[1]).strip()
        if a not in label_to_idx or b not in label_to_idx:
            raise ValueError(f"Obstacle markers missing from CSV: {a!r}, {b!r}")
        ia, ib = label_to_idx[a], label_to_idx[b]
        ys: list[float] = []
        for f in range(n_frames):
            for mi in (ia, ib):
                p = points[f, mi, :]
                if np.isfinite(p).all():
                    ys.append(float(p[1]))
        if ys:
            y_lo = float(np.min(ys)) - float(walking_path_y_margin_mm)
            y_hi = float(np.max(ys)) + float(walking_path_y_margin_mm)

    centroids = np.full((n_frames, 3), np.nan, dtype=np.float64)
    n_body = len(body_stems)
    envelope_outside = np.zeros((n_frames, n_body), dtype=bool)

    for f in range(n_frames):
        pts = points[f, body_idx, :]
        valid = np.isfinite(pts).all(axis=1)
        nv = int(np.sum(valid))
        if nv < int(envelope_min_visible_markers):
            continue
        c = np.mean(pts[valid], axis=0)
        centroids[f] = c
        for j, stem in enumerate(body_stems):
            if stem not in label_to_idx:
                continue
            mi = label_to_idx[stem]
            p = points[f, mi, :]
            if not np.isfinite(p).all():
                continue
            if np.linalg.norm(p - c) > float(envelope_radius_mm):
                envelope_outside[f, j] = True
            if y_lo is not None and y_hi is not None:
                if p[1] < y_lo or p[1] > y_hi:
                    envelope_outside[f, j] = True

    info: dict[str, Any] = {}
    if y_lo is not None:
        info["obstacle_y_min_mm"] = y_lo + walking_path_y_margin_mm
        info["obstacle_y_max_mm"] = y_hi - walking_path_y_margin_mm
        info["walking_y_band_mm"] = (y_lo, y_hi)
    return centroids, envelope_outside, info


def _stem_envelope_violation(
    stem: str,
    f: int,
    body_stems: Sequence[str],
    envelope_outside: np.ndarray,
) -> bool:
    if stem not in body_stems:
        return False
    j = list(body_stems).index(stem)
    return bool(envelope_outside[f, j])


def compute_diagnostics_with_envelope(
    metadata: dict[str, Any],
    reference_geometry: Mapping[str, Any],
    segment_markers_dict: Mapping[str, Sequence[str]],
    visibility_ratio_indices: Sequence[int],
    body_stems: Sequence[str],
    envelope_outside: np.ndarray,
    *,
    residual_threshold_mm: float,
    residual_threshold_thigh_mm: float,
    thigh_segment_keywords: Sequence[str],
    visible_ratio_threshold: float,
    envelope_outlier_threshold: int,
) -> list[dict[str, Any]]:
    """Like :func:`marker_label.trial_trim.compute_diagnostics` plus envelope-outside count."""
    points: np.ndarray = metadata["points"]
    n_frames = int(metadata["n_frames"])
    label_to_idx: dict[str, int] = metadata["label_to_marker_idx"]
    frames: np.ndarray = metadata["frames"]

    seg_thresholds = {
        sn: _segment_residual_threshold_mm(
            sn,
            default_mm=residual_threshold_mm,
            thigh_mm=residual_threshold_thigh_mm,
            thigh_keywords=thigh_segment_keywords,
        )
        for sn in segment_markers_dict
    }

    n_vis = len(visibility_ratio_indices)
    if n_vis == 0:
        raise ValueError("visibility_ratio_indices is empty")

    diagnostics: list[dict[str, Any]] = []
    for f in range(n_frames):
        vis_pts = points[f, visibility_ratio_indices, :]
        valid = np.isfinite(vis_pts).all(axis=1)
        visible_count = int(np.sum(valid))
        visible_ratio = float(visible_count / max(n_vis, 1))

        env_count = int(np.sum(envelope_outside[f]))
        bad_envelope = env_count > int(envelope_outlier_threshold)

        seg_residual: dict[str, float] = {}
        seg_bad: dict[str, bool] = {}
        for seg_name, ref in reference_geometry.items():
            names_ref: list[str] = ref["marker_names"]
            P_loc = ref["P_local"]
            rows_loc: list[np.ndarray] = []
            rows_glob: list[np.ndarray] = []
            for stem, pl in zip(names_ref, P_loc, strict=True):
                if stem not in label_to_idx:
                    continue
                if _stem_envelope_violation(stem, f, body_stems, envelope_outside):
                    continue
                mi = label_to_idx[stem]
                g = points[f, mi, :]
                if not np.isfinite(g).all():
                    continue
                rows_loc.append(np.asarray(pl, dtype=np.float64))
                rows_glob.append(np.asarray(g, dtype=np.float64))
            if len(rows_loc) < 3:
                max_err = np.inf
            else:
                a_loc = np.stack(rows_loc, axis=0)
                a_glob = np.stack(rows_glob, axis=0)
                r, t = kabsch(a_loc, a_glob)
                pred = a_loc @ r.T + t
                err = np.linalg.norm(a_glob - pred, axis=1)
                max_err = float(np.max(err))
            seg_residual[seg_name] = max_err
            thr = seg_thresholds[seg_name]
            seg_bad[seg_name] = max_err > thr

        bad_residual = any(seg_bad.values())
        bad_vis = visible_ratio < float(visible_ratio_threshold)
        is_bad = bad_residual or bad_vis or bad_envelope

        max_res = max(seg_residual.values()) if seg_residual else np.inf

        diagnostics.append(
            {
                "frame": int(frames[f]),
                "frame_row": f,
                "segment_residual_mm": seg_residual,
                "segment_bad_residual": seg_bad,
                "max_residual_mm": float(max_res) if np.isfinite(max_res) else float("inf"),
                "visible_count": visible_count,
                "visible_ratio": visible_ratio,
                "envelope_outside_count": env_count,
                "bad_residual": bad_residual,
                "bad_visibility": bad_vis,
                "bad_envelope": bad_envelope,
                "is_bad": is_bad,
            }
        )
    return diagnostics


def _trusted_range_frames(
    frames: np.ndarray,
    trim_start_row: int,
    trim_end_row: int,
) -> tuple[int, int]:
    """Half-open frame-number span matching row slice ``[trim_start_row, trim_end_row)``."""
    t0 = int(frames[trim_start_row])
    t1 = int(frames[trim_end_row - 1]) + 1
    return t0, t1


def _best_row_idx(frames: np.ndarray, best_frame: int) -> int:
    f0 = int(frames[0])
    idx = int(best_frame - f0)
    if idx < 0 or idx >= len(frames) or int(frames[idx]) != int(best_frame):
        raise ValueError(
            f"best_frame {best_frame} not found in frame column (contiguous layout required)"
        )
    return idx


def _find_switch_overlap(
    diagnostics1: Sequence[Mapping[str, Any]],
    diagnostics2: Sequence[Mapping[str, Any]],
    overlap_start: int,
    overlap_end_excl: int,
    frames: np.ndarray,
    bf1: int,
    bf2: int,
) -> int:
    f0 = int(frames[0])
    best_score = np.inf
    best_f: int | None = None
    mid = 0.5 * (bf1 + bf2)
    for f in range(overlap_start, overlap_end_excl):
        ri = f - f0
        if ri < 0 or ri >= len(diagnostics1):
            continue
        s1 = float(diagnostics1[ri]["max_residual_mm"])
        s2 = float(diagnostics2[ri]["max_residual_mm"])
        if not np.isfinite(s1):
            s1 = 1e30
        if not np.isfinite(s2):
            s2 = 1e30
        score = s1 + s2
        dist = abs(f - mid)
        if score < best_score or (np.isclose(score, best_score) and best_f is not None and dist < abs(best_f - mid)):
            best_score = score
            best_f = f
    if best_f is None:
        raise RuntimeError("Could not pick switch frame inside overlap")
    return int(best_f)


def _find_switch_gap(bf1: int, bf2: int, gap_start: int, gap_end: int) -> int:
    """gap_start..gap_end inclusive."""
    if gap_end < gap_start:
        raise ValueError(
            f"Empty gap between trusted ranges (gap_start={gap_start}, gap_end={gap_end}); "
            "cannot place switch frame"
        )
    cand = int((bf1 + bf2) // 2)
    return int(max(gap_start, min(gap_end, cand)))


def _write_combined_csv(
    meta_primary: dict[str, Any],
    meta_secondary: dict[str, Any],
    trim_start_row: int,
    trim_end_row: int,
    switch_frame: int | None,
    two_pass: bool,
    output_csv_path: str | Path,
) -> None:
    """Rows ``[trim_start_row, trim_end_row)``; marker triplets from pass1 or pass2 by switch_frame."""
    out = Path(output_csv_path)
    rows1 = meta_primary["data_rows"]
    rows2 = meta_secondary["data_rows"] if two_pass else rows1
    frames_arr: np.ndarray = meta_primary["frames"]
    header_line = meta_primary["original_header_line"]
    stem_to_triplet: dict[str, tuple[int, int, int]] = meta_primary["stem_to_col_triplet"]
    all_stems: list[str] = meta_primary["all_stems"]

    with open(out, "w", newline="") as f:
        f.write(header_line)
        w = csv.writer(f)
        for row_i in range(trim_start_row, trim_end_row):
            r1 = list(rows1[row_i])
            r2 = list(rows2[row_i])
            fv = int(frames_arr[row_i])
            use_p2 = two_pass and switch_frame is not None and fv >= int(switch_frame)
            src = r2 if use_p2 else r1
            out_row = list(r1)
            for stem in all_stems:
                ix, iy, iz = stem_to_triplet[stem]
                out_row[ix] = src[ix]
                out_row[iy] = src[iy]
                out_row[iz] = src[iz]
            w.writerow(out_row)


def _save_trim_log(
    path: str | Path,
    frames_slice: np.ndarray,
    times_slice: np.ndarray,
    t1: tuple[int, int],
    t2: tuple[int, int] | None,
    switch_frame: int | None,
    two_pass: bool,
) -> None:
    p = Path(path)
    t1s, t1e = t1
    t2s, t2e = t2 if t2 is not None else (None, None)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame",
                "time",
                "in_trusted_pass1",
                "in_trusted_pass2",
                "pass_used",
                "is_switch_frame",
            ]
        )
        for i in range(len(frames_slice)):
            fr = int(frames_slice[i])
            tm = float(times_slice[i]) if np.isfinite(times_slice[i]) else ""
            in1 = t1s <= fr < t1e
            in2 = (t2s is not None) and (t2s <= fr < t2e) if two_pass else False
            if not two_pass:
                pu = 1
            else:
                pu = 2 if (switch_frame is not None and fr >= int(switch_frame)) else 1
            is_sw = two_pass and switch_frame is not None and fr == int(switch_frame)
            w.writerow([fr, tm, int(in1), int(in2), pu, int(is_sw)])


def plot_trusted_ranges(
    trusted_ranges: dict[str, tuple[int, int]],
    best_frames: Sequence[int],
    switch_frame: int | None,
    trim_start: int,
    trim_end: int,
    n_frames_orig: int,
    save_path: str | Path | None,
) -> Any:
    """Matplotlib summary: trusted bars, best frames, switch, trim bounds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plot_trusted_ranges requires matplotlib") from e

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax in axes:
        ax.axvline(trim_start, color="r", linestyle="--", linewidth=1.0, label="trim_start")
        ax.axvline(trim_end - 1, color="r", linestyle="--", linewidth=1.0, label="trim_end-1")
        if switch_frame is not None:
            ax.axvline(switch_frame, color="g", linewidth=1.2, label="switch_frame")
        ax.set_xlim(0, max(1, n_frames_orig - 1))

    ax0 = axes[0]
    ax0.set_title("Trusted ranges (half-open in frame space)")
    y = 0.7
    for i, (key, (ts, te)) in enumerate(sorted(trusted_ranges.items())):
        ax0.barh(y, te - ts, left=ts, height=0.15, label=key)
        bf = best_frames[i] if i < len(best_frames) else best_frames[0]
        ax0.scatter([bf], [y + 0.075], s=40, zorder=5, marker="o")
        y -= 0.25
    ax0.set_yticks([])
    ax0.legend(loc="upper right", fontsize=7)

    axes[1].text(0.02, 0.5, "switch / trim markers (see x-axis)", transform=axes[1].transAxes)
    axes[1].set_yticks([])
    axes[2].text(0.02, 0.5, "Use trim boundaries (red) for output span", transform=axes[2].transAxes)
    axes[2].set_xlabel("frame (original numbering)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=150)
    return fig


def trim_and_combine(
    csv_paths: Sequence[str | Path],
    output_csv_path: str | Path,
    segment_markers_dict: Mapping[str, Sequence[str]],
    *,
    bestframe_paths: Sequence[str | Path] | None = None,
    body_marker_names: Sequence[str] | None = None,
    obstacle_marker_pair: tuple[str, str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Trim (single pass) or trim + merge two passes around a switch frame.

    Parameters
    ----------
    csv_paths
        One or two labeled CSV paths.
    bestframe_paths
        Optional sidecar per CSV; default ``<csv>.csv.bestframe`` via :func:`bestframe_sidecar_path`.
    body_marker_names
        Markers for centroid / envelope; default: deduped union of the **trim-QC** segment dict
        (see ``trim_qc_preset`` in ``config``).
    obstacle_marker_pair
        If set, Y-band filter uses these markers' trial Y extent ± margin.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    paths = [Path(p) for p in csv_paths]
    if len(paths) not in (1, 2):
        raise ValueError("csv_paths must have length 1 or 2")

    if bestframe_paths is None:
        bf_paths = [bestframe_sidecar_path(p) for p in paths]
    else:
        bf_paths = [Path(p) for p in bestframe_paths]
        if len(bf_paths) != len(paths):
            raise ValueError("bestframe_paths must match csv_paths length when provided")

    for seg, names in segment_markers_dict.items():
        if len(names) < 3:
            raise ValueError(f"Segment {seg!r} needs ≥3 markers")

    metas: list[dict[str, Any]] = []
    best_frames: list[int] = []
    for csv_p, bf_p in zip(paths, bf_paths, strict=True):
        _, meta = parse_labeled_csv(csv_p)
        if not bf_p.is_file():
            raise FileNotFoundError(f"Best-frame sidecar not found: {bf_p}")
        bf = int(bf_p.read_text().strip().split()[0])
        best_frames.append(bf)
        metas.append(meta)

    two_pass = len(paths) == 2
    if two_pass:
        union_stems = set(metas[0]["all_stems"]) | set(metas[1]["all_stems"])
        for seg, names in segment_markers_dict.items():
            for raw in names:
                st = str(raw).strip()
                if st not in union_stems:
                    raise ValueError(
                        f"Missing marker {st!r} for segment {seg!r} in at least one CSV "
                        f"(not found in the union of column stems from both passes)"
                    )
        _validate_two_pass_frames_time(metas[0], metas[1])
        can = _canonical_stems_pass1_then_pass2_only(metas[0]["all_stems"], metas[1]["all_stems"])
        if metas[0]["all_stems"] != can or metas[1]["all_stems"] != can:
            logger.info(
                "Aligning marker columns for two-pass merge: %d canonical stems "
                "(pass1=%d cols, pass2=%d cols)",
                len(can),
                len(metas[0]["all_stems"]),
                len(metas[1]["all_stems"]),
            )
        metas[0] = _remap_metadata_to_canonical(metas[0], can)
        metas[1] = _remap_metadata_to_canonical(metas[1], can)
    else:
        meta0 = metas[0]
        for seg, names in segment_markers_dict.items():
            for raw in names:
                st = str(raw).strip()
                if st not in meta0["label_to_marker_idx"]:
                    raise ValueError(f"Missing marker {st!r} for segment {seg!r} in {paths[0]}")

    order = np.argsort(np.array(best_frames, dtype=np.int64))
    if not np.array_equal(order, np.arange(len(paths))):
        logger.info(
            "Reordering passes by increasing best frame: original indices %s -> sorted %s",
            list(range(len(paths))),
            order.tolist(),
        )
        metas = [metas[int(i)] for i in order]
        paths = [paths[int(i)] for i in order]
        best_frames = [best_frames[int(i)] for i in order]

    qc_seg = _resolve_trim_qc_segment_dict(cfg, segment_markers_dict)
    for seg, names in qc_seg.items():
        if len(names) < 3:
            raise ValueError(f"Trim-QC segment {seg!r} needs ≥3 markers")
    for seg, names in qc_seg.items():
        for raw in names:
            st = str(raw).strip()
            if st not in metas[0]["label_to_marker_idx"]:
                raise ValueError(
                    f"Trim-QC marker {st!r} (segment {seg!r}) missing from CSV columns"
                )

    body_union = (
        list(body_marker_names)
        if body_marker_names is not None
        else _markers_union(qc_seg)
    )
    if len(body_union) < int(cfg["envelope_min_visible_markers"]):
        raise ValueError("body_marker_names (or trim-QC segment union) too small for envelope")

    col_eps = float(cfg["collinearity_eps_mm"])
    exclude_obs = bool(cfg["exclude_obstacles_not_in_segments"])
    vis_union_idx, vis_union_lab = build_visibility_union_indices(
        metas[0], qc_seg, exclude_obstacles_not_in_segments=exclude_obs
    )
    vis_ratio_idx, _ = resolve_visibility_ratio_marker_indices(
        metas[0], vis_union_idx, vis_union_lab, cfg
    )

    trusted_row_spans: list[tuple[int, int]] = []
    all_diag: list[list[dict[str, Any]]] = []
    env_info_last: dict[str, Any] = {}

    for pi, meta in enumerate(metas):
        bf = best_frames[pi]
        best_idx = _best_row_idx(meta["frames"], bf)
        ref = compute_reference_geometry(meta, qc_seg, best_idx, collinearity_eps_mm=col_eps)
        centroids, env_out, env_info = compute_subject_envelope(
            meta,
            body_union,
            obstacle_marker_pair,
            envelope_radius_mm=float(cfg["envelope_radius_mm"]),
            envelope_min_visible_markers=int(cfg["envelope_min_visible_markers"]),
            walking_path_y_margin_mm=float(cfg["walking_path_y_margin_mm"]),
        )
        env_info_last = env_info
        diag = compute_diagnostics_with_envelope(
            meta,
            ref,
            qc_seg,
            vis_ratio_idx,
            body_union,
            env_out,
            residual_threshold_mm=float(cfg["residual_threshold_mm"]),
            residual_threshold_thigh_mm=float(cfg["residual_threshold_thigh_mm"]),
            thigh_segment_keywords=tuple(str(x).lower() for x in cfg["thigh_segment_keywords"]),
            visible_ratio_threshold=float(cfg["visible_ratio_threshold"]),
            envelope_outlier_threshold=int(cfg["envelope_outlier_threshold"]),
        )
        d0 = diag[best_idx]
        for sn, val in d0["segment_residual_mm"].items():
            if not np.isfinite(val) or val > 1e-4:
                logger.warning("Pass %s: large residual at best frame for %s: %s mm", pi, sn, val)

        ts_row, te_row = find_trim_boundary(
            diag,
            best_idx,
            consecutive_bad_frames=int(cfg["consecutive_bad_frames"]),
        )
        if not (ts_row <= best_idx < te_row):
            raise ValueError(
                f"Pass {pi}: best frame row not inside trusted trim [{ts_row}, {te_row}); "
                "relax thresholds or fix labeling"
            )
        trusted_row_spans.append((ts_row, te_row))
        all_diag.append(diag)

    frames0 = metas[0]["frames"]
    warnings_list: list[str] = []

    if not two_pass:
        trim_start_row, trim_end_row = trusted_row_spans[0]
        switch_frame: int | None = None
        switch_type = None
        t1_frames = _trusted_range_frames(frames0, trim_start_row, trim_end_row)
        t2_frames = None
        out_bf = int(best_frames[0])
        meta_secondary = metas[0]
    else:
        (t1s_row, t1e_row), (t2s_row, t2e_row) = trusted_row_spans
        t1_start, t1_end = _trusted_range_frames(frames0, t1s_row, t1e_row)
        t2_start, t2_end = _trusted_range_frames(frames0, t2s_row, t2e_row)
        trim_start_row = t1s_row
        trim_end_row = t2e_row
        bf1, bf2 = int(best_frames[0]), int(best_frames[1])

        overlap_len = 0
        gap_len = 0
        if t1_end > t2_start:
            switch_type = "overlap"
            overlap_start = max(t1_start, t2_start)
            overlap_end_excl = min(t1_end, t2_end)
            switch_frame = _find_switch_overlap(
                all_diag[0],
                all_diag[1],
                overlap_start,
                overlap_end_excl,
                frames0,
                bf1,
                bf2,
            )
            overlap_len = max(0, overlap_end_excl - overlap_start)
        else:
            switch_type = "gap"
            gap_start = t1_end
            gap_end = t2_start - 1
            gap_len = gap_end - gap_start + 1
            if gap_len > int(cfg["gap_warning_threshold"]):
                msg = f"Gap length {gap_len} exceeds gap_warning_threshold={cfg['gap_warning_threshold']}"
                warnings.warn(msg, UserWarning, stacklevel=2)
                warnings_list.append(msg)
            switch_frame = _find_switch_gap(bf1, bf2, gap_start, gap_end)

        trim_start_f = int(frames0[trim_start_row])
        trim_end_f = int(frames0[trim_end_row - 1]) + 1
        if not (trim_start_f <= switch_frame < trim_end_f):
            raise ValueError(
                f"switch_frame {switch_frame} not in trimmed half-open range "
                f"[{trim_start_f}, {trim_end_f})"
            )
        out_bf = int(switch_frame)
        meta_secondary = metas[1]

    trim_len = trim_end_row - trim_start_row
    if trim_len < int(cfg["min_trial_length"]):
        msg = f"Trimmed length {trim_len} < min_trial_length={cfg['min_trial_length']}"
        warnings.warn(msg, UserWarning, stacklevel=2)
        warnings_list.append(msg)

    _write_combined_csv(
        metas[0],
        meta_secondary,
        trim_start_row,
        trim_end_row,
        switch_frame,
        two_pass,
        output_csv_path,
    )
    save_bestframe_sidecar(output_csv_path, out_bf)

    out_p = Path(output_csv_path)
    trim_start_f = int(frames0[trim_start_row])
    trim_end_f = int(frames0[trim_end_row - 1]) + 1
    frames_slice = frames0[trim_start_row:trim_end_row]
    times_slice = metas[0]["times"][trim_start_row:trim_end_row]

    if two_pass:
        t1f = _trusted_range_frames(frames0, trusted_row_spans[0][0], trusted_row_spans[0][1])
        t2f = _trusted_range_frames(frames0, trusted_row_spans[1][0], trusted_row_spans[1][1])
    else:
        t1f = (trim_start_f, trim_end_f)
        t2f = None

    _save_trim_log(
        out_p.with_suffix(out_p.suffix + ".trim_log.csv"),
        frames_slice,
        times_slice,
        t1f,
        t2f,
        switch_frame,
        two_pass,
    )

    quality: dict[str, Any] = {
        "mode": "two_pass" if two_pass else "single_pass",
        "trim_qc_preset": cfg.get("trim_qc_preset"),
        "trim_qc_segment_names": list(qc_seg.keys()),
        "best_frames_input": list(best_frames),
        "best_frame_output": out_bf,
        "original_n_frames": int(metas[0]["n_frames"]),
        "trim_start": trim_start_f,
        "trim_end": trim_end_f,
        "trimmed_n_frames": int(trim_len),
        "trusted_ranges": {
            "pass1": list(t1f),
            **({"pass2": list(t2f)} if two_pass and t2f is not None else {}),
        },
        "switch_frame": switch_frame,
        "switch_type": switch_type,
        "warnings": warnings_list,
        "envelope_info": env_info_last,
    }
    if two_pass:
        quality["overlap_length"] = int(overlap_len)
        quality["gap_length"] = int(gap_len)
        quality["frames_from_pass1"] = int(switch_frame - trim_start_f) if switch_frame is not None else 0
        quality["frames_from_pass2"] = int(trim_end_f - switch_frame) if switch_frame is not None else 0

    json_path = out_p.with_suffix(out_p.suffix + ".quality.json")
    json_path.write_text(json.dumps(quality, indent=2))

    if not bool(cfg.get("_skip_plot", False)):
        png_path = out_p.with_suffix(out_p.suffix + ".trusted_ranges.png")
        try:
            plot_trusted_ranges(
                quality["trusted_ranges"],
                best_frames,
                switch_frame,
                trim_start_f,
                trim_end_f,
                int(metas[0]["n_frames"]),
                png_path,
            )
        except ImportError:
            logger.warning("matplotlib not installed; skip trusted_ranges plot")

    _, meta_out = parse_labeled_csv(out_p)
    points_out = meta_out["points"]
    labels_out = meta_out["all_stems"]
    trimmed_markers = {
        labels_out[m]: points_out[:, m, :].copy() for m in range(len(labels_out))
    }

    return {
        "trimmed_markers": trimmed_markers,
        "trim_start": trim_start_f,
        "trim_end": trim_end_f,
        "switch_frame": switch_frame,
        "quality_metrics": quality,
        "trusted_ranges": quality["trusted_ranges"],
        "trim_start_row": trim_start_row,
        "trim_end_row": trim_end_row,
        "best_frame_output": out_bf,
        "best_frames_input": list(best_frames),
    }


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Trim or trim+combine labeled CSV(s) around best frame(s) with envelope-aware QC."
    )
    parser.add_argument("input_csvs", nargs="+", help="One or two labeled CSV paths")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--preset",
        choices=sorted(TRIM_SEGMENT_PRESET_CHOICES),
        default="lower-body",
        help=(
            "Markers that must exist in the CSV (combine/validation). "
            "Same presets as marker-label-trial-trim (e.g. lower-body includes pelvis + legs)."
        ),
    )
    parser.add_argument(
        "--trim-qc-preset",
        dest="trim_qc_preset",
        choices=sorted(TRIM_SEGMENT_PRESET_CHOICES | frozenset({"same-as-segments"})),
        default="legs-feet",
        help=(
            "Segment preset used only for trim trusted-range QC, envelope default body markers, "
            "and overlap switch scoring. Default: legs-feet (thigh–shank–foot). "
            "same-as-segments: use the same marker set as --preset (or --segments-json)."
        ),
    )
    parser.add_argument(
        "--segments-json",
        default=None,
        metavar="PATH",
        help=(
            'If set, overrides --preset. JSON file: {"L_Thigh":["LASI","LTHI","LKNE"], ...}'
        ),
    )
    parser.add_argument(
        "--bestframe",
        nargs="*",
        default=None,
        help="Optional sidecar path(s); default one per input (*.csv.bestframe)",
    )
    parser.add_argument(
        "--obstacle-pair",
        nargs=2,
        metavar=("L", "R"),
        default=None,
        help="Obstacle marker names for walking-path Y filter",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing .trusted_ranges.png",
    )
    args = parser.parse_args()

    if args.segments_json is not None:
        path = Path(args.segments_json)
        if not path.is_file():
            print(f"Error: segments JSON not found: {path.resolve()}", file=sys.stderr)
            raise SystemExit(2)
        seg = json.loads(path.read_text())
        if not isinstance(seg, dict):
            print("segments-json must be a JSON object", file=sys.stderr)
            raise SystemExit(2)
        segment_dict = {str(k): list(v) for k, v in seg.items()}
        print("Note: using --segments-json for segment definitions (--preset ignored).", file=sys.stderr)
    else:
        segment_dict = segment_markers_dict_for_trim_preset(args.preset)
        if args.preset == "full-body":
            from .segments import SEGMENTS

            skipped = sorted(k for k, v in SEGMENTS.items() if len(v) < 3)
            if skipped:
                print(
                    "Note: skipping SEGMENTS with fewer than 3 markers: " + ", ".join(skipped),
                    file=sys.stderr,
                )

    pair = tuple(args.obstacle_pair) if args.obstacle_pair else None
    cfg: dict[str, Any] = {}
    if args.no_plot:
        cfg["_skip_plot"] = True
    if args.segments_json is not None:
        cfg["trim_qc_preset"] = None
    else:
        tq = args.trim_qc_preset
        cfg["trim_qc_preset"] = None if tq == "same-as-segments" else tq

    res = trim_and_combine(
        args.input_csvs,
        args.output,
        segment_dict,
        bestframe_paths=args.bestframe if args.bestframe else None,
        obstacle_marker_pair=pair,
        config=cfg,
    )

    print(json.dumps({k: res[k] for k in ("trim_start", "trim_end", "switch_frame", "best_frame_output")}, indent=2))


if __name__ == "__main__":
    main()
