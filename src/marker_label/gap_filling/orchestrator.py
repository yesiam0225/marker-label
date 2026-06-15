"""Gap-fill pipeline: rigid → ASIS-only pelvis → spline → continuity → unfillable log."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from marker_label.trial_trim import bestframe_sidecar_path, parse_labeled_csv

from .asis_only_fill import asis_only_fill
from .continuity_check import apply_continuity_check_to_fills, frame_column_to_row
from .foot_heel_synthesis import qc_stems_from_meta, synthesize_missing_foot_heels
from .hand_marker_synthesis import (
    synthesize_hand_from_contralateral,
    synthesize_missing_hand_markers,
)
from .gap_detection import categorize_gap, find_gaps
from .quality import compute_quality_metrics
from .reference import build_robust_reference
from .rigid_fill import rigid_body_fill
from .shoulder_from_thorax import shoulder_from_thorax_fill
from .spline_fill import spline_fill
from .static_reference import load_static_reference_bundle
from .two_marker_static import static_offsets_for_three_marker_segment
from .visualization import plot_gap_fill_summary

logger = logging.getLogger(__name__)

DEFAULT_GAP_FILLING_CONFIG: dict[str, Any] = {
    "reference_n_clean_samples": 30,
    "rigid_fill_high_residual_mm": 5,
    "rigid_fill_medium_residual_mm": 15,
    "rigid_fill_max_residual_mm": 30,
    "spline_max_length": 10,
    "spline_padding_frames": 5,
    "spline_min_anchor_frames": 2,
    "asis_only_search_radius": 40,
    "asis_only_min_anchor_frames": 2,
    "validate_continuity": True,
    "max_velocity_mm_per_frame": 50,
    "continuity_violation_action": "downgrade",
    "long_gap_threshold": 50,
    "attempt_long_gaps": True,
    "asis_only_enabled": True,
    "allow_two_marker_rigid": True,
    "two_marker_rigid_segment_names": None,
    "lab_vertical": (0.0, 1.0, 0.0),
    "enable_shoulder_from_thorax": False,
    "shoulder_markers": ("LSHO", "RSHO"),
    "thorax_markers": ("C7", "CLAV", "RBAK"),
    "synthesize_missing_foot_heels": True,
    "synthesize_missing_hand_markers": True,
    "synthesize_hand_from_contralateral": True,
}


def _sync_data_rows_from_points(meta: dict[str, Any]) -> None:
    stem_to_triplet: dict[str, tuple[int, int, int]] = meta["stem_to_col_triplet"]
    all_stems: list[str] = meta["all_stems"]
    points: np.ndarray = meta["points"]
    rows: list[list[str]] = meta["data_rows"]
    for r in range(int(meta["n_frames"])):
        row = rows[r]
        for si, stem in enumerate(all_stems):
            ix, iy, iz = stem_to_triplet[stem]
            for k, j in enumerate((ix, iy, iz)):
                v = points[r, si, k]
                row[j] = "" if not np.isfinite(v) else str(float(v))


def _read_best_frame_column(csv_path: Path) -> int | None:
    p = bestframe_sidecar_path(csv_path)
    if not p.is_file():
        return None
    try:
        return int(p.read_text().strip().split()[0])
    except (ValueError, OSError):
        return None


def _unfillable_row(
    frame: int,
    marker: str,
    gap_length: int,
    reason: str = "unfillable_remaining",
) -> dict[str, Any]:
    return {
        "frame": int(frame),
        "marker": marker,
        "method": "unfillable",
        "success": False,
        "confidence": "",
        "predicted_x": np.nan,
        "predicted_y": np.nan,
        "predicted_z": np.nan,
        "fit_residual_mm": np.nan,
        "source_markers": "",
        "gap_length": int(gap_length),
        "reason": reason,
    }


def _apply_success_row(meta: dict[str, Any], row: Mapping[str, Any], frames: np.ndarray) -> None:
    if not row.get("success"):
        return
    fc = int(row["frame"])
    r = frame_column_to_row(frames, fc)
    if r is None:
        return
    m = str(row["marker"]).strip()
    mi = meta["label_to_marker_idx"][m]
    meta["points"][r, mi, 0] = float(row["predicted_x"])
    meta["points"][r, mi, 1] = float(row["predicted_y"])
    meta["points"][r, mi, 2] = float(row["predicted_z"])


def gap_fill(
    input_csv_path: str,
    output_csv_path: str,
    segment_markers_dict: dict[str, list[str]],
    bilateral_pelvis_markers: tuple[str, ...] = ("LPSI", "RPSI"),
    asis_markers: tuple[str, ...] = ("LASI", "RASI"),
    config: dict[str, Any] | None = None,
    *,
    static_csv_path: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Load corrected CSV, fill gaps, write filled CSV and sidecar artifacts.

    Preserves ``*.csv.bestframe`` from input to output path when present.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    cfg: dict[str, Any] = dict(DEFAULT_GAP_FILLING_CONFIG)
    if config:
        cfg.update(config)

    in_path = Path(input_csv_path)
    out_path = Path(output_csv_path)
    qc_stems, meta = parse_labeled_csv(in_path)
    points_before = np.array(meta["points"], copy=True)
    frames = np.asarray(meta["frames"], dtype=np.int64)
    label_to_idx: dict[str, int] = meta["label_to_marker_idx"]

    bilateral = tuple(str(x).strip() for x in bilateral_pelvis_markers)
    asis = tuple(str(x).strip() for x in asis_markers)

    bf = _read_best_frame_column(in_path)
    lab_vertical = tuple(float(x) for x in cfg.get("lab_vertical", (0.0, 1.0, 0.0)))
    reference_source = "dynamic"
    two_marker_offsets: dict[str, dict[str, tuple]] = {}
    pelvis_psi_static: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    shoulder_local: dict[str, np.ndarray] = {}

    if static_csv_path:
        bundle = load_static_reference_bundle(
            static_csv_path,
            segment_markers_dict,
            label_to_idx,
            n_samples=int(cfg["reference_n_clean_samples"]),
            lab_vertical=lab_vertical,
        )
        reference = bundle["reference"]
        ref_warnings = bundle["warnings"]
        two_marker_offsets = bundle["two_marker_offsets"]
        pelvis_psi_static = bundle["pelvis_psi_offset"]
        shoulder_local = bundle["shoulder_local"]
        reference_source = "static"
        logger.info("Using static trial reference: %s", static_csv_path)
    else:
        reference, ref_warnings = build_robust_reference(
            meta["points"],
            label_to_idx,
            segment_markers_dict,
            n_samples=int(cfg["reference_n_clean_samples"]),
            best_frame=bf,
            csv_path=in_path,
            frames=frames,
        )
        vert_arr = np.array(lab_vertical, dtype=np.float64)
        for seg, names in segment_markers_dict.items():
            offs = static_offsets_for_three_marker_segment(
                meta["points"], label_to_idx, names, vert_arr
            )
            if offs:
                two_marker_offsets[str(seg)] = offs
    for w in ref_warnings:
        logger.warning("%s", w)

    heel_fills: list[dict[str, Any]] = []
    hand_fills: list[dict[str, Any]] = []
    contra_fills: list[dict[str, Any]] = []
    if static_csv_path and cfg.get("synthesize_missing_foot_heels", True):
        heel_fills = synthesize_missing_foot_heels(
            meta,
            segment_markers_dict,
            two_marker_offsets,
            frames,
            lab_vertical,
        )
        if heel_fills:
            logger.info(
                "Synthesized %d missing foot heel positions (static_foot_heel)",
                len(heel_fills),
            )
            qc_stems = qc_stems_from_meta(meta)
            label_to_idx = meta["label_to_marker_idx"]
            if meta["points"].shape[1] > points_before.shape[1]:
                expanded = np.full(
                    (points_before.shape[0], meta["points"].shape[1], 3),
                    np.nan,
                    dtype=np.float64,
                )
                expanded[:, :points_before.shape[1], :] = points_before
                points_before = expanded

    if static_csv_path and cfg.get("synthesize_missing_hand_markers", True):
        hand_fills = synthesize_missing_hand_markers(
            meta,
            segment_markers_dict,
            static_csv_path,
            frames,
            label_to_idx,
            lab_vertical,
        )
        if hand_fills:
            logger.info(
                "Synthesized %d missing hand marker positions (static_hand_forearm)",
                len(hand_fills),
            )
            qc_stems = qc_stems_from_meta(meta)
            label_to_idx = meta["label_to_marker_idx"]
            if meta["points"].shape[1] > points_before.shape[1]:
                expanded = np.full(
                    (points_before.shape[0], meta["points"].shape[1], 3),
                    np.nan,
                    dtype=np.float64,
                )
                n_old = min(points_before.shape[1], expanded.shape[1])
                expanded[:, :n_old, :] = points_before[:, :n_old, :]
                points_before = expanded

    if cfg.get("synthesize_hand_from_contralateral", True):
        contra_fills = synthesize_hand_from_contralateral(
            meta,
            segment_markers_dict,
            frames,
            lab_vertical,
            static_csv_path=static_csv_path,
            dynamic_label_to_idx=label_to_idx,
        )
        if contra_fills:
            logger.info(
                "Synthesized %d hand marker positions from contralateral side",
                len(contra_fills),
            )
            qc_stems = qc_stems_from_meta(meta)
            label_to_idx = meta["label_to_marker_idx"]
            if meta["points"].shape[1] > points_before.shape[1]:
                expanded = np.full(
                    (points_before.shape[0], meta["points"].shape[1], 3),
                    np.nan,
                    dtype=np.float64,
                )
                n_old = min(points_before.shape[1], expanded.shape[1])
                expanded[:, :n_old, :] = points_before[:, :n_old, :]
                points_before = expanded

    initial_gaps: dict[str, list[tuple[int, int, int]]] = {}
    for stem in qc_stems:
        if stem not in label_to_idx:
            continue
        mi = label_to_idx[stem]
        initial_gaps[stem] = find_gaps(meta["points"][:, mi, :])

    fills_map: dict[tuple[int, str], dict[str, Any]] = {}
    for r in heel_fills:
        fills_map[(int(r["frame"]), str(r["marker"]).strip())] = dict(r)
    for r in hand_fills:
        fills_map[(int(r["frame"]), str(r["marker"]).strip())] = dict(r)
    for r in contra_fills:
        fills_map[(int(r["frame"]), str(r["marker"]).strip())] = dict(r)

    # --- Pass 1: rigid (only gaps categorized as rigid on *initial* layout) ---
    logger.info("Pass 1: rigid_body")
    for marker, gaps in initial_gaps.items():
        if marker not in label_to_idx:
            continue
        for gap in gaps:
            cat, seg = categorize_gap(
                marker,
                gap,
                meta["points"],
                label_to_idx,
                segment_markers_dict,
                bilateral,
                asis,
                cfg,
            )
            if cat != "rigid_body" or seg is None:
                continue
            rows = rigid_body_fill(
                meta["points"],
                marker,
                gap,
                seg,
                segment_markers_dict,
                reference,
                label_to_idx,
                cfg,
                frames,
                two_marker_offsets=two_marker_offsets,
                lab_vertical=lab_vertical,
            )
            for r in rows:
                key = (int(r["frame"]), str(marker).strip())
                if r.get("success"):
                    fills_map[key] = dict(r)
                    _apply_success_row(meta, r, frames)
                elif key not in fills_map:
                    fills_map[key] = dict(r)

    # --- Pass 2: ASIS-only for PSI markers on remaining gaps ---
    logger.info("Pass 2: asis_only pelvis fallback")
    for marker in bilateral:
        if marker not in label_to_idx or not cfg.get("asis_only_enabled", True):
            continue
        mi = label_to_idx[marker]
        for gap in find_gaps(meta["points"][:, mi, :]):
            cat, _ = categorize_gap(
                marker,
                gap,
                meta["points"],
                label_to_idx,
                segment_markers_dict,
                bilateral,
                asis,
                cfg,
            )
            if cat != "asis_only":
                continue
            static_off = pelvis_psi_static.get(str(marker).strip())
            rows = asis_only_fill(
                meta["points"],
                marker,
                gap,
                asis,
                segment_markers_dict,
                reference,
                label_to_idx,
                cfg,
                frames,
                static_psi_offset=static_off,
            )
            for r in rows:
                key = (int(r["frame"]), str(marker).strip())
                if r.get("success"):
                    fills_map[key] = dict(r)
                    _apply_success_row(meta, r, frames)
                elif key not in fills_map:
                    fills_map[key] = dict(r)

    # --- Pass 2b: shoulder from thorax (walking / minimal arm swing) ---
    if cfg.get("enable_shoulder_from_thorax") and shoulder_local:
        logger.info("Pass 2b: shoulder_from_thorax")
        for sh in cfg.get("shoulder_markers", ("LSHO", "RSHO")):
            sh = str(sh).strip()
            if sh not in label_to_idx:
                continue
            mi = label_to_idx[sh]
            for gap in find_gaps(meta["points"][:, mi, :]):
                rows = shoulder_from_thorax_fill(
                    meta["points"],
                    sh,
                    gap,
                    label_to_idx,
                    frames,
                    shoulder_local,
                    thorax_markers=tuple(str(x).strip() for x in cfg.get("thorax_markers", ("C7", "CLAV", "RBAK"))),
                )
                for r in rows:
                    key = (int(r["frame"]), sh)
                    if r.get("success"):
                        fills_map[key] = dict(r)
                        _apply_success_row(meta, r, frames)
                    elif key not in fills_map:
                        fills_map[key] = dict(r)

    # --- Pass 3: spline on remaining short gaps ---
    logger.info("Pass 3: spline")
    for marker in qc_stems:
        if marker not in label_to_idx:
            continue
        mi = label_to_idx[marker]
        for gap in find_gaps(meta["points"][:, mi, :]):
            cat, _ = categorize_gap(
                marker,
                gap,
                meta["points"],
                label_to_idx,
                segment_markers_dict,
                bilateral,
                asis,
                cfg,
            )
            if cat != "spline":
                continue
            rows = spline_fill(
                meta["points"],
                marker,
                gap,
                label_to_idx,
                cfg,
                frames,
            )
            for r in rows:
                key = (int(r["frame"]), str(marker).strip())
                if r.get("success"):
                    fills_map[key] = dict(r)
                    _apply_success_row(meta, r, frames)
                elif key not in fills_map:
                    fills_map[key] = dict(r)

    # --- Pass 4: continuity ---
    cont_w = rev = 0
    if cfg.get("validate_continuity", True):
        logger.info("Pass 4: continuity validation")
        fills_list = list(fills_map.values())
        cont_w, rev = apply_continuity_check_to_fills(
            meta["points"],
            fills_list,
            label_to_idx,
            frames,
            action=str(cfg.get("continuity_violation_action", "downgrade")),
            max_velocity_mm_per_frame=float(cfg.get("max_velocity_mm_per_frame", 50)),
        )
        for r in fills_list:
            key = (int(r["frame"]), str(r["marker"]).strip())
            fills_map[key] = r
            if not r.get("success"):
                mi = label_to_idx[str(r["marker"]).strip()]
                rr = frame_column_to_row(frames, int(r["frame"]))
                if rr is not None:
                    meta["points"][rr, mi, :] = np.nan

    # --- Pass 5: remaining NaNs → unfillable ---
    logger.info("Pass 5: unfillable log")
    for marker in qc_stems:
        if marker not in label_to_idx:
            continue
        mi = label_to_idx[marker]
        for start, end, length in find_gaps(meta["points"][:, mi, :]):
            for row in range(start, end + 1):
                fc = int(frames[row])
                key = (fc, str(marker).strip())
                if key in fills_map and fills_map[key].get("success"):
                    continue
                fills_map[key] = _unfillable_row(fc, marker, length)

    fills_log = sorted(fills_map.values(), key=lambda x: (int(x["frame"]), str(x["marker"])))

    quality = compute_quality_metrics(
        points_before,
        meta["points"],
        qc_stems,
        label_to_idx,
        fills_log,
        segment_markers_dict,
        continuity_warnings=cont_w,
        reverted_fills=rev,
    )
    quality["trial_name"] = in_path.name
    quality["reference_source"] = reference_source
    if static_csv_path:
        quality["static_csv"] = str(static_csv_path)

    stem = str(out_path)
    fills_path = stem.replace(".csv", "_fills.csv")
    quality_path = stem.replace(".csv", "_quality.json")
    summary_path = stem.replace(".csv", "_summary.png")

    _sync_data_rows_from_points(meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write(meta["original_header_line"])
        w = csv.writer(f)
        for row in meta["data_rows"]:
            w.writerow(row)

    with open(fills_path, "w", newline="") as f:
        cols = [
            "frame",
            "marker",
            "method",
            "success",
            "confidence",
            "predicted_x",
            "predicted_y",
            "predicted_z",
            "fit_residual_mm",
            "source_markers",
            "gap_length",
            "reason",
        ]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in fills_log:
            w.writerow(row)

    Path(quality_path).write_text(json.dumps(quality, indent=2))
    plot_gap_fill_summary(quality, summary_path)

    src_bf = bestframe_sidecar_path(in_path)
    if src_bf.is_file():
        dst_bf = bestframe_sidecar_path(out_path)
        shutil.copy2(src_bf, dst_bf)

    logger.info(
        "Gap fill done: %s fills logged, outputs -> %s",
        len(fills_log),
        out_path,
    )

    return {
        "output_csv": str(out_path),
        "fills_csv": fills_path,
        "quality_json": quality_path,
        "summary_png": summary_path,
        "quality": quality,
        "fills_log": fills_log,
    }
