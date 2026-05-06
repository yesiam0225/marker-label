"""Tests for trimmed CSV marker correction (reference-guided tiers and swaps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marker_label.trial_trim import bestframe_sidecar_path
from marker_label.trimmed_csv_marker_correction import correct_markers


def _write_flat_csv(
    path: Path,
    stems: list[str],
    *,
    n_frames: int = 8,
    z: float = 1000.0,
    swap_frame: int | None = None,
    swap_pair: tuple[str, str] | None = None,
    outlier: tuple[int, str, np.ndarray] | None = None,
    sustained: tuple[str, int, int] | None = None,
) -> None:
    """Rigid-ish layout: markers placed on a grid in x-y at height z."""
    lines = ["frame,time," + ",".join(f"{s}_x,{s}_y,{s}_z" for s in stems) + "\n"]
    # Fixed non-coplanar template; per-stem translation so lists stay non-degenerate.
    tpl = np.array(
        [
            [0.0, 0.0, 0.0],
            [120.0, 5.0, 35.0],
            [45.0, 95.0, 70.0],
            [80.0, 50.0, 130.0],
        ],
        dtype=np.float64,
    )
    pos: dict[str, np.ndarray] = {}
    for i, s in enumerate(stems):
        pos[s] = tpl[i % len(tpl)].copy() + np.array([float(i * 3.0), float(i * 7.0), z], dtype=np.float64)
    for f in range(n_frames):
        t = f * 0.01
        row_pos = dict(pos)
        if swap_frame is not None and swap_pair is not None and f == swap_frame:
            a, b = swap_pair
            row_pos[a], row_pos[b] = pos[b].copy(), pos[a].copy()
        if outlier is not None and f == outlier[0]:
            row_pos[outlier[1]] = outlier[2]
        if sustained is not None:
            name, f0, f1 = sustained
            if f0 <= f <= f1 and name in row_pos:
                row_pos[name] = row_pos[name] + np.array([200.0, 0.0, 0.0])
        parts = [str(f), f"{t:.4f}"]
        for s in stems:
            p = row_pos[s]
            parts.extend([str(p[0]), str(p[1]), str(p[2])])
        lines.append(",".join(parts) + "\n")
    path.write_text("".join(lines))


def test_noop_small_trial(tmp_path: Path) -> None:
    stems = ["A", "B", "C", "OBSTACLE_L", "OBSTACLE_R"]
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_flat_csv(src, stems)
    _write_flat_csv(trim, stems)
    bestframe_sidecar_path(trim).write_text("4")
    bestframe_sidecar_path(src).write_text("4")
    seg = {"left_thigh": ["A", "B", "C"]}
    cfg = {"envelope_min_visible_markers": 3}
    res = correct_markers(
        trim,
        src,
        None,
        out,
        seg,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        config=cfg,
    )
    assert out.is_file()
    assert len(res["correction_log"]) == 0
    assert (
        res["quality_metrics"]["reference_source"]["original_best_frame_source"]
        == "original_csv_bestframe_sidecar"
    )


def test_trunk_t10_strn_swap_reverted(tmp_path: Path) -> None:
    stems = ["C7", "CLAV", "MID", "T10", "STRN", "OBSTACLE_L", "OBSTACLE_R"]
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_flat_csv(src, stems)
    _write_flat_csv(trim, stems, swap_frame=3, swap_pair=("T10", "STRN"))
    bestframe_sidecar_path(trim).write_text("4")
    seg = {"trunk": ["C7", "CLAV", "MID", "T10", "STRN"]}
    cfg = {
        "envelope_min_visible_markers": 3,
        "swap_ratio": 0.3,
        "swap_absolute_max_mm": 500.0,
    }
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(("T10", "STRN"),),
        cross_segment_swap_pairs=(),
        config=cfg,
    )
    actions = res["correction_log"]["action"].tolist() if len(res["correction_log"]) else []
    assert "same_segment_swap" in actions


def test_pelvis_permutation_recovery(tmp_path: Path) -> None:
    stems = ["LASI", "RASI", "LPSI", "RPSI", "OBSTACLE_L", "OBSTACLE_R"]
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_flat_csv(src, stems, z=500.0)
    _write_flat_csv(trim, stems, z=500.0, swap_frame=2, swap_pair=("LASI", "RASI"))
    bestframe_sidecar_path(trim).write_text("4")
    seg = {"Pelvis": ["LASI", "RASI", "LPSI", "RPSI"]}
    cfg = {
        "envelope_min_visible_markers": 3,
        "pelvis_swap_ratio": 0.99,
        "swap_absolute_max_mm": 500.0,
    }
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        pelvis_segment_name="Pelvis",
        config=cfg,
    )
    acts = res["correction_log"]["action"].astype(str).tolist()
    assert any("pelvis_combinatorial" in a for a in acts)


def test_wrong_label_hand_near_thigh(tmp_path: Path) -> None:
    stems = ["LTHI", "LKNE", "LTIB", "AUX", "LFIN", "LFRM", "LWRA", "OBSTACLE_L", "OBSTACLE_R"]
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_flat_csv(src, stems)
    # Copy LTHI's coordinates onto LFIN at one frame (hand channel shows thigh point).
    p_lthi = None
    with open(src) as f:
        hdr = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if int(parts[0]) == 4:
                ix = hdr.split(",").index("LTHI_x")
                p_lthi = np.array([float(parts[ix]), float(parts[ix + 1]), float(parts[ix + 2])])
                break
    assert p_lthi is not None
    _write_flat_csv(trim, stems, outlier=(2, "LFIN", p_lthi.copy()))
    bestframe_sidecar_path(trim).write_text("4")
    seg = {
        "left_thigh": ["LTHI", "LKNE", "LTIB", "AUX"],
        "left_hand": ["LFIN", "LFRM", "LWRA"],
    }
    marker_tiers = {s: 1 for s in ("LTHI", "LKNE", "LTIB", "AUX")}
    marker_tiers.update({s: 3 for s in ("LFIN", "LFRM", "LWRA")})
    cfg = {"envelope_min_visible_markers": 3, "wrong_label_distance_mm": 150.0}
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        marker_tiers=marker_tiers,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(("LFIN", "LTHI"),),
        config=cfg,
    )
    log_df = res["correction_log"]
    acts = log_df["action"].astype(str).tolist() if len(log_df) and "action" in log_df.columns else []
    assert any("wrong_label" in a for a in acts)


def test_single_frame_outlier_rejected(tmp_path: Path) -> None:
    stems = ["C7", "CLAV", "MID", "T10", "STRN", "OBSTACLE_L", "OBSTACLE_R"]
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_flat_csv(src, stems)
    # Small spatial spike on one frame (still inside envelope) so stage-1 does not clear T10 first.
    hdr = "frame,time," + ",".join(f"{s}_x,{s}_y,{s}_z" for s in stems) + "\n"
    lines = [hdr]
    for f in range(8):
        t = f * 0.01
        row_pos: dict[str, np.ndarray] = {}
        tpl = np.array(
            [[0.0, 0.0, 0.0], [120.0, 5.0, 35.0], [45.0, 95.0, 70.0], [80.0, 50.0, 130.0]],
            dtype=np.float64,
        )
        for i, s in enumerate(stems):
            row_pos[s] = tpl[i % len(tpl)].copy() + np.array([float(i * 3.0), float(i * 7.0), 1000.0])
        if f == 4:
            row_pos["T10"] = row_pos["T10"] + np.array([120.0, 0.0, 0.0])
        parts = [str(f), f"{t:.4f}"]
        for s in stems:
            p = row_pos[s]
            parts.extend([str(p[0]), str(p[1]), str(p[2])])
        lines.append(",".join(parts) + "\n")
    trim.write_text("".join(lines))
    bestframe_sidecar_path(trim).write_text("4")
    seg = {"trunk": ["C7", "CLAV", "MID", "T10", "STRN"]}
    cfg = {
        "envelope_min_visible_markers": 3,
        "tier_suspicious_threshold_mm": {1: 25, 2: 80.0, 3: 500.0},
        "suspicious_neighbor_window": 2,
        "same_segment_swap_pairs": (),
        "cross_segment_swap_pairs": (),
    }
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        marker_tiers={"C7": 2, "CLAV": 2, "MID": 2, "T10": 2, "STRN": 2},
        config=cfg,
    )
    log_df = res["correction_log"]
    acts = log_df["action"].astype(str).tolist() if len(log_df) and "action" in log_df.columns else []
    assert any("single_frame_outlier" in a for a in acts)


def test_sustained_drift_warning_not_rejected(tmp_path: Path) -> None:
    stems = ["C7", "CLAV", "MID", "T10", "STRN", "OBSTACLE_L", "OBSTACLE_R"]
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_flat_csv(src, stems)
    _write_flat_csv(trim, stems, sustained=("T10", 1, 6))
    bestframe_sidecar_path(trim).write_text("4")
    seg = {"trunk": ["C7", "CLAV", "MID", "T10", "STRN"]}
    cfg = {
        "envelope_min_visible_markers": 3,
        "tier_suspicious_threshold_mm": {2: 50.0},
        "suspicious_neighbor_window": 2,
    }
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        marker_tiers={"C7": 2, "CLAV": 2, "MID": 2, "T10": 2, "STRN": 2},
        config=cfg,
    )
    warns = res["quality_metrics"].get("sustained_drift_warnings", [])
    assert isinstance(warns, list)
    assert len(warns) >= 1


def test_trimmed_extra_unlabeled_columns_ok(tmp_path: Path) -> None:
    """Original may omit numeric / extra columns that exist only on the trimmed merge."""
    stems_full = ["A", "B", "C", "99", "OBSTACLE_L", "OBSTACLE_R"]
    stems_orig = ["A", "B", "C", "OBSTACLE_L", "OBSTACLE_R"]
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_flat_csv(src, stems_orig)
    _write_flat_csv(trim, stems_full)
    bestframe_sidecar_path(trim).write_text("4")
    seg = {"left_thigh": ["A", "B", "C"]}
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        config={"envelope_min_visible_markers": 3},
    )
    assert "trimmed_only_marker_columns" in res["quality_metrics"]
    assert "99" in res["quality_metrics"]["trimmed_only_marker_columns"]


def test_trimmed_original_column_mismatch_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_flat_csv(a, ["A", "B", "C"])
    _write_flat_csv(b, ["A", "B", "X"])
    bestframe_sidecar_path(a).write_text("1")
    with pytest.raises(ValueError, match="only_in_original_not_in_trimmed"):
        correct_markers(
            a,
            b,
            1,
            tmp_path / "o.csv",
            {"s": ["A", "B", "C"]},
            config={"envelope_min_visible_markers": 3},
        )
