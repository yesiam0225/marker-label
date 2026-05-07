"""Tests for trimmed CSV marker correction (reference-guided tiers and swaps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marker_label.trial_trim import bestframe_sidecar_path
from marker_label.trimmed_csv_marker_correction import (
    correct_markers,
    find_consecutive_intervals,
)


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
        bilateral_chains={},
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
        bilateral_chains={},
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
        bilateral_chains={},
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
        bilateral_chains={},
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
        bilateral_chains={},
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
        bilateral_chains={},
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
        bilateral_chains={},
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        config={"envelope_min_visible_markers": 3},
    )
    assert "trimmed_only_marker_columns" in res["quality_metrics"]
    assert "99" in res["quality_metrics"]["trimmed_only_marker_columns"]


def _write_bilateral_chain_csv(
    path: Path,
    n_frames: int,
    *,
    bad_slice: tuple[int, int] | None = None,
    midline_arms: bool = False,
    lasi_nan_row: int | None = None,
) -> None:
    """LASI/RASI define +Y=left; AL1/AL2 vs AR1/AR2 are a minimal bilateral chain."""
    stems = ["LASI", "RASI", "LPSI", "AL1", "AL2", "AR1", "AR2", "OBSTACLE_L", "OBSTACLE_R"]
    lines = ["frame,time," + ",".join(f"{s}_x,{s}_y,{s}_z" for s in stems) + "\n"]
    for f in range(n_frames):
        t = f * 0.01
        px = -80.0 + 2.0 * f
        lasi = np.array([px, 100.0, 1000.0])
        rasi = np.array([px, -100.0, 1000.0])
        lpsi = np.array([px + 40.0, 0.0, 1020.0])
        if midline_arms:
            al1, al2 = np.array([6.0, 8.0, 1000.0]), np.array([10.0, 12.0, 1005.0])
            ar1, ar2 = np.array([-6.0, -8.0, 1000.0]), np.array([-10.0, -12.0, 1005.0])
        else:
            al1, al2 = np.array([12.0, 80.0, 1000.0]), np.array([18.0, 86.0, 1010.0])
            ar1, ar2 = np.array([-10.0, -80.0, 1000.0]), np.array([-16.0, -84.0, 1005.0])
        if bad_slice is not None:
            lo, hi = bad_slice
            if lo <= f <= hi:
                al1, al2, ar1, ar2 = ar1.copy(), ar2.copy(), al1.copy(), al2.copy()
        obs_l = np.array([0.0, 500.0, 0.0])
        obs_r = np.array([0.0, 600.0, 0.0])
        if lasi_nan_row is not None and f == lasi_nan_row:
            lasi = np.full(3, np.nan)
        row = [str(f), f"{t:.4f}"]
        for p in (lasi, rasi, lpsi, al1, al2, ar1, ar2, obs_l, obs_r):
            row.extend([str(p[0]), str(p[1]), str(p[2])])
        lines.append(",".join(row) + "\n")
    path.write_text("".join(lines))


def test_find_consecutive_intervals() -> None:
    assert find_consecutive_intervals([True] * 9 + [False], 10) == []
    assert find_consecutive_intervals([True] * 10, 10) == [(0, 9)]


def test_full_bilateral_chain_swap_reverted(tmp_path: Path) -> None:
    n = 80
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_bilateral_chain_csv(src, n, bad_slice=None)
    _write_bilateral_chain_csv(trim, n, bad_slice=(30, 55))
    bestframe_sidecar_path(trim).write_text("4")
    bestframe_sidecar_path(src).write_text("4")
    seg = {
        "pelvis": ["LASI", "RASI", "LPSI"],
        "left_arm": ["LASI", "AL1", "AL2"],
        "right_arm": ["RASI", "AR1", "AR2"],
    }
    chains = {"arm": (["AL1", "AL2"], ["AR1", "AR2"])}
    cfg = {"envelope_min_visible_markers": 3, "chain_swap_min_consecutive_frames": 10}
    # Keep pelvis + arms out of envelope NaN-ing (they are tier-2/3 by segment names).
    mt = {s: 1 for s in ("LASI", "RASI", "LPSI", "AL1", "AL2", "AR1", "AR2", "OBSTACLE_L", "OBSTACLE_R")}
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        bilateral_chains=chains,
        marker_tiers=mt,
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        config=cfg,
    )
    q = res["quality_metrics"]
    assert q["corrections_by_stage"].get("stage5_5_full_chain_swap_intervals", 0) >= 1
    assert len(q.get("full_chain_swap_intervals", [])) >= 1
    # After revert, left markers should be back on +Y side at a mid-bad frame
    from marker_label.trial_trim import parse_labeled_csv

    _, meta = parse_labeled_csv(out)
    pts = meta["points"]
    l2i = meta["label_to_marker_idx"]
    f40 = 40
    yl = float(np.mean([pts[f40, l2i["AL1"], 1], pts[f40, l2i["AL2"], 1]]))
    assert yl > 50.0


def test_chain_swap_short_interval_not_triggered(tmp_path: Path) -> None:
    n = 60
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_bilateral_chain_csv(src, n)
    _write_bilateral_chain_csv(trim, n, bad_slice=(20, 26))
    bestframe_sidecar_path(trim).write_text("4")
    bestframe_sidecar_path(src).write_text("4")
    seg = {
        "pelvis": ["LASI", "RASI", "LPSI"],
        "la": ["LASI", "AL1", "AL2"],
        "ra": ["RASI", "AR1", "AR2"],
    }
    mt = {s: 1 for s in ("LASI", "RASI", "LPSI", "AL1", "AL2", "AR1", "AR2", "OBSTACLE_L", "OBSTACLE_R")}
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        bilateral_chains={"arm": (["AL1", "AL2"], ["AR1", "AR2"])},
        marker_tiers=mt,
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        config={"envelope_min_visible_markers": 3, "chain_swap_min_consecutive_frames": 10},
    )
    assert res["quality_metrics"]["corrections_by_stage"].get("stage5_5_full_chain_swap_intervals", 0) == 0


def test_chain_swap_midline_not_triggered(tmp_path: Path) -> None:
    n = 40
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_bilateral_chain_csv(src, n, midline_arms=True)
    _write_bilateral_chain_csv(trim, n, midline_arms=True, bad_slice=(5, 30))
    bestframe_sidecar_path(trim).write_text("4")
    bestframe_sidecar_path(src).write_text("4")
    seg = {
        "pelvis": ["LASI", "RASI", "LPSI"],
        "la": ["LASI", "AL1", "AL2"],
        "ra": ["RASI", "AR1", "AR2"],
    }
    mt = {s: 1 for s in ("LASI", "RASI", "LPSI", "AL1", "AL2", "AR1", "AR2", "OBSTACLE_L", "OBSTACLE_R")}
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        bilateral_chains={"arm": (["AL1", "AL2"], ["AR1", "AR2"])},
        marker_tiers=mt,
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        config={"envelope_min_visible_markers": 3},
    )
    assert res["quality_metrics"]["corrections_by_stage"].get("stage5_5_full_chain_swap_intervals", 0) == 0


def test_chain_swap_lasi_nan_breaks_run(tmp_path: Path) -> None:
    n = 80
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_bilateral_chain_csv(src, n)
    # Swapped segment split by one LASI-NaN frame so no run reaches min consecutive length.
    _write_bilateral_chain_csv(trim, n, bad_slice=(40, 54), lasi_nan_row=47)
    bestframe_sidecar_path(trim).write_text("4")
    bestframe_sidecar_path(src).write_text("4")
    seg = {
        "pelvis": ["LASI", "RASI", "LPSI"],
        "la": ["LASI", "AL1", "AL2"],
        "ra": ["RASI", "AR1", "AR2"],
    }
    mt = {s: 1 for s in ("LASI", "RASI", "LPSI", "AL1", "AL2", "AR1", "AR2", "OBSTACLE_L", "OBSTACLE_R")}
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        bilateral_chains={"arm": (["AL1", "AL2"], ["AR1", "AR2"])},
        marker_tiers=mt,
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        config={"envelope_min_visible_markers": 3, "chain_swap_min_consecutive_frames": 10},
    )
    assert res["quality_metrics"]["corrections_by_stage"].get("stage5_5_full_chain_swap_intervals", 0) == 0


def _write_ltoe_envelope_recovery_case(
    path: Path,
    *,
    include_star120: bool,
    bad_ltoe_frame: int | None = 4,
) -> None:
    """Leg + pelvis cluster; optional LTOE spike on ``bad_ltoe_frame``; *120 holds true toe (trim only)."""
    stems = ["LASI", "RASI", "LPSI", "RPSI", "LTHI", "LKNE", "LTOE", "LHEE", "OBSTACLE_L", "OBSTACLE_R"]
    if include_star120:
        stems = stems[:-2] + ["*120", "OBSTACLE_L", "OBSTACLE_R"]
    z0 = 1000.0
    base = {
        "LASI": np.array([0.0, 50.0, z0]),
        "RASI": np.array([0.0, -50.0, z0]),
        "LPSI": np.array([20.0, 0.0, z0]),
        "RPSI": np.array([-20.0, 0.0, z0]),
        "LTHI": np.array([10.0, 30.0, z0]),
        "LKNE": np.array([25.0, 35.0, z0]),
        "LTOE": np.array([30.0, 40.0, z0]),
        "LHEE": np.array([40.0, 38.0, z0]),
        "OBSTACLE_L": np.array([0.0, 500.0, 0.0]),
        "OBSTACLE_R": np.array([0.0, 600.0, 0.0]),
    }
    if include_star120:
        base["*120"] = np.array([30.0, 40.0, z0])
    lines = ["frame,time," + ",".join(f"{s}_x,{s}_y,{s}_z" for s in stems) + "\n"]
    for f in range(8):
        t = f * 0.01
        row = dict(base)
        if bad_ltoe_frame is not None and f == bad_ltoe_frame:
            row["LTOE"] = np.array([5000.0, 0.0, z0])
        parts = [str(f), f"{t:.4f}"]
        for s in stems:
            p = row[s]
            parts.extend([str(p[0]), str(p[1]), str(p[2])])
        lines.append(",".join(parts) + "\n")
    path.write_text("".join(lines))


def test_envelope_invalidation_priority1_recovers_from_unlabeled(tmp_path: Path) -> None:
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_ltoe_envelope_recovery_case(src, include_star120=False, bad_ltoe_frame=None)
    _write_ltoe_envelope_recovery_case(trim, include_star120=True, bad_ltoe_frame=4)
    bestframe_sidecar_path(trim).write_text("4")
    bestframe_sidecar_path(src).write_text("4")
    seg = {
        "pelvis": ["LASI", "RASI", "LPSI", "RPSI"],
        "left_leg": ["LTHI", "LKNE", "LTOE", "LHEE"],
    }
    mt = {s: 1 for s in ("LASI", "RASI", "LPSI", "RPSI", "LTHI", "LKNE", "LHEE", "OBSTACLE_L", "OBSTACLE_R")}
    mt["LTOE"] = 3
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        bilateral_chains={},
        marker_tiers=mt,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        config={
            "envelope_min_visible_markers": 3,
            # Centroid is computed before invalidation and includes the outlier; pick a
            # radius that still flags the bad LTOE but keeps *120 near the true toe inside.
            "envelope_radius_mm": 3000.0,
            "assignment_threshold_mm": 80.0,
        },
    )
    log = res["correction_log"]
    assert ((log["action"] == "unlabeled_replacement") & (log["priority"].astype(str) == "1")).any()
    from marker_label.trial_trim import parse_labeled_csv

    _, meta = parse_labeled_csv(out)
    pts = meta["points"]
    l2i = meta["label_to_marker_idx"]
    p = pts[4, l2i["LTOE"], :]
    assert np.isfinite(p).all()
    assert float(np.linalg.norm(p - np.array([30.0, 40.0, 1000.0]))) < 5.0
    s = res["quality_metrics"]["invalidation_recovery_summary"]
    assert s["recovered_via_unlabeled"] >= 1
    assert s["recovery_rate"] > 0.0


def test_outlier_invalidation_priority1_recovers_from_unlabeled(tmp_path: Path) -> None:
    stems = ["C7", "CLAV", "MID", "T10", "STRN", "*seed", "OBSTACLE_L", "OBSTACLE_R"]
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    z = 1000.0
    tpl = {
        "C7": np.array([0.0, 0.0, z]),
        "CLAV": np.array([120.0, 5.0, z + 35]),
        "MID": np.array([45.0, 95.0, z + 70]),
        "T10": np.array([80.0, 50.0, z + 130]),
        "STRN": np.array([90.0, 55.0, z + 135]),
        "OBSTACLE_L": np.array([0.0, 500.0, 0.0]),
        "OBSTACLE_R": np.array([0.0, 600.0, 0.0]),
    }

    def write_csv(path: Path, *, with_seed: bool, spike_strn: bool) -> None:
        st = [s for s in stems if with_seed or s != "*seed"]
        lines = ["frame,time," + ",".join(f"{s}_x,{s}_y,{s}_z" for s in st) + "\n"]
        for f in range(8):
            t = f * 0.01
            row = {k: v.copy() for k, v in tpl.items()}
            row["STRN"] = tpl["STRN"].copy()
            if spike_strn and f == 4:
                row["STRN"] = tpl["STRN"] + np.array([120.0, 0.0, 0.0])
            if with_seed and "*seed" in st:
                row["*seed"] = tpl["STRN"].copy()
            parts = [str(f), f"{t:.4f}"]
            for s in st:
                p = row[s]
                parts.extend([str(p[0]), str(p[1]), str(p[2])])
            lines.append(",".join(parts) + "\n")
        path.write_text("".join(lines))

    write_csv(src, with_seed=False, spike_strn=False)
    write_csv(trim, with_seed=True, spike_strn=True)
    bestframe_sidecar_path(trim).write_text("4")
    bestframe_sidecar_path(src).write_text("4")
    seg = {"trunk": ["C7", "CLAV", "MID", "T10", "STRN"]}
    cfg = {
        "envelope_min_visible_markers": 3,
        "tier_suspicious_threshold_mm": {1: 25, 2: 80.0, 3: 500.0},
        "suspicious_neighbor_window": 2,
    }
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        bilateral_chains={},
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        marker_tiers={"C7": 2, "CLAV": 2, "MID": 2, "T10": 2, "STRN": 2},
        config=cfg,
    )
    log = res["correction_log"]
    assert any(str(a) == "single_frame_outlier_rejection" for a in log["action"])
    assert ((log["action"] == "unlabeled_replacement") & (log["priority"].astype(str) == "1")).any()


def test_stage5_priority_tier2_wins_same_unlabeled_as_tier3(tmp_path: Path) -> None:
    stems = ["H", "P", "Q", "P2", "P3", "*U", "OBSTACLE_L", "OBSTACLE_R"]
    z = 1000.0
    tpl = {
        "H": np.array([0.0, 0.0, z]),
        "P": np.array([100.0, 0.0, z]),
        "Q": np.array([0.0, 100.0, z]),
        "P2": np.array([50.0, 5.0, z]),
        "P3": np.array([52.0, 8.0, z]),
        "OBSTACLE_L": np.array([0.0, 500.0, 0.0]),
        "OBSTACLE_R": np.array([0.0, 600.0, 0.0]),
        "*U": np.array([51.0, 6.0, z]),
    }

    def line_for(f: int) -> str:
        row = {k: v.copy() for k, v in tpl.items()}
        if f == 0:
            row["P2"] = np.array([4000.0, 0.0, z])
            row["P3"] = np.array([4000.0, 10.0, z])
        parts = [str(f), f"{f * 0.01:.4f}"]
        for s in stems:
            p = row[s]
            parts.extend([str(p[0]), str(p[1]), str(p[2])])
        return ",".join(parts) + "\n"

    hdr = "frame,time," + ",".join(f"{s}_x,{s}_y,{s}_z" for s in stems) + "\n"
    body = "".join(line_for(f) for f in range(8))
    text = hdr + body
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    src.write_text(text)
    trim.write_text(text)
    bestframe_sidecar_path(trim).write_text("4")
    bestframe_sidecar_path(src).write_text("4")
    seg = {"arm": ["H", "P", "Q", "P2", "P3"]}
    mt = {"H": 1, "P": 1, "Q": 1, "P2": 2, "P3": 3, "OBSTACLE_L": 1, "OBSTACLE_R": 1}
    res = correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        bilateral_chains={},
        marker_tiers=mt,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        config={"envelope_min_visible_markers": 3, "envelope_radius_mm": 2000.0},
    )
    log = res["correction_log"]
    rep = log[log["action"] == "unlabeled_replacement"]
    assert (rep["marker"] == "P2").any()
    assert not (rep["marker"] == "P3").any()
    nc = log[(log["action"] == "no_candidate") & (log["marker"] == "P3")]
    assert len(nc) >= 1


def test_stage5_priority1_before_priority2_for_unlabeled(tmp_path: Path) -> None:
    """Invalidated LTOE must consume *120 before occluded X can use it."""
    src = tmp_path / "orig.csv"
    trim = tmp_path / "trim.csv"
    out = tmp_path / "out.csv"
    _write_ltoe_envelope_recovery_case(src, include_star120=False, bad_ltoe_frame=None)
    stems_trim = [
        "LASI",
        "RASI",
        "LPSI",
        "RPSI",
        "LTHI",
        "LKNE",
        "LTOE",
        "LHEE",
        "X",
        "*120",
        "OBSTACLE_L",
        "OBSTACLE_R",
    ]
    z0 = 1000.0
    base = {
        "LASI": np.array([0.0, 50.0, z0]),
        "RASI": np.array([0.0, -50.0, z0]),
        "LPSI": np.array([20.0, 0.0, z0]),
        "RPSI": np.array([-20.0, 0.0, z0]),
        "LTHI": np.array([10.0, 30.0, z0]),
        "LKNE": np.array([25.0, 35.0, z0]),
        "LTOE": np.array([30.0, 40.0, z0]),
        "LHEE": np.array([40.0, 38.0, z0]),
        "OBSTACLE_L": np.array([0.0, 500.0, 0.0]),
        "OBSTACLE_R": np.array([0.0, 600.0, 0.0]),
        "*120": np.array([30.0, 40.0, z0]),
    }
    lines = ["frame,time," + ",".join(f"{s}_x,{s}_y,{s}_z" for s in stems_trim) + "\n"]
    for f in range(8):
        t = f * 0.01
        row = dict(base)
        if f == 4:
            row["LTOE"] = np.array([5000.0, 0.0, z0])
        parts = [str(f), f"{t:.4f}"]
        for s in stems_trim:
            if s == "X":
                parts.extend(["", "", ""])
                continue
            p = row[s]
            parts.extend([str(p[0]), str(p[1]), str(p[2])])
        lines.append(",".join(parts) + "\n")
    trim.write_text("".join(lines))
    bestframe_sidecar_path(trim).write_text("4")
    bestframe_sidecar_path(src).write_text("4")
    seg = {
        "pelvis": ["LASI", "RASI", "LPSI", "RPSI"],
        "left_leg": ["LTHI", "LKNE", "LTOE", "LHEE", "X"],
    }
    mt = {s: 1 for s in ("LASI", "RASI", "LPSI", "RPSI", "LTHI", "LKNE", "LHEE", "OBSTACLE_L", "OBSTACLE_R")}
    mt["LTOE"] = 3
    mt["X"] = 3
    correct_markers(
        trim,
        src,
        4,
        out,
        seg,
        bilateral_chains={},
        marker_tiers=mt,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        same_segment_swap_pairs=(),
        cross_segment_swap_pairs=(),
        config={
            "envelope_min_visible_markers": 3,
            "envelope_radius_mm": 3000.0,
            "assignment_threshold_mm": 80.0,
        },
    )
    from marker_label.trial_trim import parse_labeled_csv

    _, meta = parse_labeled_csv(out)
    pts = meta["points"]
    l2i = meta["label_to_marker_idx"]
    assert np.isfinite(pts[4, l2i["LTOE"], :]).all()
    assert not np.isfinite(pts[4, l2i["X"], :]).any()


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
            bilateral_chains={},
            config={"envelope_min_visible_markers": 3},
        )
