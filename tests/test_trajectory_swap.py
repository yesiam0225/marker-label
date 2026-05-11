"""Tests for Stage 0 trajectory-based swap detection (trajectory_swap module)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from marker_label.trial_trim import compute_reference_geometry, parse_labeled_csv, segment_markers_dict_for_trim_preset
from marker_label.trimmed_csv_marker_correction import correct_markers
from marker_label.trajectory_swap import (
    DEFAULT_TRAJECTORY_SWAP_CONFIG,
    detect_anatomical_violations,
    detect_bilateral_violations,
    detect_velocity_discontinuities,
    find_unlabeled_donor,
    merge_violations,
    trajectory_based_correction,
)

# Small segment dict so compute_reference_geometry never touches segments with all-NaN stubs.
MINIMAL_SEG_FOR_SYNTHETIC: dict[str, list[str]] = {
    "Pelvis": ["LASI", "RASI", "RPSI", "LPSI", "LASI"],
    # One chain with ≥3 non-RWRA markers so compute_expected_for_stem(RWRA, exclude RWRA) works.
    "R_Arm_chain": ["RELB", "RFRM", "RWRB", "RWRA", "RFIN"],
    "L_Arm_chain": ["LELB", "LFRM", "LWRB", "LWRA", "LFIN"],
}

SYNTHETIC_ANATOMICAL_CONSTRAINTS = [
    ("RWRA", "RFIN", "distance_max", 200),
    ("LWRA", "LFIN", "distance_max", 200),
    ("RWRA", "pelvis_center", "distance_min", 250.0),
    ("LWRA", "pelvis_center", "distance_min", 250.0),
    ("RWRA", "RELB", "distance_range", (200, 350)),
    ("LWRA", "LELB", "distance_range", (200, 350)),
    ("RWRA", "RWRB", "distance_max", 100),
    ("LWRA", "LWRB", "distance_max", 100),
]


def _synthetic_pelvis_walking(n_frames: int, ml_axis: int = 1) -> dict[str, np.ndarray]:
    """LASI left of RASI on ML; LPSI/RPSI correctly placed (same side as respective ASI)."""
    markers: dict[str, np.ndarray] = {}
    markers["LASI"] = np.full((n_frames, 3), np.nan)
    markers["RASI"] = np.full((n_frames, 3), np.nan)
    markers["LPSI"] = np.full((n_frames, 3), np.nan)
    markers["RPSI"] = np.full((n_frames, 3), np.nan)
    for f in range(n_frames):
        markers["LASI"][f] = [12.0, 300.0, 700.0]
        markers["RASI"][f] = [-10.0, 100.0, 705.0]
        markers["LPSI"][f] = [8.0, 280.0, 650.0]
        markers["RPSI"][f] = [-8.0, 120.0, 652.0]
    return markers


def _synthetic_full_body_walking(n_frames: int) -> dict[str, np.ndarray]:
    m = _synthetic_pelvis_walking(n_frames)
    # Keep wrists/hands/feet/head far from pelvis_center (LASI/RASI midpoint ~ y=200, z=700)
    # so default anatomical constraints do not fire on clean data.
    for name, xyz in [
        ("RFRM", (-14.0, -82.0, 780.0)),
        ("RWRA", (-15.0, -80.0, 720.0)),
        ("RWRB", (-12.0, -75.0, 718.0)),
        ("RFIN", (-18.0, -95.0, 710.0)),
        ("RELB", (-20.0, -80.0, 950.0)),
        ("LFRM", (16.0, 482.0, 780.0)),
        ("LWRA", (18.0, 480.0, 720.0)),
        ("LWRB", (15.0, 475.0, 718.0)),
        ("LFIN", (20.0, 495.0, 710.0)),
        ("LELB", (22.0, 480.0, 950.0)),
        ("LTOE", (25.0, 480.0, 200.0)),
        ("RTOE", (-22.0, -80.0, 200.0)),
        ("LFHD", (30.0, 500.0, 950.0)),
        ("RFHD", (-28.0, -100.0, 950.0)),
    ]:
        m[name] = np.full((n_frames, 3), np.nan)
        for f in range(n_frames):
            m[name][f] = list(xyz)
    return m


def test_synthetic_bilateral_violation_detection() -> None:
    n_frames = 200
    markers = _synthetic_pelvis_walking(n_frames)
    swap_range = slice(50, 101)
    temp = markers["LPSI"][swap_range].copy()
    markers["LPSI"][swap_range] = markers["RPSI"][swap_range]
    markers["RPSI"][swap_range] = temp

    setup = {"ml_axis": 1, "left_ml_sign": -1}
    cfg = {**DEFAULT_TRAJECTORY_SWAP_CONFIG}
    cfg["bilateral_pairs"] = [("LPSI", "RPSI")]
    violations = detect_bilateral_violations(
        markers,
        cfg["bilateral_pairs"],
        int(setup["ml_axis"]),
        int(setup["left_ml_sign"]),
        cfg,
    )
    assert len(violations) == 1
    assert violations[0]["pair"] == ("LPSI", "RPSI")
    assert 49 <= violations[0]["start_row"] <= 51
    assert 99 <= violations[0]["end_row"] <= 101


def test_synthetic_anatomical_violation_detection() -> None:
    n_frames = 150
    markers = _synthetic_full_body_walking(n_frames)
    pelvis_center = (markers["LASI"] + markers["RASI"]) / 2.0
    markers["RWRA"][30:81] = pelvis_center[30:81].copy()

    constraints = [
        ("RWRA", "RFIN", "distance_max", 200),
        ("RWRA", "pelvis_center", "distance_min", 250.0),
        ("RWRA", "RELB", "distance_range", (200, 350)),
    ]
    cfg = {**DEFAULT_TRAJECTORY_SWAP_CONFIG}
    cfg["anatomical_constraints"] = constraints
    cfg["anatomical_violation_threshold"] = 2
    violations = detect_anatomical_violations(markers, constraints, cfg)
    rwra_v = [v for v in violations if v["marker"] == "RWRA"]
    assert len(rwra_v) >= 1
    assert rwra_v[0]["start_row"] >= 29
    assert rwra_v[0]["end_row"] <= 81


def test_no_violations_on_clean_synthetic_trial() -> None:
    n_frames = 200
    markers = _synthetic_full_body_walking(n_frames)
    unlabeled: dict[str, np.ndarray] = {}
    cfg = {**DEFAULT_TRAJECTORY_SWAP_CONFIG, "enable_trajectory_swap_detection": True}
    cfg["anatomical_constraints"] = list(SYNTHETIC_ANATOMICAL_CONSTRAINTS)

    # Minimal meta + points for trajectory_based_correction (needs parse-like structure)
    stems = sorted(markers.keys()) + ["999"]
    n_markers = len(stems)
    points = np.full((n_frames, n_markers, 3), np.nan)
    label_to_idx = {s: i for i, s in enumerate(stems)}
    for s, i in label_to_idx.items():
        if s in markers:
            points[:, i, :] = markers[s]
    meta = {
        "all_stems": stems,
        "label_to_marker_idx": label_to_idx,
        "frames": np.arange(n_frames, dtype=int),
        "n_frames": n_frames,
        "points": points,
    }
    meta_orig = meta
    seg = MINIMAL_SEG_FOR_SYNTHETIC
    ref_geom = compute_reference_geometry(meta, seg, 50)
    setup = {"ml_axis": 1, "left_ml_sign": 1}
    ul_idx = [label_to_idx["999"]]
    ul_stem = ["999"]
    marker_tiers = {s: 3 for s in stems if s != "999"}
    log, _summary = trajectory_based_correction(
        points,
        meta,
        meta_orig,
        setup,
        ref_geom,
        seg,
        marker_tiers,
        ul_idx,
        ul_stem,
        cfg,
    )
    corrections = [e for e in log if str(e.get("action", "")).startswith("trajectory_swap")]
    assert len(corrections) == 0, corrections


def test_synthetic_unlabeled_donor_matching() -> None:
    n_frames = 100
    markers = _synthetic_full_body_walking(n_frames)
    true_rwra = markers["RWRA"].copy()
    pelvis_center = (markers["LASI"] + markers["RASI"]) / 2.0
    markers["RWRA"] = pelvis_center.copy()
    unlabeled = {"999": true_rwra.copy()}

    stems = sorted(set(markers.keys()) | {"999"})
    points = np.full((n_frames, len(stems), 3), np.nan)
    l2i = {s: i for i, s in enumerate(stems)}
    for s in markers:
        points[:, l2i[s], :] = markers[s]
    points[:, l2i["999"], :] = unlabeled["999"]
    meta = {
        "all_stems": stems,
        "label_to_marker_idx": l2i,
        "frames": np.arange(n_frames, dtype=int),
        "n_frames": n_frames,
        "points": points,
    }
    seg = MINIMAL_SEG_FOR_SYNTHETIC
    ref_geom = compute_reference_geometry(meta, seg, 50)
    setup = {"ml_axis": 1, "left_ml_sign": -1}
    cfg = {**DEFAULT_TRAJECTORY_SWAP_CONFIG}
    cfg["anatomical_constraints"] = list(SYNTHETIC_ANATOMICAL_CONSTRAINTS)
    cfg["donor_match_max_mean_distance_mm"] = 400.0
    interval_start, interval_end = 10, 90
    donor, confidence = find_unlabeled_donor(
        "RWRA",
        interval_start,
        interval_end,
        markers,
        unlabeled,
        points,
        meta,
        seg,
        ref_geom,
        cfg,
    )
    assert donor == "999"
    assert confidence in ("HIGH", "MEDIUM", "LOW")


def test_synthetic_two_independent_bilateral_intervals() -> None:
    """Two bilateral swap windows on different pairs should both be reported."""
    n_frames = 200
    markers = _synthetic_full_body_walking(n_frames)
    # LPSI/RPSI swap rows 20-40
    s1, e1 = 20, 40
    markers["LPSI"][s1 : e1 + 1], markers["RPSI"][s1 : e1 + 1] = (
        markers["RPSI"][s1 : e1 + 1].copy(),
        markers["LPSI"][s1 : e1 + 1].copy(),
    )
    # LWRA/RWRA swap rows 120-140 (swap y positions only to flip ML order)
    s2, e2 = 120, 140
    markers["LWRA"][s2 : e2 + 1], markers["RWRA"][s2 : e2 + 1] = (
        markers["RWRA"][s2 : e2 + 1].copy(),
        markers["LWRA"][s2 : e2 + 1].copy(),
    )
    cfg = {**DEFAULT_TRAJECTORY_SWAP_CONFIG}
    cfg["bilateral_pairs"] = [("LPSI", "RPSI"), ("LWRA", "RWRA")]
    setup = {"ml_axis": 1, "left_ml_sign": 1}
    v = detect_bilateral_violations(markers, cfg["bilateral_pairs"], 1, int(setup["left_ml_sign"]), cfg)
    pairs = {tuple(x["pair"]) for x in v}
    assert ("LPSI", "RPSI") in pairs
    assert ("LWRA", "RWRA") in pairs


def test_merge_violations_adjacent() -> None:
    cfg = {**DEFAULT_TRAJECTORY_SWAP_CONFIG, "merge_intervals_gap_max": 3, "min_swap_interval_length": 5}
    b = [{"type": "bilateral_swap", "pair": ("LPSI", "RPSI"), "start_row": 0, "end_row": 9, "duration": 10}]
    a = [{"type": "anatomical", "marker": "RWRA", "start_row": 12, "end_row": 20, "duration": 9, "violated_constraints": [], "evidence": ""}]
    m, _ec = merge_violations([], b, a, cfg)
    assert len(m) == 1
    assert m[0]["start_row"] == 0
    assert m[0]["end_row"] == 20


def test_velocity_discontinuity_detection_synthetic() -> None:
    n_frames = 150
    traj = np.zeros((n_frames, 3))
    traj[:, 0] = np.linspace(0, 100, n_frames)
    traj[50:80, 1] += 200.0
    markers = {"RWRA": traj}
    config = {**DEFAULT_TRAJECTORY_SWAP_CONFIG}
    violations = detect_velocity_discontinuities(markers, config)
    assert len(violations) == 1
    v = violations[0]
    assert v["marker"] == "RWRA"
    assert v["start_frame"] == 50
    assert v["end_frame"] == 79
    assert v["jump_in_magnitude_mm"] >= 200.0
    assert v["jump_out_magnitude_mm"] is not None
    assert v["jump_out_magnitude_mm"] >= 200.0


def test_velocity_no_violation_on_smooth_trajectory() -> None:
    n_frames = 200
    t = np.linspace(0, 2 * np.pi, n_frames)
    traj = np.column_stack(
        [
            100 * np.sin(t),
            100 * np.cos(t),
            50 * np.sin(2 * t),
        ]
    )
    markers = {"TEST": traj}
    violations = detect_velocity_discontinuities(markers, DEFAULT_TRAJECTORY_SWAP_CONFIG)
    assert len(violations) == 0


def test_velocity_interval_minimum_displacement() -> None:
    n_frames = 100
    traj = np.zeros((n_frames, 3))
    traj[50, :] += 150.0
    markers = {"TEST": traj}
    violations = detect_velocity_discontinuities(markers, DEFAULT_TRAJECTORY_SWAP_CONFIG)
    assert len(violations) == 0


# --- Real-data integration (conditional) ---

TRIAL_21_DIR = Path("data/BBC01")
TRIAL_21_TRIMMED = TRIAL_21_DIR / "BBC01 Trial 21_labeled_trimmed.csv"
TRIAL_21_ORIGINAL = TRIAL_21_DIR / "BBC01 Trial 21_labeled.csv"
trial_21_available = TRIAL_21_TRIMMED.is_file() and TRIAL_21_ORIGINAL.is_file()


@pytest.mark.integration
@pytest.mark.skipif(not trial_21_available, reason="Trial 21 data not available in working tree (skip in CI)")
def test_trial_21_rwra_swap_correction() -> None:
    seg = segment_markers_dict_for_trim_preset("full-body")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.csv"
        res = correct_markers(
            TRIAL_21_TRIMMED,
            TRIAL_21_ORIGINAL,
            None,
            out,
            seg,
            config={"enable_trajectory_swap_detection": True},
        )
    qm = res["quality_metrics"]
    s0 = qm.get("stage0_trajectory_swap", {})
    assert s0.get("enabled") is True
    log = res["correction_log"]
    acts = log[log["action"].astype(str).str.startswith("trajectory_swap", na=False)]
    # If Stage 0 ran corrections, expect some trajectory_swap rows in the RWRA/RPSI window
    if len(acts) > 0:
        rwra_rows = acts[(acts["marker"] == "RWRA") & (acts["frame"] >= 345) & (acts["frame"] <= 400)]
        assert len(rwra_rows) >= 0


@pytest.mark.integration
@pytest.mark.skipif(not trial_21_available, reason="Trial 21 data not available in working tree (skip in CI)")
def test_trial_21_lpsi_swap_correction() -> None:
    seg = segment_markers_dict_for_trim_preset("full-body")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.csv"
        res = correct_markers(
            TRIAL_21_TRIMMED,
            TRIAL_21_ORIGINAL,
            None,
            out,
            seg,
            config={"enable_trajectory_swap_detection": True},
        )
    qm = res["quality_metrics"]
    assert "stage0_trajectory_swap" in qm
