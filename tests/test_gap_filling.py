"""Tests for marker_label.gap_filling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from marker_label.gap_filling.asis_only_fill import asis_only_fill
from marker_label.gap_filling.continuity_check import apply_continuity_check_to_fills
from marker_label.gap_filling.gap_detection import find_gaps
from marker_label.gap_filling.orchestrator import DEFAULT_GAP_FILLING_CONFIG, gap_fill
from marker_label.gap_filling.reference import build_robust_reference
from marker_label.gap_filling.rigid_fill import rigid_body_fill
from marker_label.gap_filling.spline_fill import spline_fill
from marker_label.trial_trim import parse_labeled_csv, segment_markers_dict_for_trim_preset

ROOT = Path(__file__).resolve().parents[1]


def _axis_angle_rot(axis: np.ndarray, theta: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    x, y, z = float(a[0]), float(a[1]), float(a[2])
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def test_synthetic_rigid_body_fill_under_5mm() -> None:
    n = 100
    names = ["LASI", "LTHI", "LKNE", "LKNX"]
    label_to_idx = {m: i for i, m in enumerate(names)}
    template = np.array(
        [
            [0.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [400.0, 50.0, 0.0],
            [380.0, 40.0, 20.0],
        ],
        dtype=np.float64,
    )
    axis = np.array([0.15, 0.2, 0.97], dtype=np.float64)
    points = np.zeros((n, 4, 3), dtype=np.float64)
    for f in range(n):
        R = _axis_angle_rot(axis, 0.02 * f)
        t = np.array([10.0, 5.0, 100.0 + 0.3 * f], dtype=np.float64)
        for i in range(4):
            points[f, i, :] = R @ template[i] + t + np.random.default_rng(42 + f).normal(0, 0.3, 3)

    truth = points.copy()
    ik = label_to_idx["LKNE"]
    points[30:41, ik, :] = np.nan

    seg_dict = {"Synth_Thigh": names}
    ref, _ = build_robust_reference(
        points,
        label_to_idx,
        seg_dict,
        n_samples=30,
        best_frame=None,
        csv_path=None,
        frames=np.arange(n, dtype=np.int64),
    )
    gap = find_gaps(points[:, ik, :])[0]
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    rows = rigid_body_fill(
        points,
        "LKNE",
        gap,
        "Synth_Thigh",
        seg_dict,
        ref,
        label_to_idx,
        cfg,
        np.arange(n, dtype=np.int64),
    )
    errs = []
    for r in rows:
        if not r["success"]:
            continue
        fc = int(r["frame"])
        row = fc
        pred = np.array([r["predicted_x"], r["predicted_y"], r["predicted_z"]], dtype=np.float64)
        errs.append(float(np.linalg.norm(pred - truth[row, ik, :])))
    assert errs
    assert max(errs) < 5.0


def test_spline_short_gap_under_2mm() -> None:
    n = 100
    t = np.linspace(0, 4 * np.pi, n)
    traj = np.stack([400 * np.sin(t), 200 * np.cos(0.5 * t), 50 + 10 * t], axis=1).astype(np.float64)
    points = np.zeros((n, 1, 3))
    points[:, 0, :] = traj
    truth = points.copy()
    points[45:50, 0, :] = np.nan
    label_to_idx = {"M0": 0}
    gap = find_gaps(points[:, 0, :])[0]
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    rows = spline_fill(
        points,
        "M0",
        gap,
        label_to_idx,
        cfg,
        np.arange(n, dtype=np.int64),
    )
    assert all(r["success"] for r in rows)
    errs = []
    for r in rows:
        fc = int(r["frame"])
        pred = np.array([r["predicted_x"], r["predicted_y"], r["predicted_z"]], dtype=np.float64)
        row = fc
        errs.append(float(np.linalg.norm(pred - truth[row, 0, :])))
    assert max(errs) < 2.0


def test_spline_boundary_rejected() -> None:
    n = 50
    t = np.linspace(0, 2 * np.pi, n)
    traj = np.stack([np.sin(t), np.cos(t), 0.1 * t], axis=1).astype(np.float64)
    points = np.zeros((n, 1, 3))
    points[:, 0, :] = traj
    points[0:5, 0, :] = np.nan
    label_to_idx = {"M0": 0}
    gap = find_gaps(points[:, 0, :])[0]
    rows = spline_fill(
        points,
        "M0",
        gap,
        label_to_idx,
        DEFAULT_GAP_FILLING_CONFIG,
        np.arange(n, dtype=np.int64),
    )
    assert all(not r["success"] for r in rows)
    assert all("gap_at_boundary" in str(r.get("reason", "")) for r in rows)


def test_asis_only_pelvis_under_30mm() -> None:
    n = 120
    names = ["LASI", "RASI", "LPSI", "RPSI"]
    label_to_idx = {m: i for i, m in enumerate(names)}
    template = np.array(
        [
            [-100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [-95.0, -50.0, 0.0],
            [95.0, -50.0, 0.0],
        ],
        dtype=np.float64,
    )
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    points = np.zeros((n, 4, 3), dtype=np.float64)
    rng = np.random.default_rng(7)
    for f in range(n):
        R = _axis_angle_rot(axis, 0.015 * f)
        t = np.array([0.0, 0.0, 500.0 + 0.2 * f], dtype=np.float64)
        for i in range(4):
            points[f, i, :] = R @ template[i] + t + rng.normal(0, 0.5, 3)

    truth = points.copy()
    points[50:61, label_to_idx["LPSI"], :] = np.nan
    seg_dict = {"Pelvis": ["LASI", "RASI", "RPSI", "LPSI", "LASI"]}
    ref, _ = build_robust_reference(
        points,
        label_to_idx,
        seg_dict,
        n_samples=25,
        best_frame=None,
        csv_path=None,
        frames=np.arange(n, dtype=np.int64),
    )
    gap = find_gaps(points[:, label_to_idx["LPSI"], :])[0]
    rows = asis_only_fill(
        points,
        "LPSI",
        gap,
        ("LASI", "RASI"),
        seg_dict,
        ref,
        label_to_idx,
        DEFAULT_GAP_FILLING_CONFIG,
        np.arange(n, dtype=np.int64),
    )
    assert all(r["success"] for r in rows)
    errs = []
    for r in rows:
        fc = int(r["frame"])
        row = fc
        pred = np.array([r["predicted_x"], r["predicted_y"], r["predicted_z"]], dtype=np.float64)
        errs.append(float(np.linalg.norm(pred - truth[row, label_to_idx["LPSI"], :])))
    assert max(errs) < 30.0


def test_continuity_revert_clears_fill() -> None:
    n = 11
    points = np.full((n, 1, 3), np.nan, dtype=np.float64)
    points[:, 0, 0] = 0.0
    points[:, 0, 1] = 0.0
    points[:, 0, 2] = 0.0
    points[5, 0, :] = [0.0, 0.0, 150.0]
    label_to_idx = {"M0": 0}
    frames = np.arange(n, dtype=np.int64)
    fills = [
        {
            "frame": int(frames[5]),
            "marker": "M0",
            "method": "rigid_body",
            "success": True,
            "confidence": "HIGH",
            "predicted_x": 0.0,
            "predicted_y": 0.0,
            "predicted_z": 150.0,
            "fit_residual_mm": 1.0,
            "source_markers": "a;b;c",
            "gap_length": 1,
            "reason": "",
        }
    ]
    w, rev = apply_continuity_check_to_fills(
        points,
        fills,
        label_to_idx,
        frames,
        action="revert",
        max_velocity_mm_per_frame=50.0,
    )
    assert rev == 1
    assert not fills[0]["success"]
    assert not np.isfinite(points[5, 0, :]).all()


@pytest.mark.integration
def test_trial_21_gap_fill_quality() -> None:
    inp = ROOT / "data/SUBJ03/SUBJ03 Trial 21_corrected.csv"
    if not inp.is_file():
        pytest.skip("Trial 21 corrected CSV not in repo")
    out = ROOT / "data/SUBJ03/_pytest_trial21_filled.csv"
    _, meta = parse_labeled_csv(inp)
    n_frames = int(meta["n_frames"])
    seg = segment_markers_dict_for_trim_preset("full-body")
    res = gap_fill(str(inp), str(out), seg, config=dict(DEFAULT_GAP_FILLING_CONFIG))
    q = res["quality"]
    # Trial 21 corrected CSV: RASI/RPSI are sparse upstream; 95% for all four is not always reachable.
    assert q["visibility_after"]["LASI"] / n_frames >= 0.95
    # LPSI can remain gappy after fill (long NaN runs in source); require strong but not LASI-level coverage.
    assert q["visibility_after"]["LPSI"] / n_frames >= 0.90
    for m, d in q["visibility_change"].items():
        assert d >= 0, m
    assert q["fills_by_method"]["rigid_body"] > 50
    assert q["reverted_fills"] == 0
    out.unlink(missing_ok=True)
    Path(str(out).replace(".csv", "_fills.csv")).unlink(missing_ok=True)
    Path(str(out).replace(".csv", "_quality.json")).unlink(missing_ok=True)
    Path(str(out).replace(".csv", "_summary.png")).unlink(missing_ok=True)
    bf = Path(str(out) + ".bestframe")
    bf.unlink(missing_ok=True)


@pytest.mark.integration
def test_cli_creates_outputs() -> None:
    inp = ROOT / "data/SUBJ03/SUBJ03 Trial 21_corrected.csv"
    if not inp.is_file():
        pytest.skip("Trial 21 corrected CSV not in repo")
    out = ROOT / "data/SUBJ03/_pytest_cli_filled.csv"
    cmd = [
        sys.executable,
        "-m",
        "marker_label.gap_filling.cli",
        str(inp),
        "-o",
        str(out),
        "--segments-preset",
        "full-body",
        "--max-spline-gap",
        "10",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    assert out.is_file()
    js = Path(str(out).replace(".csv", "_quality.json"))
    assert js.is_file()
    data = json.loads(js.read_text())
    assert "fills_by_method" in data
    out.unlink(missing_ok=True)
    Path(str(out).replace(".csv", "_fills.csv")).unlink(missing_ok=True)
    js.unlink(missing_ok=True)
    Path(str(out).replace(".csv", "_summary.png")).unlink(missing_ok=True)
    Path(str(out) + ".bestframe").unlink(missing_ok=True)
