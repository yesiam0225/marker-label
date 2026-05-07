"""Tests for trial_trim_combine (single-pass trim + two-pass merge)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from marker_label.trial_trim import bestframe_sidecar_path, compute_reference_geometry, parse_labeled_csv
from marker_label.trial_trim_combine import compute_diagnostics_with_envelope, trim_and_combine


def _write_minimal_csv(path: Path, *, pass_offset: float = 0.0, n_frames: int = 60) -> None:
    """Synthetic walk with pelvis/toe trajectories, plus simple rigid triangle and obstacle pair."""
    lines = [
        (
            "frame,time,"
            "A_x,A_y,A_z,B_x,B_y,B_z,C_x,C_y,C_z,"
            "LASI_x,LASI_y,LASI_z,RASI_x,RASI_y,RASI_z,"
            "LTOE_x,LTOE_y,LTOE_z,RTOE_x,RTOE_y,RTOE_z,"
            "OBSTACLE_L_x,OBSTACLE_L_y,OBSTACLE_L_z,OBSTACLE_R_x,OBSTACLE_R_y,OBSTACLE_R_z\n"
        ),
    ]
    for f in range(n_frames):
        t = f * 0.01
        z = 1000.0 + pass_offset
        # Forward progression mostly on X axis.
        prog = -150.0 + 5.0 * f
        lasi_x, rasi_x = prog, prog
        lasi_y, rasi_y = 50.0, -50.0
        ltoe_x = prog + 30.0
        rtoe_x = prog + 10.0
        lines.append(
            f"{f},{t:.2f},"
            f"0,0,{z},100,0,{z},0,100,{z},"
            f"{lasi_x},{lasi_y},{z},{rasi_x},{rasi_y},{z},"
            f"{ltoe_x},{lasi_y},{z},{rtoe_x},{rasi_y},{z},"
            f"0,-50,0,0,150,0\n"
        )
    path.write_text("".join(lines))


def test_single_pass_trim_smoke(tmp_path: Path) -> None:
    src = tmp_path / "p.csv"
    out = tmp_path / "p_trim.csv"
    _write_minimal_csv(src)
    bestframe_sidecar_path(src).write_text("20")

    seg = {"tri": ["A", "B", "C"]}
    cfg = {
        "visibility_ratio_marker_subset": "segment-union",
        "min_trial_length": 1,
        "trim_qc_preset": None,
    }
    res = trim_and_combine([src], out, seg, obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"), config=cfg)

    assert res["trim_start"] <= 20 < res["trim_end"]
    assert res["best_frame_output"] == 20
    assert out.is_file()
    assert bestframe_sidecar_path(out).read_text().strip() == "20"
    q = json.loads(out.with_suffix(out.suffix + ".quality.json").read_text())
    assert q["mode"] == "single_pass"
    assert "indeterminate_segments_per_pass" in q


def test_indeterminate_segment_is_not_frame_bad(tmp_path: Path) -> None:
    p = tmp_path / "indeterminate.csv"
    p.write_text(
        "frame,time,"
        "LASI_x,LASI_y,LASI_z,LTHI_x,LTHI_y,LTHI_z,LKNE_x,LKNE_y,LKNE_z\n"
        "0,0.00,0,0,0,100,0,0,200,40,0\n"
        "1,0.01,nan,nan,nan,100,0,0,200,40,0\n",
        encoding="utf-8",
    )
    _, meta = parse_labeled_csv(p)
    seg = {"L_Thigh": ["LASI", "LTHI", "LKNE"]}
    ref = compute_reference_geometry(meta, seg, best_idx=0, collinearity_eps_mm=1e-9)
    env = np.zeros((int(meta["n_frames"]), len(meta["all_stems"])), dtype=bool)
    diag = compute_diagnostics_with_envelope(
        meta,
        ref,
        seg,
        [meta["label_to_marker_idx"]["LASI"], meta["label_to_marker_idx"]["LTHI"], meta["label_to_marker_idx"]["LKNE"]],
        list(meta["all_stems"]),
        env,
        residual_threshold_mm=15.0,
        residual_threshold_thigh_mm=25.0,
        thigh_segment_keywords=("thigh",),
        visible_ratio_threshold=0.6,
        envelope_outlier_threshold=999,
    )
    assert diag[1]["segment_state"]["L_Thigh"] == "indeterminate"
    assert diag[1]["indeterminate_segment_count"] == 1
    assert diag[1]["bad_residual"] is False
    assert diag[1]["is_bad"] is False


def test_two_pass_overlap_switch(tmp_path: Path) -> None:
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    out = tmp_path / "out.csv"
    _write_minimal_csv(p1, pass_offset=0.0)
    _write_minimal_csv(p2, pass_offset=10.0)
    bestframe_sidecar_path(p1).write_text("15")
    bestframe_sidecar_path(p2).write_text("45")

    seg = {"tri": ["A", "B", "C"]}
    cfg = {
        "visibility_ratio_marker_subset": "segment-union",
        "min_trial_length": 1,
        "_skip_plot": True,
        "trim_qc_preset": None,
    }
    res = trim_and_combine([p1, p2], out, seg, obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"), config=cfg)

    assert res["switch_frame"] is None
    assert res["trim_start"] <= res["best_frame_output"] < res["trim_end"]
    q = res["quality_metrics"]
    assert q.get("crossing_detection") is not None
    assert q.get("pass_selection_summary") is not None
    log = out.with_suffix(out.suffix + ".trim_log.csv")
    assert log.is_file()


def test_invalid_csv_paths_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="length 1 or 2"):
        trim_and_combine([], tmp_path / "o.csv", {"tri": ["A", "B", "C"]})


def test_two_pass_same_side_best_frames_warns_fallback(tmp_path: Path) -> None:
    p1 = tmp_path / "a_same.csv"
    p2 = tmp_path / "b_same.csv"
    out = tmp_path / "out_same.csv"
    _write_minimal_csv(p1, pass_offset=0.0)
    _write_minimal_csv(p2, pass_offset=20.0)
    bestframe_sidecar_path(p1).write_text("5")
    bestframe_sidecar_path(p2).write_text("8")
    seg = {"tri": ["A", "B", "C"]}
    cfg = {
        "visibility_ratio_marker_subset": "segment-union",
        "trim_qc_preset": None,
        "_skip_plot": True,
    }
    res = trim_and_combine(
        [p1, p2],
        out,
        seg,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        config=cfg,
    )
    q = res["quality_metrics"]
    warns = q.get("warnings", [])
    assert any("Both best frames on" in str(w) for w in warns)
    rel = q.get("pass_position_relative_to_crossing")
    assert rel is not None and rel["pass1"] == rel["pass2"]


def test_obstacle_prefilter_requires_best_frame_pair(tmp_path: Path) -> None:
    src = tmp_path / "p.csv"
    out = tmp_path / "o.csv"
    _write_minimal_csv(src)
    bestframe_sidecar_path(src).write_text("2")
    lines = src.read_text().splitlines()
    header = lines[0].split(",")
    i_l = header.index("OBSTACLE_L_y")
    i_r = header.index("OBSTACLE_R_y")
    row2 = lines[3].split(",")  # frame=2 row
    row2[i_l] = ""
    row2[i_r] = ""
    lines[3] = ",".join(row2)
    src.write_text("\n".join(lines) + "\n")

    seg = {"tri": ["A", "B", "C"]}
    with pytest.raises(ValueError, match="Best frame must contain finite obstacle pair coordinates"):
        trim_and_combine(
            [src],
            out,
            seg,
            obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
            config={"trim_qc_preset": None},
        )


def test_point_y_screening_sets_outside_point_nan(tmp_path: Path) -> None:
    src = tmp_path / "s.csv"
    out = tmp_path / "s_out.csv"
    # Marker C has Y far outside obstacle band at frame 1 only.
    lines = [
        "frame,time,A_x,A_y,A_z,B_x,B_y,B_z,C_x,C_y,C_z,OBSTACLE_L_x,OBSTACLE_L_y,OBSTACLE_L_z,OBSTACLE_R_x,OBSTACLE_R_y,OBSTACLE_R_z\n",
        "0,0.00,0,0,1000,100,0,1000,0,50,1000,0,-50,0,0,150,0\n",
        "1,0.01,0,0,1000,100,0,1000,0,999,1000,0,-50,0,0,150,0\n",
        "2,0.02,0,0,1000,100,0,1000,0,50,1000,0,-50,0,0,150,0\n",
        "3,0.03,0,0,1000,100,0,1000,0,50,1000,0,-50,0,0,150,0\n",
        "4,0.04,0,0,1000,100,0,1000,0,50,1000,0,-50,0,0,150,0\n",
    ]
    src.write_text("".join(lines))
    bestframe_sidecar_path(src).write_text("2")

    seg = {"tri": ["A", "B", "C"]}
    res = trim_and_combine(
        [src],
        out,
        seg,
        obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"),
        config={
            "visibility_ratio_marker_subset": "segment-union",
            "trim_qc_preset": None,
            "point_y_screening_margin_mm": 0.0,
            "min_trial_length": 1,
        },
    )
    assert res["trim_start"] <= 2 < res["trim_end"]
    pre = res["quality_metrics"]["obstacle_prefilter"][0]
    assert int(pre["point_y_screened_xyz_rows"]) >= 1
    _, meta = parse_labeled_csv(out)
    c_idx = meta["label_to_marker_idx"]["C"]
    frames = meta["frames"]
    hits = np.flatnonzero(frames == 1)
    if hits.size > 0:
        assert np.isnan(meta["points"][int(hits[0]), c_idx, :]).all()
