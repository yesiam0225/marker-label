"""Tests for trial_trim_combine (single-pass trim + two-pass merge)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from marker_label.trial_trim import bestframe_sidecar_path
from marker_label.trial_trim_combine import trim_and_combine


def _write_minimal_csv(path: Path, *, pass_offset: float = 0.0) -> None:
    """5 frames, markers A B C forming a rigid triangle + OBSTACLE markers for Y band."""
    lines = [
        "frame,time,A_x,A_y,A_z,B_x,B_y,B_z,C_x,C_y,C_z,OBSTACLE_L_x,OBSTACLE_L_y,OBSTACLE_L_z,OBSTACLE_R_x,OBSTACLE_R_y,OBSTACLE_R_z\n",
    ]
    for f in range(5):
        t = f * 0.01
        z = 1000.0 + pass_offset
        lines.append(
            f"{f},{t:.2f},0,0,{z},100,0,{z},0,100,{z},"
            f"0,500,0,0,600,0\n"
        )
    path.write_text("".join(lines))


def test_single_pass_trim_smoke(tmp_path: Path) -> None:
    src = tmp_path / "p.csv"
    out = tmp_path / "p_trim.csv"
    _write_minimal_csv(src)
    bestframe_sidecar_path(src).write_text("2")

    seg = {"tri": ["A", "B", "C"]}
    cfg = {
        "visibility_ratio_marker_subset": "segment-union",
        "min_trial_length": 1,
        "trim_qc_preset": None,
    }
    res = trim_and_combine([src], out, seg, obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"), config=cfg)

    assert res["trim_start"] <= 2 < res["trim_end"]
    assert res["best_frame_output"] == 2
    assert out.is_file()
    assert bestframe_sidecar_path(out).read_text().strip() == "2"
    q = json.loads(out.with_suffix(out.suffix + ".quality.json").read_text())
    assert q["mode"] == "single_pass"


def test_two_pass_overlap_switch(tmp_path: Path) -> None:
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    out = tmp_path / "out.csv"
    _write_minimal_csv(p1, pass_offset=0.0)
    _write_minimal_csv(p2, pass_offset=10.0)
    bestframe_sidecar_path(p1).write_text("1")
    bestframe_sidecar_path(p2).write_text("3")

    seg = {"tri": ["A", "B", "C"]}
    cfg = {
        "visibility_ratio_marker_subset": "segment-union",
        "min_trial_length": 1,
        "_skip_plot": True,
        "trim_qc_preset": None,
    }
    res = trim_and_combine([p1, p2], out, seg, obstacle_marker_pair=("OBSTACLE_L", "OBSTACLE_R"), config=cfg)

    assert res["switch_frame"] is not None
    assert res["trim_start"] <= res["switch_frame"] < res["trim_end"]
    log = out.with_suffix(out.suffix + ".trim_log.csv")
    assert log.is_file()


def test_invalid_csv_paths_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="length 1 or 2"):
        trim_and_combine([], tmp_path / "o.csv", {"tri": ["A", "B", "C"]})
