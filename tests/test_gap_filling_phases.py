"""Tests for gap-fill Phase A/B/C (static reference, two-marker rigid, shoulder)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marker_label.gap_filling.gap_detection import categorize_gap
from marker_label.gap_filling.orchestrator import DEFAULT_GAP_FILLING_CONFIG, gap_fill
from marker_label.gap_filling.rigid_fill import rigid_body_fill
from marker_label.gap_filling.static_reference import load_static_reference_bundle
from marker_label.gap_filling.segment_utils import min_visible_others_for_segment


def test_three_marker_segment_requires_two_visible_others() -> None:
    seg = ["RANK", "RHEE", "RTOE"]
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    assert min_visible_others_for_segment(seg, cfg) == 2


def test_foot_rhee_rigid_two_marker_with_static(tmp_path: Path) -> None:
    n = 20
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dyn.csv"
    static.write_text(
        "frame,time,RANK_x,RANK_y,RANK_z,RTOE_x,RTOE_y,RTOE_z,RHEE_x,RHEE_y,RHEE_z\n"
        "0,0,0,0,0,100,0,0,20,-50,0\n"
    )
    rows_dyn = ["frame,time,RANK_x,RANK_y,RANK_z,RTOE_x,RTOE_y,RTOE_z,RHEE_x,RHEE_y,RHEE_z\n"]
    for f in range(n):
        rows_dyn.append(f"{f},{f*0.01},10,0,{f},110,0,{f},,,,\n")
    dynamic.write_text("".join(rows_dyn))

    seg = {"R_Foot": ["RANK", "RHEE", "RTOE", "RANK"]}
    bundle = load_static_reference_bundle(static, seg, {"RANK": 0, "RTOE": 1, "RHEE": 2})
    pts = np.full((n, 3, 3), np.nan)
    for f in range(n):
        pts[f, 0, :] = [10, 0, float(f)]
        pts[f, 1, :] = [110, 0, float(f)]
    label_to_idx = {"RANK": 0, "RTOE": 1, "RHEE": 2}
    gap = (5, 9, 5)
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    cat, seg_n = categorize_gap("RHEE", gap, pts, label_to_idx, seg, ("LPSI", "RPSI"), ("LASI", "RASI"), cfg)
    assert cat == "rigid_body"
    rows = rigid_body_fill(
        pts,
        "RHEE",
        gap,
        seg_n or "R_Foot",
        seg,
        bundle["reference"],
        label_to_idx,
        cfg,
        np.arange(n, dtype=np.int64),
        two_marker_offsets=bundle["two_marker_offsets"],
    )
    assert any(r["success"] for r in rows)


def test_gap_fill_with_static_csv(tmp_path: Path) -> None:
    static = tmp_path / "s.csv"
    dynamic = tmp_path / "d.csv"
    out = tmp_path / "out.csv"
    static.write_text(
        "frame,time,RANK_x,RANK_y,RANK_z,RTOE_x,RTOE_y,RTOE_z,RHEE_x,RHEE_y,RHEE_z\n"
        "0,0,0,0,0,100,0,0,20,-50,0\n"
    )
    dynamic.write_text(
        "frame,time,RANK_x,RANK_y,RANK_z,RTOE_x,RTOE_y,RTOE_z,RHEE_x,RHEE_y,RHEE_z\n"
        "0,0,10,0,0,110,0,0,,,\n"
        "1,0.01,10,0,1,110,0,1,,,\n"
    )
    seg = {"R_Foot": ["RANK", "RHEE", "RTOE"]}
    res = gap_fill(str(dynamic), str(out), seg, static_csv_path=str(static), config=dict(DEFAULT_GAP_FILLING_CONFIG))
    assert res["quality"]["reference_source"] == "static"
    assert Path(out).is_file()
