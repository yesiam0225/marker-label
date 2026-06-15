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
        "frame,time,RANK_x,RANK_y,RANK_z,RTOE_x,RTOE_y,RTOE_z,RHEE_x,RHEE_y,RHEE_z,RTIB_x,RTIB_y,RTIB_z\n"
        "0,0,0,0,0,100,0,0,20,-50,0,30,80,0\n"
    )
    rows_dyn = ["frame,time,RANK_x,RANK_y,RANK_z,RTOE_x,RTOE_y,RTOE_z,RHEE_x,RHEE_y,RHEE_z,RTIB_x,RTIB_y,RTIB_z\n"]
    for f in range(n):
        rows_dyn.append(f"{f},{f*0.01},10,0,{f},110,0,{f},,,,40,80,{f}\n")
    dynamic.write_text("".join(rows_dyn))

    seg = {"R_Foot": ["RANK", "RHEE", "RTOE", "RANK"]}
    bundle = load_static_reference_bundle(static, seg, {"RANK": 0, "RTOE": 1, "RHEE": 2})
    pts = np.full((n, 4, 3), np.nan)
    for f in range(n):
        pts[f, 0, :] = [10, 0, float(f)]
        pts[f, 1, :] = [110, 0, float(f)]
        pts[f, 3, :] = [40, 80, float(f)]
    label_to_idx = {"RANK": 0, "RTOE": 1, "RHEE": 2, "RTIB": 3}
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


def test_thigh_lthi_rigid_two_marker_uses_body_frame(tmp_path: Path) -> None:
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    static.write_text(
        "frame,time,LASI_x,LASI_y,LASI_z,LTHI_x,LTHI_y,LTHI_z,LKNE_x,LKNE_y,LKNE_z\n"
        "0,0,0,400,900,200,350,650,250,380,500\n"
    )
    rows = [
        "frame,time,LASI_x,LASI_y,LASI_z,LTHI_x,LTHI_y,LTHI_z,LKNE_x,LKNE_y,LKNE_z\n"
    ]
    for f in range(5):
        z = float(f)
        rows.append(
            f"{f},{f*0.01},"
            f"{z},{400+z},{900+z},nan,nan,nan,"
            f"{250+z},{380+z},{500+z}\n"
        )
    dynamic.write_text("".join(rows))

    seg = {"L_Thigh": ["LASI", "LTHI", "LKNE"]}
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    cfg["synthesize_missing_foot_heels"] = False
    cfg["synthesize_missing_hand_markers"] = False
    cfg["synthesize_hand_from_contralateral"] = False
    gap_fill(str(dynamic), str(out), seg, static_csv_path=str(static), config=cfg)

    import pandas as pd

    df = pd.read_csv(out)
    row = df.iloc[0]
    assert row["LTHI_z"] > row["LKNE_z"]
    assert row["LTHI_z"] < row["LASI_z"]
    assert abs(float(row["LTHI_y"]) - 350.0) < 80.0


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


def test_gap_fill_synthesizes_missing_lhee_column(tmp_path: Path) -> None:
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    static.write_text(
        "frame,time,LANK_x,LANK_y,LANK_z,LTOE_x,LTOE_y,LTOE_z,LHEE_x,LHEE_y,LHEE_z,LTIB_x,LTIB_y,LTIB_z\n"
        "0,0,0,0,0,100,0,0,20,-50,0,30,80,0\n"
    )
    rows = ["frame,time,LANK_x,LANK_y,LANK_z,LTOE_x,LTOE_y,LTOE_z,LTIB_x,LTIB_y,LTIB_z\n"]
    for f in range(5):
        rows.append(f"{f},{f*0.01},10,0,{f},110,0,{f},40,80,{f}\n")
    dynamic.write_text("".join(rows))

    seg = {"L_Foot": ["LANK", "LHEE", "LTOE", "LANK"]}
    gap_fill(str(dynamic), str(out), seg, static_csv_path=str(static), config=dict(DEFAULT_GAP_FILLING_CONFIG))

    import pandas as pd

    df = pd.read_csv(out)
    assert "LHEE_x" in df.columns
    assert df["LHEE_x"].notna().all()


def test_gap_fill_synthesizes_missing_lwrb_from_forearm_frame(tmp_path: Path) -> None:
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    static.write_text(
        "frame,time,LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LWRB_x,LWRB_y,LWRB_z,LFIN_x,LFIN_y,LFIN_z\n"
        "0,0,0,0,0,50,0,0,60,10,0,70,20,0\n"
    )
    rows = [
        "frame,time,LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LFIN_x,LFIN_y,LFIN_z\n"
    ]
    for f in range(5):
        rows.append(f"{f},{f*0.01},0,0,{float(f)},50,0,{float(f)},70,20,{float(f)}\n")
    dynamic.write_text("".join(rows))

    seg = {
        "L_Hand": ["LWRA", "LWRB", "LFIN"],
        "L_Forearm": ["LELB", "LFRM", "LWRA", "LWRB"],
    }
    gap_fill(str(dynamic), str(out), seg, static_csv_path=str(static), config=dict(DEFAULT_GAP_FILLING_CONFIG))

    import pandas as pd

    df = pd.read_csv(out)
    assert "LWRB_x" in df.columns
    assert df["LWRB_x"].notna().all()
    assert abs(float(df["LWRB_x"].iloc[0]) - 60.0) < 1.0


def test_gap_fill_synthesizes_missing_lfin_from_forearm_frame(tmp_path: Path) -> None:
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    static.write_text(
        "frame,time,LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LWRB_x,LWRB_y,LWRB_z,LFIN_x,LFIN_y,LFIN_z\n"
        "0,0,0,0,0,50,0,0,60,10,0,70,20,0\n"
    )
    rows = [
        "frame,time,LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LWRB_x,LWRB_y,LWRB_z\n"
    ]
    for f in range(5):
        rows.append(
            f"{f},{f*0.01},0,0,{float(f)},50,0,{float(f)},60,10,{float(f)}\n"
        )
    dynamic.write_text("".join(rows))

    seg = {"L_Hand": ["LWRA", "LWRB", "LFIN"]}
    gap_fill(str(dynamic), str(out), seg, static_csv_path=str(static), config=dict(DEFAULT_GAP_FILLING_CONFIG))

    import pandas as pd

    df = pd.read_csv(out)
    assert "LFIN_x" in df.columns
    assert df["LFIN_x"].notna().all()
    assert abs(float(df["LFIN_x"].iloc[0]) - 70.0) < 1.0


def test_gap_fill_synthesizes_lwrb_from_contralateral_hand(tmp_path: Path) -> None:
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    rows = [
        "frame,time,"
        "LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LFIN_x,LFIN_y,LFIN_z,"
        "RFRM_x,RFRM_y,RFRM_z,RWRA_x,RWRA_y,RWRA_z,RWRB_x,RWRB_y,RWRB_z,RFIN_x,RFIN_y,RFIN_z\n"
    ]
    for f in range(5):
        z = float(f)
        rows.append(
            f"{f},{f*0.01},"
            f"0,0,{z},-10,0,{z},-10,10,{z},"
            f"0,0,{z},10,0,{z},12,0,{z - 8.0},10,10,{z}\n"
        )
    dynamic.write_text("".join(rows))

    seg = {"L_Hand": ["LWRA", "LWRB", "LFIN"], "R_Hand": ["RWRA", "RWRB", "RFIN"]}
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    cfg["synthesize_missing_hand_markers"] = False
    cfg["synthesize_hand_from_contralateral"] = True
    cfg["synthesize_missing_foot_heels"] = False
    gap_fill(str(dynamic), str(out), seg, static_csv_path=None, config=cfg)

    import pandas as pd

    df = pd.read_csv(out)
    assert "LWRB_x" in df.columns
    assert df["LWRB_x"].notna().all()
    assert abs(float(df["LWRB_x"].iloc[0]) - (-12.0)) < 1.0
    assert float(df["LWRB_z"].iloc[0]) < float(df["LWRA_z"].iloc[0])


def test_contralateral_lwrb_posterior_with_pelvis_ap(tmp_path: Path) -> None:
    """Regression: sagittal lock must not flip a correct z-mirrored WRB (BBA09-like)."""
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    rows = [
        "frame,time,"
        "LASI_x,LASI_y,LASI_z,RASI_x,RASI_y,RASI_z,"
        "LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LFIN_x,LFIN_y,LFIN_z,"
        "RFRM_x,RFRM_y,RFRM_z,RWRA_x,RWRA_y,RWRA_z,RWRB_x,RWRB_y,RWRB_z,RFIN_x,RFIN_y,RFIN_z\n"
    ]
    rows.append(
        "160,1.6,"
        "-1155.574,447.818,957.89,-1150.594,186.603,955.693,"
        "-1135.8,565.8,964,-1093.2,508.7,849,-1078.1,556.5,783.3,"
        "-1235,34.6,985.5,-1167,88.4,825.4,-1230.2,48.6,816.2,-1174.7,41.3,761\n"
    )
    dynamic.write_text("".join(rows))
    seg = {"L_Hand": ["LWRA", "LWRB", "LFIN"], "R_Hand": ["RWRA", "RWRB", "RFIN"]}
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    cfg["synthesize_missing_hand_markers"] = False
    cfg["synthesize_hand_from_contralateral"] = True
    cfg["synthesize_missing_foot_heels"] = False
    gap_fill(str(dynamic), str(out), seg, static_csv_path=None, config=cfg)

    import numpy as np
    import pandas as pd

    df = pd.read_csv(out)
    row = df.iloc[0]
    ml = np.array([row.RASI_x - row.LASI_x, row.RASI_y - row.LASI_y, row.RASI_z - row.LASI_z])
    ml /= np.linalg.norm(ml)
    ap = np.cross(ml, np.array([0.0, 1.0, 0.0]))
    ap /= np.linalg.norm(ap)
    lwra = np.array([row.LWRA_x, row.LWRA_y, row.LWRA_z])
    lwrb = np.array([row.LWRB_x, row.LWRB_y, row.LWRB_z])
    rwra = np.array([row.RWRA_x, row.RWRA_y, row.RWRA_z])
    rwrb = np.array([row.RWRB_x, row.RWRB_y, row.RWRB_z])
    assert float(np.dot(lwrb - lwra, ap)) < 0.0
    assert float(np.dot(rwrb - rwra, ap)) < 0.0


def test_contralateral_skipped_when_static_has_side_hand(tmp_path: Path) -> None:
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    static.write_text(
        "frame,time,LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LWRB_x,LWRB_y,LWRB_z,LFIN_x,LFIN_y,LFIN_z\n"
        "0,0,0,0,0,10,0,0,12,0,-8,10,10,0\n"
    )
    dynamic.write_text(
        "frame,time,LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LFIN_x,LFIN_y,LFIN_z\n"
        "0,0,0,0,0,10,0,0,10,10,0\n"
    )
    seg = {"L_Hand": ["LWRA", "LWRB", "LFIN"]}
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    cfg["synthesize_missing_foot_heels"] = False
    gap_fill(
        str(dynamic),
        str(out),
        seg,
        static_csv_path=str(static),
        config=cfg,
    )

    import pandas as pd

    fills = pd.read_csv(str(out).replace(".csv", "_fills.csv"))
    assert (fills["marker"] == "LWRB").any()
    assert (fills.loc[fills["marker"] == "LWRB", "method"] == "static_hand_forearm").all()
    assert not (fills["method"] == "contralateral_hand").any()


def test_contralateral_used_when_static_lacks_side_hand(tmp_path: Path) -> None:
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    static.write_text(
        "frame,time,RFRM_x,RFRM_y,RFRM_z,RWRA_x,RWRA_y,RWRA_z,RWRB_x,RWRB_y,RWRB_z,RFIN_x,RFIN_y,RFIN_z\n"
        "0,0,0,0,0,10,0,0,12,0,-8,10,10,0\n"
    )
    rows = [
        "frame,time,"
        "LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LFIN_x,LFIN_y,LFIN_z,"
        "RFRM_x,RFRM_y,RFRM_z,RWRA_x,RWRA_y,RWRA_z,RWRB_x,RWRB_y,RWRB_z,RFIN_x,RFIN_y,RFIN_z\n"
    ]
    rows.append("0,0,0,0,0,-10,0,0,-10,10,0,0,0,0,10,0,0,12,0,-8,10,10,0\n")
    dynamic.write_text("".join(rows))
    seg = {"L_Hand": ["LWRA", "LWRB", "LFIN"], "R_Hand": ["RWRA", "RWRB", "RFIN"]}
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    cfg["synthesize_missing_foot_heels"] = False
    gap_fill(
        str(dynamic),
        str(out),
        seg,
        static_csv_path=str(static),
        config=cfg,
    )

    import pandas as pd

    fills = pd.read_csv(str(out).replace(".csv", "_fills.csv"))
    lwrb = fills[fills["marker"] == "LWRB"]
    assert not lwrb.empty
    assert (lwrb["method"] == "contralateral_hand").all()
    rwrb = fills[fills["marker"] == "RWRB"]
    assert rwrb.empty or (rwrb["method"] == "static_hand_forearm").all()


def test_static_wrb_stays_posterior_to_wra(tmp_path: Path) -> None:
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    static.write_text(
        "frame,time,LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LWRB_x,LWRB_y,LWRB_z,LFIN_x,LFIN_y,LFIN_z\n"
        "0,0,0,0,0,10,0,0,12,0,-8,10,10,0\n"
    )
    dynamic.write_text(
        "frame,time,LFRM_x,LFRM_y,LFRM_z,LWRA_x,LWRA_y,LWRA_z,LFIN_x,LFIN_y,LFIN_z\n"
        "0,0,0,0,0,10,0,0,10,10,0\n"
    )
    seg = {"L_Hand": ["LWRA", "LWRB", "LFIN"]}
    cfg = dict(DEFAULT_GAP_FILLING_CONFIG)
    cfg["synthesize_hand_from_contralateral"] = False
    cfg["synthesize_missing_foot_heels"] = False
    gap_fill(str(dynamic), str(out), seg, static_csv_path=str(static), config=cfg)

    import pandas as pd

    df = pd.read_csv(out)
    assert float(df["LWRB_z"].iloc[0]) < float(df["LWRA_z"].iloc[0])
