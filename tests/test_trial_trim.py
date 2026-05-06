"""Tests for trial_trim: Kabsch, trim boundaries, sidecar, CSV round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from marker_label.trial_trim import (
    _interval_contains_k_consecutive_bad,
    bestframe_sidecar_path,
    build_visibility_union_indices,
    compute_diagnostics,
    compute_reference_geometry,
    find_trim_boundary,
    kabsch,
    load_best_frame_1based,
    parse_labeled_csv,
    resolve_visibility_ratio_marker_indices,
    save_bestframe_sidecar,
    save_trimmed_csv,
    segment_markers_dict_for_trim_preset,
    trim_trial,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_kabsch_synthetic_recovery():
    rng = np.random.default_rng(42)
    p_ref = rng.standard_normal((6, 3))
    r_true = _random_rotation(rng)
    t_true = rng.standard_normal(3)
    p_cur = p_ref @ r_true.T + t_true
    r_est, t_est = kabsch(p_ref, p_cur)
    pred = p_ref @ r_est.T + t_est
    err = np.linalg.norm(p_cur - pred, axis=1).max()
    assert err < 1e-10


def test_kabsch_reflection_correction():
    """SVD without det-fix can yield det(R)<0; our Kabsch forces a proper rotation (det>0)."""
    rng = np.random.default_rng(7)
    p_ref = rng.standard_normal((5, 3))
    p_cur = rng.standard_normal((5, 3))
    r_est, _t_est = kabsch(p_ref, p_cur)
    assert np.linalg.det(r_est) > 0
    assert abs(np.linalg.det(r_est) - 1.0) < 1e-10


def test_interval_k_bad_run():
    bad = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=bool)
    assert _interval_contains_k_consecutive_bad(bad, 3, 0, 7)
    assert not _interval_contains_k_consecutive_bad(bad, 3, 4, 5)
    assert _interval_contains_k_consecutive_bad(bad, 4, 3, 6)


def _write_minimal_csv(
    path: Path,
    *,
    n_frames: int,
    first_frame: int = 1,
    rate: float = 100.0,
    labels: list[str],
    point_fn,
) -> None:
    """point_fn(i_frame, label_idx) -> (x,y,z)"""
    lines = ["frame,time," + ",".join(f"{lab}_x,{lab}_y,{lab}_z" for lab in labels) + "\n"]
    for i in range(n_frames):
        fnum = first_frame + i
        t = i / rate
        row = [str(fnum), f"{t:.6f}"]
        for li, _lab in enumerate(labels):
            x, y, z = point_fn(i, li)
            row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        lines.append(",".join(row) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_parse_frames_must_be_contiguous(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text(
        "frame,time,A_x,A_y,A_z\n"
        "1,0.0,0,0,0\n"
        "3,0.02,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contiguous"):
        parse_labeled_csv(p)


def test_best_frame_sidecar_path():
    p = Path("/tmp/foo_labeled.csv")
    assert bestframe_sidecar_path(p) == Path("/tmp/foo_labeled.csv.bestframe")


def test_residuals_zero_at_best_and_trim_sidecar(tmp_path):
    """Rigid motion of a triangle + noiseless segment; best in middle; trim full range."""
    labels = ["M0", "M1", "M2"]
    n = 120
    best_idx = 55
    best_1 = best_idx + 1

    tri = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [30.0, 80.0, 0.0],
        ]
    )

    # Per-frame rotation + translation (rigid body)
    rots = [_random_rotation(np.random.default_rng(i + 1)) for i in range(n)]
    trans = [np.array([i * 0.1, -i * 0.05, i * 0.02]) for i in range(n)]

    def pf(i: int, li: int):
        return rots[i] @ tri[li] + trans[i]

    csv_in = tmp_path / "trial_labeled.csv"
    _write_minimal_csv(csv_in, n_frames=n, labels=labels, point_fn=pf)
    bestframe_sidecar_path(csv_in).write_text(str(best_1), encoding="utf-8")

    seg = {"seg0": ["M0", "M1", "M2"]}
    _, meta = parse_labeled_csv(csv_in)
    ref = compute_reference_geometry(meta, seg, best_idx)
    vis_idx = [0, 1, 2]
    diag = compute_diagnostics(
        meta,
        ref,
        seg,
        visibility_ratio_indices=vis_idx,
        residual_threshold_mm=15.0,
        residual_threshold_thigh_mm=25.0,
        thigh_segment_keywords=("thigh", "femur"),
        visible_ratio_threshold=0.9,
    )
    for k, v in diag[best_idx]["segment_residual_mm"].items():
        assert v < 1e-6, f"{k} residual at best {v}"

    out_csv = tmp_path / "out_labeled.csv"
    result = trim_trial(csv_in, out_csv, seg, config={"consecutive_bad_frames": 5})
    assert result["trim_start"] == 1
    assert result["trim_end"] == n + 1
    assert result["best_frame_1based"] == best_1
    assert load_best_frame_1based(out_csv) == best_1

    _, meta_o = parse_labeled_csv(out_csv)
    assert meta_o["n_frames"] == n
    np.testing.assert_allclose(meta_o["points"], meta["points"], rtol=0, atol=1e-5)


def test_trim_shortens_after_many_bad_frames(tmp_path):
    labels = ["M0", "M1", "M2"]
    n = 80
    tri = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [10.0, 40.0, 0.0]])

    def pf(i: int, li: int):
        # Non-rigid change: translation is rigid, but per-marker scaling breaks the segment
        if i > 40:
            return (tri[li] * np.array([3.0 + 0.5 * li, 1.0, 1.0])).copy()
        return tri[li]

    csv_in = tmp_path / "b.csv"
    _write_minimal_csv(csv_in, n_frames=n, labels=labels, point_fn=pf)
    best_1 = 25
    bestframe_sidecar_path(csv_in).write_text(str(best_1), encoding="utf-8")
    seg = {"arm": ["M0", "M1", "M2"]}
    out_csv = tmp_path / "bout.csv"
    res = trim_trial(
        csv_in,
        out_csv,
        seg,
        config={
            "consecutive_bad_frames": 3,
            "residual_threshold_mm": 1.0,
            "min_trial_length": 10,
        },
    )
    assert res["trim_end_row"] - res["trim_start_row"] < n
    assert res["trim_start"] <= best_1 < res["trim_end"]


def test_thigh_segment_higher_threshold_in_diagnostics():
    n = 5
    labels = ["M0", "M1", "M2"]
    tri = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [10.0, 40.0, 0.0]])

    def pf(_i: int, li: int):
        return tri[li]

    # in-memory metadata via temp file
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        _write_minimal_csv(p, n_frames=n, labels=labels, point_fn=pf)
        _, meta = parse_labeled_csv(p)
    seg = {"L_Thigh": ["M0", "M1", "M2"]}
    ref = compute_reference_geometry(meta, seg, 2)
    vis = [0, 1, 2]
    diag = compute_diagnostics(
        meta,
        ref,
        seg,
        visibility_ratio_indices=vis,
        residual_threshold_mm=15.0,
        residual_threshold_thigh_mm=25.0,
        thigh_segment_keywords=("thigh", "femur"),
        visible_ratio_threshold=0.9,
    )
    assert diag[0]["segment_threshold_mm"]["L_Thigh"] == 25.0


def test_find_trim_boundary_raises_if_best_bad():
    diag = []
    for _ in range(10):
        diag.append(
            {
                "is_bad": False,
                "segment_residual_mm": {},
                "segment_threshold_mm": {},
            }
        )
    diag[5]["is_bad"] = True
    with pytest.raises(ValueError, match="Best frame is QC-bad"):
        find_trim_boundary(diag, 5, consecutive_bad_frames=1)


def test_save_trimmed_preserves_header_padding(tmp_path):
    header = "frame,time,  AB_x ,  AB_y ,  AB_z \n"
    rows = ["1,0.01,1,2,3\n", "2,0.02,1,2,3\n"]
    p = tmp_path / "padded.csv"
    p.write_text(header + "".join(rows), encoding="utf-8")
    _, meta = parse_labeled_csv(p)
    out = tmp_path / "out.csv"
    save_trimmed_csv(meta, 0, 2, out)
    first = out.read_text(encoding="utf-8").splitlines(True)[0]
    assert first == header


def test_plot_trim_diagnostics_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from marker_label.trial_trim import plot_trim_diagnostics

    labels = ["M0", "M1", "M2"]
    n = 30

    def pf(i: int, li: int):
        base = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [2.0, 8.0, 0.0]])
        return base[li] + np.array([0.01 * i, 0.0, 0.0])

    p = tmp_path / "d.csv"
    _write_minimal_csv(p, n_frames=n, labels=labels, point_fn=pf)
    seg = {"s": ["M0", "M1", "M2"]}
    _, meta = parse_labeled_csv(p)
    ref = compute_reference_geometry(meta, seg, 10)
    diag = compute_diagnostics(
        meta,
        ref,
        seg,
        visibility_ratio_indices=[0, 1, 2],
        residual_threshold_mm=15.0,
        residual_threshold_thigh_mm=25.0,
        thigh_segment_keywords=(),
        visible_ratio_threshold=0.5,
    )
    png = tmp_path / "fig.png"
    fig = plot_trim_diagnostics(
        diag,
        meta,
        best_frame_1based=11,
        trim_start=1,
        trim_end=31,
        save_path=png,
    )
    assert png.is_file()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_trim_trial_cli_json_segments(tmp_path, monkeypatch, capsys):
    labels = ["M0", "M1", "M2"]
    n = 20

    def pf(_i: int, li: int):
        b = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [2.0, 8.0, 0.0]])
        return b[li]

    csv_in = tmp_path / "c.csv"
    _write_minimal_csv(csv_in, n_frames=n, labels=labels, point_fn=pf)
    bestframe_sidecar_path(csv_in).write_text("10", encoding="utf-8")
    seg_path = tmp_path / "seg.json"
    seg_path.write_text(json.dumps({"s": ["M0", "M1", "M2"]}), encoding="utf-8")
    out = tmp_path / "co.csv"

    from marker_label import trial_trim

    monkeypatch.setattr(
        "sys.argv",
        [
            "trial_trim",
            str(csv_in),
            "-o",
            str(out),
            "--segments-json",
            str(seg_path),
            "--min-trial-length",
            "5",
        ],
    )
    trial_trim.main()
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["best_frame_1based"] == 10
    assert Path(out).is_file()


def test_save_bestframe_sidecar(tmp_path):
    p = tmp_path / "x_labeled.csv"
    save_bestframe_sidecar(p, 42)
    assert bestframe_sidecar_path(p).read_text() == "42"


def test_segment_markers_dict_for_trim_preset_lower_body_no_psis():
    d = segment_markers_dict_for_trim_preset("lower-body")
    assert "Pelvis_ASIS_LThigh" in d
    all_names = {str(m).strip() for names in d.values() for m in names}
    assert "LPSI" not in all_names and "RPSI" not in all_names
    assert {"LASI", "RASI", "LANK", "RANK"}.issubset(all_names)


def test_segment_markers_dict_for_trim_preset_legs_feet_no_asis():
    d = segment_markers_dict_for_trim_preset("legs-feet")
    all_names = {str(m).strip() for names in d.values() for m in names}
    assert "LASI" not in all_names and "RASI" not in all_names
    assert "LPSI" not in all_names and "RPSI" not in all_names
    assert {"LTHI", "LANK", "RTHI", "RANK"}.issubset(all_names)


def test_segment_markers_dict_for_trim_preset_legs_feet_alias():
    lf = segment_markers_dict_for_trim_preset("legs-feet")
    assert segment_markers_dict_for_trim_preset("legs_feet") == lf
    assert segment_markers_dict_for_trim_preset("default") == lf


def test_segment_markers_dict_for_trim_preset_default_is_legs_feet_no_asis():
    d = segment_markers_dict_for_trim_preset("default")
    all_names = {str(m).strip() for names in d.values() for m in names}
    assert "LASI" not in all_names and "RASI" not in all_names
    assert "L_Thigh" in d


def test_segment_markers_dict_for_trim_preset_full_body_skips_two_marker_segments():
    d = segment_markers_dict_for_trim_preset("full-body")
    assert "L_Forearm_FRM_WRB" not in d
    assert "L_Thigh" in d
    assert "Head" in d


def test_segment_markers_dict_for_trim_preset_unknown():
    with pytest.raises(ValueError, match="Unknown segment preset"):
        segment_markers_dict_for_trim_preset("torso-only")


def test_resolve_visibility_ratio_falls_back_when_no_leg_columns(tmp_path):
    labels = ["M0", "M1", "M2"]

    def pf(_i: int, li: int):
        return np.zeros(3)

    p = tmp_path / "m.csv"
    _write_minimal_csv(p, n_frames=3, labels=labels, point_fn=pf)
    _, meta = parse_labeled_csv(p)
    seg = {"s": ["M0", "M1", "M2"]}
    u_idx, u_lab = build_visibility_union_indices(meta, seg)
    legs_idx, legs_lab = resolve_visibility_ratio_marker_indices(
        meta, u_idx, u_lab, {"visibility_ratio_marker_subset": "legs-feet"}
    )
    assert legs_idx == u_idx
    su_idx, su_lab = resolve_visibility_ratio_marker_indices(
        meta, u_idx, u_lab, {"visibility_ratio_marker_subset": "segment-union"}
    )
    assert su_idx == u_idx


def test_trim_trial_with_legs_feet_preset_smoke(tmp_path):
    labels = [
        "LTHI",
        "LKNE",
        "LTIB",
        "LANK",
        "LHEE",
        "LTOE",
        "RTHI",
        "RKNE",
        "RTIB",
        "RANK",
        "RHEE",
        "RTOE",
    ]
    rng = np.random.default_rng(101)
    base = rng.standard_normal((len(labels), 3)) * 100.0

    def pf(i: int, li: int):
        r = _random_rotation(np.random.default_rng(i + 7))
        return r @ base[li] + np.array([0.0, 0.0, 0.01 * i])

    n = 40
    csv_in = tmp_path / "lf.csv"
    _write_minimal_csv(csv_in, n_frames=n, labels=labels, point_fn=pf)
    bestframe_sidecar_path(csv_in).write_text("20", encoding="utf-8")
    out = tmp_path / "lf_out.csv"
    seg = segment_markers_dict_for_trim_preset("legs-feet")
    res = trim_trial(csv_in, out, seg, config={"min_trial_length": 5})
    assert res["best_frame_1based"] == 20
    assert Path(out).is_file()


def test_trim_trial_with_lower_body_preset_smoke(tmp_path):
    """Rigid motion of all lower-body preset markers; trim should succeed."""
    labels = [
        "LASI",
        "RASI",
        "LTHI",
        "LKNE",
        "LTIB",
        "LANK",
        "LHEE",
        "LTOE",
        "RTHI",
        "RKNE",
        "RTIB",
        "RANK",
        "RHEE",
        "RTOE",
    ]
    rng = np.random.default_rng(99)
    base = rng.standard_normal((len(labels), 3)) * 100.0

    def pf(i: int, li: int):
        r = _random_rotation(np.random.default_rng(i + 3))
        return r @ base[li] + np.array([0.0, 0.0, 0.01 * i])

    n = 50
    csv_in = tmp_path / "lb.csv"
    _write_minimal_csv(csv_in, n_frames=n, labels=labels, point_fn=pf)
    bestframe_sidecar_path(csv_in).write_text("25", encoding="utf-8")
    out = tmp_path / "lb_out.csv"
    seg = segment_markers_dict_for_trim_preset("lower-body")
    res = trim_trial(
        csv_in,
        out,
        seg,
        config={"min_trial_length": 10, "consecutive_bad_frames": 5},
    )
    assert res["best_frame_1based"] == 25
    assert Path(out).is_file()


def test_trim_trial_cli_segments_json_warns_if_preset_not_default(tmp_path, monkeypatch, capsys):
    labels = ["M0", "M1", "M2"]
    n = 10

    def pf(_i: int, li: int):
        b = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [2.0, 8.0, 0.0]])
        return b[li]

    csv_in = tmp_path / "c2.csv"
    _write_minimal_csv(csv_in, n_frames=n, labels=labels, point_fn=pf)
    bestframe_sidecar_path(csv_in).write_text("5", encoding="utf-8")
    seg_path = tmp_path / "seg2.json"
    seg_path.write_text(json.dumps({"s": ["M0", "M1", "M2"]}), encoding="utf-8")
    out = tmp_path / "co2.csv"

    from marker_label import trial_trim

    monkeypatch.setattr(
        "sys.argv",
        [
            "trial_trim",
            str(csv_in),
            "-o",
            str(out),
            "--preset",
            "lower-body",
            "--segments-json",
            str(seg_path),
            "--min-trial-length",
            "3",
        ],
    )
    trial_trim.main()
    err = capsys.readouterr().err
    assert "ignoring --preset" in err
