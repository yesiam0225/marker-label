"""Tests for CSV loading alongside C3D in marker_label.io."""

from __future__ import annotations

import numpy as np

from marker_label.io import load_c3d_or_csv


def test_load_c3d_or_csv_reads_flat_labeled_csv(tmp_path) -> None:
    p = tmp_path / "trial.csv"
    p.write_text(
        "frame,time,LASI_x,LASI_y,LASI_z,RASI_x,RASI_y,RASI_z\n"
        "0,0.0,10,20,30,40,50,60\n"
        "1,0.01,11,21,31,41,51,61\n"
    )
    d = load_c3d_or_csv(p, scale_factor=1.0)
    assert d["n_frames"] == 2
    assert d["n_points"] == 2
    assert d["labels"] == ["LASI", "RASI"]
    assert d["residual"] is None
    assert np.allclose(d["points"][0, 0], [10.0, 20.0, 30.0])
    assert int(d["first_frame"]) == 0
    assert abs(d["rate"] - 100.0) < 1e-6


def test_load_c3d_or_csv_scale_factor(tmp_path) -> None:
    p = tmp_path / "one.csv"
    p.write_text("frame,time,A_x,A_y,A_z\n0,0.0,1,2,3\n")
    d = load_c3d_or_csv(p, scale_factor=1000.0)
    assert np.allclose(d["points"][0, 0], [1000.0, 2000.0, 3000.0])
