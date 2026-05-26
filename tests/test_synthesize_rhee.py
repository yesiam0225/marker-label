"""Tests for RHEE synthesis from static foot geometry."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from marker_label.synthesize_rhee_from_static import (
    predict_rhee_row,
    static_rhee_coefficients,
    synthesize_rhee_csv,
)


def test_static_coefficients_and_prediction_round_trip() -> None:
    """Static geometry: heel offset in ankle basis is recovered when dynamic matches static."""
    n = 5
    pts = np.full((n, 3, 3), np.nan)
    # RANK, RTOE, RHEE indices 0,1,2
    for f in range(n):
        pts[f, 0, :] = [0.0, 0.0, float(f)]
        pts[f, 1, :] = [100.0, 0.0, float(f)]
        pts[f, 2, :] = [30.0, -40.0, float(f)]
    li = {"RANK": 0, "RTOE": 1, "RHEE": 2}
    vert = np.array([0.0, 1.0, 0.0])
    c, href = static_rhee_coefficients(pts, li, vert)
    pred = predict_rhee_row(pts[0, 0, :], pts[0, 1, :], c, vert, href)
    assert np.allclose(pred, pts[0, 2, :], atol=1e-6)


def test_synthesize_inserts_rhee_column(tmp_path: Path) -> None:
    static = tmp_path / "static.csv"
    dynamic = tmp_path / "dyn.csv"
    out = tmp_path / "out.csv"
    static.write_text(
        "frame,time,RANK_x,RANK_y,RANK_z,RTOE_x,RTOE_y,RTOE_z,RHEE_x,RHEE_y,RHEE_z\n"
        "0,0,0,0,0,200,0,0,50,-80,0\n"
    )
    dynamic.write_text(
        "frame,time,RANK_x,RANK_y,RANK_z,RTOE_x,RTOE_y,RTOE_z\n"
        "0,0,10,0,0,210,0,0\n"
        "1,0.01,10,0,1,210,0,1\n"
    )
    synthesize_rhee_csv(static, dynamic, out)
    text = out.read_text()
    assert "RHEE_x" in text
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    assert len(rows) == 2
    r0 = float(rows[0]["RHEE_x"])
    assert abs(r0 - 60.0) < 1e-3
