"""Tests for RHEE synthesis from static foot geometry."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from marker_label.synthesize_rhee_from_static import (
    foot_basis_from_ankle_toe,
    medial_reference_ankle_toe_tibia,
    predict_rhee_row,
    shin_axis_ankle_toe_tibia,
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
    c, href, shin_ref = static_rhee_coefficients(pts, li, vert)
    pred = predict_rhee_row(pts[0, 0, :], pts[0, 1, :], c, vert, href, shin_axis_reference=shin_ref)
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


def test_foot_basis_uses_tibia_for_mediolateral() -> None:
    """Mediolateral foot axis follows ankle-tibia, not lab Y alone."""
    vert = np.array([0.0, 1.0, 0.0])
    ank = np.array([0.0, 0.0, 0.0])
    toe = np.array([200.0, 0.0, 0.0])
    tib = np.array([50.0, 80.0, 0.0])
    heel = np.array([40.0, 30.0, 0.0])
    pts = np.stack([ank, toe, heel], axis=0)
    li = {"RANK": 0, "RTOE": 1, "RHEE": 2, "RTIB": 3}
    static_pts = np.full((3, 4, 3), np.nan)
    for f in range(3):
        static_pts[f, 0, :] = ank
        static_pts[f, 1, :] = toe
        static_pts[f, 2, :] = heel
        static_pts[f, 3, :] = tib
    c, href, shin_ref = static_rhee_coefficients(static_pts, li, vert)
    med = medial_reference_ankle_toe_tibia(ank, toe, tib)
    assert float(np.dot(heel - ank, med)) > 0.0
    pred = predict_rhee_row(
        ank, toe, c, vert, href, tibia_xyz=tib, shin_axis_reference=shin_ref
    )
    assert np.allclose(pred, heel, atol=1e-6)


def test_shin_axis_locked_to_static_reference() -> None:
    """Dynamic shin hemisphere flip does not invert synthesized heel."""
    vert = np.array([0.0, 1.0, 0.0])
    ank = np.array([0.0, 0.0, 0.0])
    toe = np.array([200.0, 0.0, 0.0])
    tib = np.array([50.0, 80.0, 0.0])
    heel = np.array([40.0, 30.0, 0.0])
    li = {"RANK": 0, "RTOE": 1, "RHEE": 2, "RTIB": 3}
    static_pts = np.full((3, 4, 3), np.nan)
    for f in range(3):
        static_pts[f, 0, :] = ank
        static_pts[f, 1, :] = toe
        static_pts[f, 2, :] = heel
        static_pts[f, 3, :] = tib
    c, href, shin_ref = static_rhee_coefficients(static_pts, li, vert)
    assert shin_ref is not None
    # Extreme pose: tibia nearly opposite toe; shin projection sign unstable without lock.
    tib_flipped = ank - (tib - ank)
    pred = predict_rhee_row(
        ank, toe, c, vert, href, tibia_xyz=tib_flipped, shin_axis_reference=shin_ref
    )
    assert np.allclose(pred, heel, atol=1e-5)
    _, y_free, _ = foot_basis_from_ankle_toe(ank, toe, vert, tibia=tib_flipped)
    _, y_locked, _ = foot_basis_from_ankle_toe(
        ank, toe, vert, tibia=tib_flipped, shin_axis_reference=shin_ref
    )
    assert float(np.dot(y_free, shin_ref)) < 0.0
    assert float(np.dot(y_locked, shin_ref)) > 0.0
