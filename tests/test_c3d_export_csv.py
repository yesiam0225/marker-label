"""Tests for C3D-style CSV export parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marker_label.inspect_quality import load_labeled_csv
from marker_label.trial_trim import parse_labeled_csv

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "c3d_export_mini.csv"


def test_load_c3d_export_csv():
    points, labels, rate = load_labeled_csv(str(FIXTURE))
    assert points.shape == (2, 2, 3)
    assert rate > 0
    assert "LFHD" in [l.strip() for l in labels]
    assert np.isfinite(points).all()


def test_parse_c3d_export_csv():
    stems, meta = parse_labeled_csv(FIXTURE)
    assert meta["n_frames"] == 2
    assert meta["n_markers"] == 2
    assert "LFHD" in stems
    assert meta.get("csv_format") == "c3d_export"
