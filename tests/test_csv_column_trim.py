"""Tests for labeled CSV marker column trimming."""

import csv
from pathlib import Path

import pytest

from marker_label.csv_column_trim import iter_marker_xyz_triplets, trim_labeled_csv_marker_columns


def test_iter_marker_xyz_triplets():
    header = ["frame", "time", "A_x", "A_y", "A_z", "B_x", "B_y", "B_z"]
    t = iter_marker_xyz_triplets(header)
    assert t == [(2, 3, 4, "A"), (5, 6, 7, "B")]


def test_iter_marker_xyz_triplets_strips_padding():
    header = ["frame", "time", "*1                          _x", "*1                          _y", "*1                          _z"]
    t = iter_marker_xyz_triplets(header)
    assert len(t) == 1
    assert t[0][3] == "*1"


def test_iter_marker_xyz_triplets_bad_stem_mismatch():
    header = ["frame", "time", "A_x", "B_y", "A_z"]
    with pytest.raises(ValueError, match="Mismatched"):
        iter_marker_xyz_triplets(header)


def test_trim_labeled_csv_marker_columns(tmp_path: Path):
    header = ["frame", "time", "PELO_x", "PELO_y", "PELO_z", "*9_x", "*9_y", "*9_z", "55_x", "55_y", "55_z"]
    rows = [header, ["0", "0.0", "1", "2", "3", "", "", "", "7", "8", "9"]]
    inp = tmp_path / "in.csv"
    with open(inp, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    out = tmp_path / "out.csv"
    n_keep, n_drop, dropped = trim_labeled_csv_marker_columns(
        inp, out, stem_regexes=[r"^\*\d+$", r"^\d+$"]
    )
    assert n_keep == 1
    assert n_drop == 2
    assert set(dropped) == {"*9", "55"}
    with open(out, newline="") as f:
        got = list(csv.reader(f))
    assert got[0] == ["frame", "time", "PELO_x", "PELO_y", "PELO_z"]
    assert got[1][:5] == ["0", "0.0", "1", "2", "3"]
