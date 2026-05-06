"""Tests for manual frame-range trimming of labeled CSV."""

from __future__ import annotations

from pathlib import Path

import pytest

from marker_label.trim_csv_by_frames import trim_labeled_csv_by_frame_range
from marker_label.trial_trim import bestframe_sidecar_path, parse_labeled_csv


def _minimal_csv(path: Path) -> None:
    lines = [
        "frame,time,A_x,A_y,A_z\n",
        "0,0.0,1,1,1\n",
        "1,0.01,2,2,2\n",
        "2,0.02,3,3,3\n",
        "3,0.03,4,4,4\n",
        "4,0.04,5,5,5\n",
    ]
    path.write_text("".join(lines))


def test_trim_by_frame_range_keeps_rows_and_frames(tmp_path: Path) -> None:
    src = tmp_path / "t.csv"
    out = tmp_path / "t_trim.csv"
    _minimal_csv(src)
    info = trim_labeled_csv_by_frame_range(src, out, 1, 3)
    assert info["trim_start_row"] == 1
    assert info["trim_end_row"] == 4
    text = out.read_text()
    assert "1,0.01" in text
    assert "3,0.03" in text
    assert "0,0.0" not in text
    assert "4,0.04" not in text
    _, meta = parse_labeled_csv(out)
    assert list(meta["frames"]) == [1, 2, 3]


def test_trim_copies_sidecar_when_in_range(tmp_path: Path) -> None:
    src = tmp_path / "t.csv"
    out = tmp_path / "t_trim.csv"
    _minimal_csv(src)
    bestframe_sidecar_path(src).write_text("2")
    trim_labeled_csv_by_frame_range(src, out, 1, 3)
    side = bestframe_sidecar_path(out)
    assert side.is_file()
    assert side.read_text() == "2"


def test_trim_skips_sidecar_when_out_of_range(tmp_path: Path) -> None:
    src = tmp_path / "t.csv"
    out = tmp_path / "t_trim.csv"
    _minimal_csv(src)
    bestframe_sidecar_path(src).write_text("99")
    info = trim_labeled_csv_by_frame_range(src, out, 1, 3)
    assert info["sidecar_written"] is False
    assert not bestframe_sidecar_path(out).is_file()


def test_trim_end_before_start_raises(tmp_path: Path) -> None:
    src = tmp_path / "t.csv"
    _minimal_csv(src)
    with pytest.raises(ValueError, match="end_frame"):
        trim_labeled_csv_by_frame_range(src, tmp_path / "o.csv", 3, 1)


def test_trim_end_beyond_last_frame_raises(tmp_path: Path) -> None:
    src = tmp_path / "t.csv"
    _minimal_csv(src)
    with pytest.raises(ValueError, match="outside CSV frame span"):
        trim_labeled_csv_by_frame_range(src, tmp_path / "o.csv", 1, 10)
