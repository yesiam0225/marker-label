"""Minimal tests for pipeline components (no C3D files required)."""

import numpy as np
import pytest

from marker_label.obstacle import (
    detect_obstacle_markers,
    detect_obstacle_markers_rod_pair,
    median_motion_per_marker,
    median_y_per_marker,
    motion_score_per_marker,
    screened_indices_extra_stationary_to_drop,
    visibility_fraction,
)
from marker_label.body_labeling import (
    assign_strn_t10_arm4_after_clav_rbak,
    best_frame_for_matching,
    match_markers_to_template,
)
from marker_label.gap_fill import fill_gaps_1d, fill_gaps_trajectory
from marker_label.errors import LabelingPipelineError
from marker_label.pipeline import loaded_column_indices_matching_label_regex


def test_motion_score_shape():
    n_frames, n_points = 50, 10
    points = np.random.randn(n_frames, n_points, 3).astype(np.float64)
    score = motion_score_per_marker(points)
    assert score.shape == (n_points,)


def test_visibility_fraction():
    points = np.ones((10, 3, 3))
    points[2:5, 1, :] = np.nan
    vis = visibility_fraction(points)
    assert vis[0] == 1.0
    assert vis[1] == 0.7
    assert vis[2] == 1.0


def test_median_motion_per_marker_shape():
    n_frames, n_points = 40, 4
    points = np.random.randn(n_frames, n_points, 3).astype(np.float64)
    score = median_motion_per_marker(points)
    assert score.shape == (n_points,)


def test_median_y_per_marker():
    pts = np.zeros((5, 2, 3), dtype=np.float64)
    pts[:, 0, 1] = [1.0, 2.0, np.nan, 4.0, 5.0]
    pts[:, 1, 1] = np.nan
    m = median_y_per_marker(pts)
    assert m[0] == 3.0
    assert np.isnan(m[1])


def test_detect_obstacle_rod_pair_bar_along_y():
    """Two quiet endpoints on a long rod along Y; walking noise on other columns."""
    n_frames, n_pts = 150, 8
    rng = np.random.default_rng(0)
    points = rng.standard_normal((n_frames, n_pts, 3)) * 8.0 + np.array([600.0, 200.0, 400.0])
    points[:, 0, :] = np.array([100.0, 50.0, 80.0]) + rng.standard_normal((n_frames, 3)) * 0.3
    points[:, 1, :] = np.array([110.0, 1400.0, 85.0]) + rng.standard_normal((n_frames, 3)) * 0.3
    idx, labels = detect_obstacle_markers_rod_pair(
        points,
        visibility_min=0.95,
        visibility_floor=0.55,
        motion_max_mm=5.0,
        rod_length_min_mm=400.0,
        rod_separation_axis="y",
        rod_min_overlap_frames=30,
        candidate_y_min_mm=-500.0,
        candidate_y_max_mm=1500.0,
    )
    assert set(idx) == {0, 1}
    assert len(labels) == 2


def test_detect_obstacle_rod_pair_y_band_removes_out_of_range_spurious_rod():
    """In-band true rod (cols 0,1) vs a tighter fake pair at high Y: band must select 0,1."""
    n_frames, n_pts = 120, 5
    rng = np.random.default_rng(1)
    points = rng.standard_normal((n_frames, n_pts, 3)) * 6.0 + np.array([500.0, 200.0, 400.0])
    # Spurious: very collinear and long along Y, but median Y out of default band
    points[:, 2, :] = np.array([200.0, 3500.0, 200.0]) + rng.standard_normal((n_frames, 3)) * 0.1
    points[:, 3, :] = np.array([205.0, 4200.0, 200.0]) + rng.standard_normal((n_frames, 3)) * 0.1
    # True rod ends inside -500..1500
    points[:, 0, :] = np.array([150.0, 100.0, 200.0]) + rng.standard_normal((n_frames, 3)) * 0.2
    points[:, 1, :] = np.array([150.0, 1300.0, 200.0]) + rng.standard_normal((n_frames, 3)) * 0.2
    points[:, 4, :] = np.nan
    idx, _labels = detect_obstacle_markers_rod_pair(
        points,
        visibility_min=0.95,
        visibility_floor=0.55,
        motion_max_mm=5.0,
        rod_length_min_mm=400.0,
        rod_separation_axis="y",
        rod_max_pair_candidates=5,
        rod_min_overlap_frames=20,
        candidate_y_min_mm=-500.0,
        candidate_y_max_mm=1500.0,
    )
    assert set(idx) == {0, 1}


def test_detect_obstacle_two_stationary():
    n_frames, n_points = 100, 5
    points = np.random.randn(n_frames, n_points, 3) * 0.1
    points[:, 0, :] = 1.0  # stationary
    points[:, 1, :] = 2.0  # stationary
    indices, labels = detect_obstacle_markers(
        points,
        n_obstacle=2,
        visibility_min=0.5,
        motion_max_mm=0,
    )
    assert len(indices) == 2
    assert len(labels) == 2


def test_screened_extra_stationary_drop_uses_full_trajectory_not_best_frame():
    """Contract: extra stationary drop uses mean motion over all frames; labeling best frame is unrelated."""
    n_frames = 50
    pts = np.zeros((n_frames, 4, 3), dtype=np.float64)
    pts[:, 0, :] = 1.0
    pts[:, 1, :] = 2.0
    pts[:, 2, :] = 3.0  # extra stationary (non-obstacle)
    pts[:, 3, :] = np.linspace(0.0, 500.0, n_frames)[:, np.newaxis] + np.array([4.0, 4.0, 4.0])
    obstacle_indices = [0, 1]
    dropped = screened_indices_extra_stationary_to_drop(
        pts,
        obstacle_indices,
        motion_max_mm=10.0,
        visibility_min=0.5,
    )
    assert 0 not in dropped and 1 not in dropped
    assert 2 in dropped
    assert 3 not in dropped


def test_foot_triple_longest_edge_opposite_vertex_is_ank():
    """Longest 3D edge among foot triple → HEE/TOE endpoints; third vertex → ANK (set logic)."""
    # HEE, TOE far apart on x; ANK offset — longest edge is HEE–TOE, opposite vertex is ANK.
    pts3 = np.array(
        [
            [0.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [80.0, 60.0, 0.0],
        ],
        dtype=np.float64,
    )
    pair_edges = (
        (0, 1, float(np.linalg.norm(pts3[0] - pts3[1]))),
        (0, 2, float(np.linalg.norm(pts3[0] - pts3[2]))),
        (1, 2, float(np.linalg.norm(pts3[1] - pts3[2]))),
    )
    ia, ib, max_d = max(pair_edges, key=lambda t: t[2])
    ic = ({0, 1, 2} - {ia, ib}).pop()
    assert ia == 0 and ib == 1
    assert max_d == 200.0
    assert ic == 2


def test_match_markers_to_template():
    template = {"A": np.array([0.0, 0.0, 0.0]), "B": np.array([1.0, 0.0, 0.0])}
    points = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])
    assignments = match_markers_to_template(points, template)
    assert len(assignments) == 2
    labels = [a[1] for a in assignments]
    assert "A" in labels and "B" in labels


def test_best_frame_middle():
    n_frames, n_points = 100, 5
    points = np.random.randn(n_frames, n_points, 3)
    residual = np.random.rand(n_frames, n_points) * 5
    residual[50, :] = 0.1  # best at 50
    best = best_frame_for_matching(points, residual, middle_start=0.2, middle_end=0.8)
    assert 20 <= best < 80


def test_strn_t10_ap_guard_swaps_misordered_y_ref_pick():
    """If Y-ref selection inverts A/P, STRN/T10 are swapped back by AP guard."""
    pts = np.full((10, 3), np.nan, dtype=np.float64)
    # Shoulder band
    pts[0] = [0.0, 200.0, 1200.0]  # LSHO
    pts[1] = [0.0, 500.0, 1190.0]  # RSHO
    # Two STRN/T10 candidates in shoulder Y-band (posterior has larger x because d_back=[1,0])
    pts[2] = [100.0, 340.0, 1100.0]  # anterior candidate (should become STRN)
    pts[3] = [300.0, 330.0, 1095.0]  # posterior candidate (should become T10)
    # Four arm candidates
    pts[4] = [80.0, 160.0, 1080.0]
    pts[5] = [70.0, 180.0, 1075.0]
    pts[6] = [90.0, 520.0, 1070.0]
    pts[7] = [85.0, 540.0, 1065.0]
    # Y references used by STRN/T10 split
    pts[8] = [0.0, 330.0, 1000.0]  # CLAV
    pts[9] = [0.0, 340.0, 1010.0]  # C7

    template = {
        "STRN": np.array([0.0, 0.0, 0.0]),
        "T10": np.array([0.0, 0.0, 0.0]),
        "LUPA": np.array([0.0, 0.0, 0.0]),
        "RUPA": np.array([0.0, 0.0, 0.0]),
        "LELB": np.array([0.0, 0.0, 0.0]),
        "RELB": np.array([0.0, 0.0, 0.0]),
    }
    out = assign_strn_t10_arm4_after_clav_rbak(
        pts,
        d_back_xy=np.array([1.0, 0.0]),
        d_right_xy=np.array([0.0, 1.0]),
        exclude_pt_indices={0, 1, 8, 9},
        lsho_idx=0,
        rsho_idx=1,
        template=template,
        clav_body_idx=8,
        c7_body_idx=9,
        strn_t10_ap_margin_mm=10.0,
    )
    by_label = {lab: pi for pi, lab in out}
    assert by_label["STRN"] == 2
    assert by_label["T10"] == 3


def test_fill_gaps_1d_short():
    x = np.array([1.0, np.nan, np.nan, 4.0])
    out = fill_gaps_1d(x, max_interp_frames=10)
    assert np.isfinite(out).all()
    assert out[0] == 1.0 and out[3] == 4.0


def test_fill_gaps_trajectory():
    pts = np.ones((5, 3))
    pts[1:3, :] = np.nan
    out = fill_gaps_trajectory(pts, max_interp_frames=10)
    assert np.isfinite(out).all()


def test_loaded_column_indices_matching_label_regex():
    labels = ["A", "*1                            ", "B*", "xx*yy"]
    assert loaded_column_indices_matching_label_regex(labels, r"^\*") == [1]
    assert loaded_column_indices_matching_label_regex(labels, r"\*") == [1, 2, 3]


def test_loaded_column_indices_matching_label_regex_invalid():
    with pytest.raises(LabelingPipelineError):
        loaded_column_indices_matching_label_regex(["a"], "(")
