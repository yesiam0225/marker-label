"""Minimal tests for pipeline components (no C3D files required)."""

import numpy as np
import pytest

from marker_label.obstacle import motion_score_per_marker, visibility_fraction, detect_obstacle_markers
from marker_label.body_labeling import match_markers_to_template, best_frame_for_matching
from marker_label.gap_fill import fill_gaps_1d, fill_gaps_trajectory


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


def test_detect_obstacle_two_stationary():
    n_frames, n_points = 100, 5
    points = np.random.randn(n_frames, n_points, 3) * 0.1
    points[:, 0, :] = 1.0  # stationary
    points[:, 1, :] = 2.0  # stationary
    indices, labels = detect_obstacle_markers(points, n_obstacle=2, visibility_min=0.5)
    assert len(indices) == 2
    assert len(labels) == 2


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
