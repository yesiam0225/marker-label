"""Tests for walking setup auto-detection and lead-foot crossing."""

from __future__ import annotations

import numpy as np

from marker_label.walking_setup import determine_setup, find_lead_foot_crossing


def _make_markers(
    *,
    n: int = 80,
    walk_axis: int = 0,
    direction: int = 1,
    ltoe_leads: bool = True,
) -> dict[str, np.ndarray]:
    z = np.full(n, 1000.0)
    prog = np.linspace(-200.0, 200.0, n) * float(direction)
    x = prog if walk_axis == 0 else np.zeros(n)
    y = prog if walk_axis == 1 else np.zeros(n)

    lasi = np.stack([x.copy(), y.copy(), z.copy()], axis=1)
    rasi = np.stack([x.copy(), y.copy(), z.copy()], axis=1)
    # ML separation
    ml_axis = 1 - walk_axis
    lasi[:, ml_axis] = 50.0
    rasi[:, ml_axis] = -50.0

    ltoe = np.stack([x.copy(), y.copy(), z.copy()], axis=1)
    rtoe = np.stack([x.copy(), y.copy(), z.copy()], axis=1)
    # Lead foot crosses obstacle earlier.
    lead_offset = 40.0 if ltoe_leads else 10.0
    trail_offset = 10.0 if ltoe_leads else 40.0
    ltoe[:, walk_axis] += lead_offset
    rtoe[:, walk_axis] += trail_offset

    ob_l = np.zeros((n, 3))
    ob_r = np.zeros((n, 3))
    ob_l[:, ml_axis] = -100.0
    ob_r[:, ml_axis] = 100.0

    return {
        "LASI": lasi,
        "RASI": rasi,
        "LTOE": ltoe,
        "RTOE": rtoe,
        "OBSTACLE_L": ob_l,
        "OBSTACLE_R": ob_r,
    }


def test_determine_setup_direction_neg_x() -> None:
    markers = _make_markers(walk_axis=0, direction=-1, ltoe_leads=True)
    setup = determine_setup(markers, ("OBSTACLE_L", "OBSTACLE_R"), warnings_list=[])
    assert setup["walking_axis"] == 0
    assert setup["walking_direction"] == -1
    assert setup["ml_axis"] == 1
    assert setup["left_ml_sign"] == 1


def test_determine_setup_direction_pos_x() -> None:
    markers = _make_markers(walk_axis=0, direction=1, ltoe_leads=True)
    setup = determine_setup(markers, ("OBSTACLE_L", "OBSTACLE_R"), warnings_list=[])
    assert setup["walking_axis"] == 0
    assert setup["walking_direction"] == 1


def test_determine_setup_walking_axis_y() -> None:
    markers = _make_markers(walk_axis=1, direction=1, ltoe_leads=False)
    setup = determine_setup(markers, ("OBSTACLE_L", "OBSTACLE_R"), warnings_list=[])
    assert setup["walking_axis"] == 1
    assert setup["ml_axis"] == 0


def test_find_lead_foot_crossing_side() -> None:
    markers = _make_markers(walk_axis=0, direction=1, ltoe_leads=True)
    setup = determine_setup(markers, ("OBSTACLE_L", "OBSTACLE_R"), warnings_list=[])
    crossing, side = find_lead_foot_crossing(
        markers,
        walking_axis=setup["walking_axis"],
        walking_direction=setup["walking_direction"],
        obstacle_pos=setup["obstacle_pos"],
    )
    assert side == "left"
    assert isinstance(crossing, int) and crossing >= 0
