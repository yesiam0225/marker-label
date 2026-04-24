"""Static pelvis distance geometry used to pick 4 of N shoulder-band candidates."""

import numpy as np

from marker_label.body_labeling import _select_four_pelvis_indices_by_static_geometry


def test_select_four_prefers_subset_matching_template_shape():
    # Reference: 4 points in a rough square in x-y; fifth point is an outlier far in x
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([200.0, 0.0, 0.0])
    p2 = np.array([0.0, 200.0, 0.0])
    p3 = np.array([200.0, 200.0, 0.0])
    p_bad = np.array([5000.0, 0.0, 0.0])
    points = np.array([p0, p1, p2, p3, p_bad], dtype=np.float64)
    d_ref = np.linalg.norm(
        np.stack([p0, p1, p2, p3], axis=0)[:, None, :]
        - np.stack([p0, p1, p2, p3], axis=0)[None, :, :],
        axis=-1,
    )
    out, rms = _select_four_pelvis_indices_by_static_geometry(points, [0, 1, 2, 3, 4], d_ref)
    assert out is not None
    assert set(out) == {0, 1, 2, 3}
    assert rms is not None
    assert rms < 1e-6
