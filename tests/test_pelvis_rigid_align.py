import numpy as np

from marker_label.pelvis import rigid_pelvis_from_template_to_observed


def test_rigid_pelvis_identity():
    t = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [100.0, 100.0, 0.0],
        ],
        dtype=np.float64,
    )
    o = t + np.array([1.0, 2.0, 3.0])
    out = rigid_pelvis_from_template_to_observed(o, t)
    assert np.allclose(out, o, atol=1e-5)


def test_rigid_pelvis_rotated():
    rng = np.random.default_rng(0)
    t = rng.standard_normal((4, 3)) * 50.0
    r_true, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if float(np.linalg.det(r_true)) < 0:
        r_true[:, 0] *= -1
    shift = np.array([10.0, -20.0, 5.0])
    o = (r_true @ t.T).T + shift
    out = rigid_pelvis_from_template_to_observed(o, t)
    assert np.allclose(out, o, atol=1e-4)
    d_t = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1)
    d_out = np.linalg.norm(out[:, None, :] - out[None, :, :], axis=-1)
    assert np.allclose(d_t, d_out, atol=1e-3)
