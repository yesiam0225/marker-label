"""Demo walking trajectories for portfolio QC viewer GIFs."""

from __future__ import annotations

import numpy as np

from marker_label.segments import SEGMENTS

# Markers used in SEGMENTS (portfolio demo skeleton).
_DEMO_LABELS: list[str] = sorted(
    {name.strip() for chain in SEGMENTS.values() for name in chain}
)


def _find_marker(labels: list[str], name: str) -> int | None:
    target = name.strip().upper()
    for ix, lab in enumerate(labels):
        if str(lab).strip().upper() == target:
            return ix
    return None


def extract_demo_walking_clip(
    path: str,
    *,
    frame_start: int = 0,
    duration_s: float = 3.0,
) -> tuple[np.ndarray, list[str], float]:
    """
  Load a labeled trial window and return a centered clip for portfolio export.

  Removes obstacle markers, subtracts the frame-0 pelvis anchor, and recenters
  vertically so the pose matches interactive ``marker-label-view`` layout
  without absolute lab coordinates.
  """
    from marker_label.qc_viewer import load_data

    points, labels, rate = load_data(path)
    if rate <= 0:
        rate = 100.0
    n_take = max(1, int(round(duration_s * rate)))
    start = max(0, min(frame_start, points.shape[0] - 1))
    end = min(points.shape[0], start + n_take)
    clip = points[start:end].copy()

    stripped = [str(lab).strip() for lab in labels]
    keep = [i for i, s in enumerate(stripped) if not s.upper().startswith("OBSTACLE")]
    clip = clip[:, keep, :]
    labels = [labels[i] for i in keep]

    pelvis_ix = [
        i
        for name in ("LASI", "RASI", "LPSI", "RPSI")
        for i in (_find_marker(labels, name),)
        if i is not None
    ]
    if pelvis_ix:
        anchor = np.nanmean(clip[0, pelvis_ix, :], axis=0)
    else:
        anchor = np.nanmean(clip[0], axis=(0, 1))
    clip = clip - anchor

    foot_ix = [
        i
        for name in ("LHEE", "RHEE", "LTOE", "RTOE", "LANK", "RANK")
        for i in (_find_marker(labels, name),)
        if i is not None
    ]
    if foot_ix:
        z_min = np.nanmin(clip[:, foot_ix, 2])
        if np.isfinite(z_min):
            clip[:, :, 2] -= z_min - 80.0

    y_mid = np.nanmedian(clip[:, :, 1])
    if np.isfinite(y_mid):
        clip[:, :, 1] -= y_mid - 200.0

    return clip, labels, rate


def _base_pose() -> dict[str, np.ndarray]:
    """Single-frame reference pose (mm), walking toward +X."""
    return {
        "LASI": np.array([10.0, 280.0, 900.0]),
        "RASI": np.array([-8.0, 120.0, 905.0]),
        "LPSI": np.array([8.0, 270.0, 850.0]),
        "RPSI": np.array([-6.0, 130.0, 852.0]),
        "LFHD": np.array([25.0, 500.0, 1150.0]),
        "RFHD": np.array([-22.0, -100.0, 1155.0]),
        "LBHD": np.array([30.0, 520.0, 1120.0]),
        "RBHD": np.array([-25.0, -120.0, 1125.0]),
        "C7": np.array([5.0, 200.0, 1100.0]),
        "CLAV": np.array([0.0, 200.0, 1050.0]),
        "STRN": np.array([0.0, 200.0, 1000.0]),
        "T10": np.array([0.0, 200.0, 980.0]),
        "RBAK": np.array([-5.0, 180.0, 990.0]),
        "LSHO": np.array([18.0, 450.0, 1080.0]),
        "LUPA": np.array([20.0, 430.0, 1020.0]),
        "LELB": np.array([22.0, 420.0, 960.0]),
        "LFRM": np.array([24.0, 410.0, 900.0]),
        "LWRA": np.array([26.0, 400.0, 860.0]),
        "LWRB": np.array([24.0, 395.0, 858.0]),
        "LFIN": np.array([28.0, 390.0, 840.0]),
        "RSHO": np.array([-16.0, -80.0, 1085.0]),
        "RUPA": np.array([-18.0, -70.0, 1025.0]),
        "RELB": np.array([-20.0, -60.0, 965.0]),
        "RFRM": np.array([-22.0, -50.0, 905.0]),
        "RWRA": np.array([-24.0, -40.0, 855.0]),
        "RWRB": np.array([-22.0, -38.0, 853.0]),
        "RFIN": np.array([-26.0, -30.0, 835.0]),
        "LTHI": np.array([12.0, 350.0, 800.0]),
        "LKNE": np.array([14.0, 360.0, 650.0]),
        "LTIB": np.array([16.0, 370.0, 500.0]),
        "LANK": np.array([18.0, 380.0, 380.0]),
        "LHEE": np.array([20.0, 390.0, 120.0]),
        "LTOE": np.array([35.0, 400.0, 115.0]),
        "RTHI": np.array([-10.0, 150.0, 805.0]),
        "RKNE": np.array([-12.0, 140.0, 655.0]),
        "RTIB": np.array([-14.0, 130.0, 505.0]),
        "RANK": np.array([-16.0, 120.0, 385.0]),
        "RHEE": np.array([-18.0, 110.0, 122.0]),
        "RTOE": np.array([-32.0, 100.0, 118.0]),
        "OBSTACLE_L": np.array([0.0, 80.0, 450.0]),
        "OBSTACLE_R": np.array([0.0, 320.0, 450.0]),
    }


def generate_synthetic_walking(
    n_frames: int = 120,
    fs: float = 100.0,
    start_frame: int = 0,
) -> tuple[np.ndarray, list[str], float]:
    """
    Build a procedural (n_frames, n_markers, 3) demo trajectory.

    Anatomical marker names only; no participant identifiers.
    """
    base = _base_pose()
    labels = [
        lab
        for lab in _DEMO_LABELS
        if lab in base and not lab.startswith("OBSTACLE")
    ]
    points = np.full((n_frames, len(labels), 3), np.nan)
    t = np.arange(n_frames) / fs
    gait_hz = 1.1
    walk_speed_mm_s = 450.0
    x0 = 400.0

    for f in range(n_frames):
        phase_l = 2 * np.pi * gait_hz * t[f]
        phase_r = phase_l + np.pi
        dx = x0 + walk_speed_mm_s * t[f]
        swing_l = 0.5 * (1 + np.sin(phase_l))
        swing_r = 0.5 * (1 + np.sin(phase_r))
        foot_lift_l = 180.0 * np.clip(np.sin(phase_l), 0, 1)
        foot_lift_r = 180.0 * np.clip(np.sin(phase_r), 0, 1)
        arm_l = 25.0 * np.sin(phase_l)
        arm_r = 25.0 * np.sin(phase_r + np.pi)
        knee_l = 35.0 * np.clip(np.sin(phase_l), 0, 1)
        knee_r = 35.0 * np.clip(np.sin(phase_r), 0, 1)

        for j, lab in enumerate(labels):
            p = base[lab].copy()
            p[0] += dx - x0
            if lab in ("LHEE", "LTOE", "LANK"):
                p[2] += foot_lift_l
                p[1] += 15.0 * swing_l
            elif lab in ("RHEE", "RTOE", "RANK"):
                p[2] += foot_lift_r
                p[1] -= 15.0 * swing_r
            elif lab in ("LKNE", "LTIB"):
                p[2] -= knee_l * 0.6
                p[1] += 8.0 * swing_l
            elif lab in ("RKNE", "RTIB"):
                p[2] -= knee_r * 0.6
                p[1] -= 8.0 * swing_r
            elif lab in ("LWRA", "LFIN", "LFRM", "LUPA", "LELB"):
                p[1] += arm_l
            elif lab in ("RWRA", "RFIN", "RFRM", "RUPA", "RELB"):
                p[1] -= arm_r
            elif lab in ("LASI", "RASI", "LPSI", "RPSI", "C7", "STRN", "T10"):
                p[2] += 8.0 * np.sin(phase_l)
            points[f, j] = p

    return points, labels, fs


def demo_frame_indices(n_frames: int, fs: float, duration_s: float = 2.5) -> slice:
    """Return slice for a short GIF window."""
    n = min(n_frames, int(round(duration_s * fs)))
    start = max(0, (n_frames - n) // 2)
    return slice(start, start + n)
