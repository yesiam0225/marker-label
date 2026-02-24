"""Pelvis body-defined frame from Vicon pelvis markers."""

from __future__ import annotations

import numpy as np

from .constants import PELVIS_MARKERS, PELVIS_MARKER_OPTIONAL


def build_pelvis_frame(
    points: np.ndarray,
    labels: list[str],
    frame_idx: int = 0,
    *,
    pelvis_markers: tuple[str, ...] = PELVIS_MARKERS,
    optional_markers: tuple[str, ...] = PELVIS_MARKER_OPTIONAL,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """
    Build pelvis origin and rotation matrix from pelvis markers at one frame.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    labels : list of str, length n_points
    frame_idx : frame index to use
    pelvis_markers : required marker names (LASI, RASI, LPSI, RPSI)
    optional_markers : optional (e.g. SACR)

    Returns
    -------
    (origin, rotation_3x3, info) or None if insufficient markers.
    origin : (3,) in lab frame
    rotation_3x3 : matrix from lab to pelvis frame; columns are pelvis axes (anterior, lateral, up)
    info : dict with indices used
    """
    label_to_idx = {lab.strip().upper(): i for i, lab in enumerate(labels)}
    pts = points[frame_idx]
    # Collect pelvis marker positions
    found = {}
    for name in pelvis_markers:
        key = name.upper()
        if key in label_to_idx:
            pos = pts[label_to_idx[key]]
            if np.isfinite(pos).all():
                found[name] = pos
    for name in optional_markers:
        key = name.upper()
        if key in label_to_idx:
            pos = pts[label_to_idx[key]]
            if np.isfinite(pos).all():
                found[name] = pos
    # Need at least LASI, RASI, and one posterior (LPSI or RPSI or SACR)
    if "LASI" not in found or "RASI" not in found:
        return None
    mid_asis = (found["LASI"] + found["RASI"]) * 0.5
    # Posterior: SACR or midpoint of LPSI/RPSI
    if "SACR" in found:
        posterior = found["SACR"]
    elif "LPSI" in found and "RPSI" in found:
        posterior = (found["LPSI"] + found["RPSI"]) * 0.5
    elif "LPSI" in found:
        posterior = found["LPSI"]
    elif "RPSI" in found:
        posterior = found["RPSI"]
    else:
        return None
    origin = mid_asis
    # Anterior axis: from posterior to anterior (mid-ASIS)
    anterior = mid_asis - posterior
    anterior_norm = np.linalg.norm(anterior)
    if anterior_norm < 1e-6:
        return None
    anterior = anterior / anterior_norm
    # Lateral: right to left (RASI - LASI), then orthogonalize to anterior
    lateral = found["RASI"] - found["LASI"]
    lateral = lateral - np.dot(lateral, anterior) * anterior
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm < 1e-6:
        return None
    lateral = lateral / lateral_norm
    # Vertical (up): cross(anterior, lateral) or cross(lateral, anterior) for right-handed
    vertical = np.cross(lateral, anterior)
    vertical_norm = np.linalg.norm(vertical)
    if vertical_norm < 1e-6:
        return None
    vertical = vertical / vertical_norm
    # Rotation matrix: columns are pelvis axes (lab to pelvis)
    # So R.T @ p_lab gives p_pelvis (or R columns are anterior, lateral, vertical)
    R = np.column_stack([anterior, lateral, vertical])
    info = {"origin": origin.copy(), "markers_used": list(found.keys())}
    return origin, R, info


def points_to_pelvis_frame(
    points: np.ndarray,
    origin: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Transform points from lab to pelvis frame.

    Parameters
    ----------
    points : (..., 3) in lab frame
    origin : (3,) pelvis origin in lab
    R : (3, 3) rotation, columns = anterior, lateral, vertical

    Returns
    -------
    (..., 3) in pelvis frame: p_pelvis = R.T @ (p_lab - origin)
    """
    pts = np.asarray(points)
    shape = pts.shape
    flat = pts.reshape(-1, 3)
    centered = flat - origin
    # R.T @ x
    out = (R.T @ centered.T).T
    return out.reshape(shape)


def pelvis_frame_per_frame(
    points: np.ndarray,
    labels: list[str],
    *,
    pelvis_markers: tuple[str, ...] = PELVIS_MARKERS,
) -> tuple[np.ndarray, list[tuple | None]]:
    """
    Build pelvis frame for each frame; transform all points to pelvis frame.

    Uses the first frame with valid pelvis markers to define axes, then for each
    frame uses that frame's pelvis origin (so trajectory is in a moving pelvis frame).

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    labels : list of str

    Returns
    -------
    points_pelvis : (n_frames, n_points, 3) in pelvis frame
    frame_info : list of (origin, R, info) or None per frame
    """
    n_frames = points.shape[0]
    # Get rotation from first valid frame (axes definition)
    R_ref = None
    for fi in range(n_frames):
        result = build_pelvis_frame(
            points, labels, fi,
            pelvis_markers=pelvis_markers,
        )
        if result is not None:
            _, R_ref, _ = result
            break
    if R_ref is None:
        return points, [None] * n_frames
    points_pelvis = np.empty_like(points)
    points_pelvis[:] = np.nan
    frame_info = []
    for fi in range(n_frames):
        result = build_pelvis_frame(
            points, labels, fi,
            pelvis_markers=pelvis_markers,
        )
        if result is not None:
            origin, R, info = result
            frame_info.append((origin, R, info))
            pts_f = points[fi]
            points_pelvis[fi] = points_to_pelvis_frame(pts_f, origin, R)
        else:
            frame_info.append(None)
    return points_pelvis, frame_info
