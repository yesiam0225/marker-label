"""C3D file reading and writing."""

from __future__ import annotations

import numpy as np

try:
    import c3d
except ImportError as e:
    raise ImportError("Install the c3d package: pip install c3d") from e

# C3D point array: columns are typically x, y, z, residual, camera_mask
# Missing points are often (0, 0, 0) or NaN
N_COLS_POINT = 5


def _points_to_xyz_residual(points: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract (n_frames, n_points, 3) xyz and optional (n_frames, n_points) residual."""
    if points.ndim == 2:
        points = points[np.newaxis, ...]
    n_frames, n_points, n_cols = points.shape
    xyz = points[:, :, :3].copy()
    # C3D often uses 0 for missing; convert to NaN for consistency
    invalid = (points[:, :, 0] == 0) & (points[:, :, 1] == 0) & (points[:, :, 2] == 0)
    xyz[invalid] = np.nan
    residual = None
    if n_cols >= 4:
        residual = points[:, :, 3].copy()
        residual[invalid] = np.nan
    return xyz, residual


def load_c3d(path: str) -> dict:
    """
    Load a C3D file into a dict with trajectories and metadata.

    Returns
    -------
    dict with keys:
        points : (n_frames, n_points, 3) float, NaN where missing
        residual : (n_frames, n_points) float or None
        labels : list of str, length n_points (may be empty or generic)
        point_labels : same as labels (for compatibility)
        n_frames : int
        n_points : int
        rate : float, point frame rate (Hz)
        first_frame : int
    """
    with open(path, "rb") as f:
        reader = c3d.Reader(f)
        all_points = []
        all_analog = []
        for frame_i, pts, analog in reader.read_frames():
            all_points.append(pts)
            all_analog.append(analog)
        points_stack = np.array(all_points)
        # points_stack: (n_frames, n_points, 5) typically
        xyz, residual = _points_to_xyz_residual(points_stack)
        # Point labels: reader may have point_labels in parameter group
        labels = []
        try:
            if hasattr(reader, "point_labels"):
                labels = list(reader.point_labels)
            elif getattr(reader, "point_parameter", None) is not None:
                params = reader.point_parameter
                if "LABELS" in params:
                    labels = [s.strip() for s in params["LABELS"]]
        except Exception:
            pass
        if not labels and xyz.shape[1] > 0:
            labels = [f"Point_{i}" for i in range(xyz.shape[1])]
        rate = getattr(reader, "point_rate", None) or 0.0
        first_frame = int(getattr(reader, "first_frame", 1))
        return {
            "points": xyz,
            "residual": residual,
            "labels": labels,
            "point_labels": labels,
            "n_frames": xyz.shape[0],
            "n_points": xyz.shape[1],
            "rate": rate,
            "first_frame": first_frame,
        }


def save_c3d(
    path: str,
    points: np.ndarray,
    labels: list[str],
    rate: float = 0.0,
    first_frame: int = 1,
    residual: np.ndarray | None = None,
) -> None:
    """
    Write a C3D file with 3D point data and labels.

    Parameters
    ----------
    path : output file path
    points : (n_frames, n_points, 3) or (n_frames, n_points, 4+) with residual in 4th column
    labels : list of str, length n_points
    rate : point frame rate (Hz)
    first_frame : first frame index
    residual : optional (n_frames, n_points)
    """
    n_frames, n_points = points.shape[0], points.shape[1]
    if points.shape[2] >= 4 and residual is None:
        residual = points[:, :, 3]
        xyz = points[:, :, :3]
    else:
        xyz = points[:, :, :3].copy()
    # C3D Writer expects (n_points, 5) per frame: x, y, z, residual, camera_mask
    # Use 0 for missing (some readers expect 0,0,0 for invalid)
    nan_mask = np.isnan(xyz).any(axis=2)
    out = np.zeros((n_frames, n_points, N_COLS_POINT), dtype=np.float32)
    out[:, :, :3] = np.where(np.expand_dims(nan_mask, 2), 0.0, xyz)
    if residual is not None:
        out[:, :, 3] = np.where(nan_mask, 0.0, residual)
    else:
        out[:, :, 3] = 0.0
    out[:, :, 4] = -1  # camera mask: -1 often means no mask

    with open(path, "wb") as f:
        writer = c3d.Writer()
        for i in range(n_frames):
            # add_frame expects (n_points, 5) per frame
            writer.add_frame(out[i])
        # Copy point labels into writer's parameter block if supported
        _set_writer_point_labels(writer, labels, n_points)
        _set_writer_rate_first_frame(writer, rate, first_frame)
        writer.write(f)


def _set_writer_point_labels(writer, labels: list[str], n_points: int) -> None:
    """Set POINT:LABELS in writer parameter block if the library supports it."""
    try:
        if hasattr(writer, "set_point_labels"):
            writer.set_point_labels(labels)
        elif hasattr(writer, "point_labels"):
            writer.point_labels = labels[:n_points]
    except Exception:
        pass


def _set_writer_rate_first_frame(writer, rate: float, first_frame: int) -> None:
    """Set point rate and first frame if supported."""
    try:
        if hasattr(writer, "set_point_rate"):
            writer.set_point_rate(rate)
        if hasattr(writer, "set_first_frame"):
            writer.set_first_frame(first_frame)
    except Exception:
        pass
