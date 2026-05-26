"""C3D file reading and writing."""

from __future__ import annotations

from pathlib import Path

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


def load_c3d(path: str, scale_factor: float = 1.0) -> dict:
    """
    Load a C3D file into a dict with trajectories and metadata.

    Parameters
    ----------
    path : path to C3D file
    scale_factor : multiply point (and residual) coordinates by this after loading.
        Use 1000.0 when the file is in meters and you want mm internally.

    Returns
    -------
    dict with keys:
        points : (n_frames, n_points, 3) float, NaN where missing (in desired unit, e.g. mm)
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
        if scale_factor != 1.0:
            xyz = xyz * scale_factor
            if residual is not None:
                residual = residual * scale_factor
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


def load_labeled_flat_csv_as_c3d_dict(path: str | Path, scale_factor: float = 1.0) -> dict:
    """
    Load a labeled flat CSV (same layout as :func:`marker_label.trial_trim.parse_labeled_csv`)
    into the same dict shape as :func:`load_c3d`.

    Use for **static** (anatomical names per column) and/or **dynamic** (unlabeled ``*`` names ok)
    when the trial is already exported as CSV instead of C3D. Residual is always ``None``.
    """
    from .trial_trim import parse_labeled_csv

    _, meta = parse_labeled_csv(path)
    pts = np.asarray(meta["points"], dtype=np.float64).copy()
    if scale_factor != 1.0:
        pts *= float(scale_factor)
    frames = np.asarray(meta["frames"], dtype=np.int64)
    rate = float(meta.get("rate") or 0.0)
    labels = [str(s) for s in meta["all_stems"]]
    n_frames = int(meta["n_frames"])
    n_points = int(meta["n_markers"])
    first_frame = int(frames[0]) if n_frames > 0 else 1
    return {
        "points": pts,
        "residual": None,
        "labels": labels,
        "point_labels": labels,
        "n_frames": n_frames,
        "n_points": n_points,
        "rate": rate,
        "first_frame": first_frame,
    }


def load_c3d_or_csv(path: str | Path, scale_factor: float = 1.0) -> dict:
    """
    Load a motion trial from ``.csv`` (labeled flat) or ``.c3d`` (binary).

    CSV must match :func:`marker_label.trial_trim.parse_labeled_csv` (frame, time, triplets).
    """
    suf = Path(path).suffix.lower()
    if suf == ".csv":
        return load_labeled_flat_csv_as_c3d_dict(path, scale_factor=scale_factor)
    try:
        return load_c3d(path, scale_factor=scale_factor)
    except AssertionError as e:
        msg = str(e)
        if "magic" in msg.lower() and suf != ".csv":
            raise ValueError(
                f"Not a readable C3D file ({path!r}): {msg}. "
                "If this trial is a labeled flat CSV export, use a path ending in .csv "
                "(frame, time, marker_x/y/z columns per marker_label.trial_trim.parse_labeled_csv)."
            ) from e
        raise


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
        # c3d Writer.add_frames expects sequence of (point, analog) pairs per frame; use empty array for no analog
        empty_analog = np.empty((0, 0))
        frames_data = [(out[i], empty_analog) for i in range(n_frames)]
        writer.add_frames(frames_data)
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
        if hasattr(writer, "set_start_frame"):
            writer.set_start_frame(first_frame)
    except Exception:
        pass
