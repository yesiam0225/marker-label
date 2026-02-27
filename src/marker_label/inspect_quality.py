"""
Quality inspection for labeled C3D/CSV output.

Computes visibility, velocity continuity, and obstacle stationarity.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_labeled_csv(path: str) -> tuple[np.ndarray, list[str], float]:
    """
    Load a labeled CSV (frame, time, marker_x, marker_y, marker_z per marker).

    Returns
    -------
    points : (n_frames, n_markers, 3), NaN where missing
    labels : list of marker names
    rate : point frame rate (Hz), or 0 if unknown
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    # Header: frame, time, then {label}_x, {label}_y, {label}_z for each marker
    n_frames = len(rows)
    if n_frames == 0:
        return np.empty((0, 0, 3)), [], 0.0
    # Parse marker columns: groups of 3 ending with _x, _y, _z
    i = 2  # skip frame, time
    labels = []
    while i + 2 < len(header):
        xcol = header[i].strip()
        ycol = header[i + 1].strip()
        zcol = header[i + 2].strip()
        if not (xcol.endswith("_x") and ycol.endswith("_y") and zcol.endswith("_z")):
            break
        name = xcol[:-2].strip()  # remove _x
        labels.append(name)
        i += 3
    n_markers = len(labels)
    points = np.full((n_frames, n_markers, 3), np.nan)
    rate = 0.0
    for r, row in enumerate(rows):
        for m in range(n_markers):
            ix, iy, iz = 2 + m * 3, 2 + m * 3 + 1, 2 + m * 3 + 2
            if ix < len(row) and row[ix].strip():
                try:
                    points[r, m, 0] = float(row[ix])
                except ValueError:
                    pass
            if iy < len(row) and row[iy].strip():
                try:
                    points[r, m, 1] = float(row[iy])
                except ValueError:
                    pass
            if iz < len(row) and row[iz].strip():
                try:
                    points[r, m, 2] = float(row[iz])
                except ValueError:
                    pass
    if n_frames > 1:
        try:
            t0, t1 = float(rows[0][1]), float(rows[1][1])
            dt = t1 - t0
            if dt > 0:
                rate = 1.0 / dt
        except (ValueError, TypeError, IndexError):
            pass
    return points, labels, rate


def visibility_per_marker(points: np.ndarray) -> np.ndarray:
    """Fraction of frames with valid (non-NaN) data per marker. Shape (n_markers,)."""
    valid = np.isfinite(points).all(axis=2)
    return np.mean(valid, axis=0)


def velocity_magnitude_per_frame(points: np.ndarray) -> np.ndarray:
    """Per-marker velocity magnitude between consecutive frames. Shape (n_frames-1, n_markers)."""
    if points.shape[0] < 2:
        return np.empty((0, points.shape[1]))
    delta = np.diff(points, axis=0)
    vel = np.linalg.norm(delta, axis=2)
    vel[~np.isfinite(vel)] = np.nan
    return vel


def large_velocity_jumps(
    points: np.ndarray,
    threshold_mm: float = 100.0,
) -> list[tuple[int, int, float]]:
    """
    Find (frame, marker_idx, velocity_mm) where frame-to-frame velocity exceeds threshold.

    Returns list of (frame_index, marker_index, velocity_mm).
    """
    vel = velocity_magnitude_per_frame(points)
    out = []
    for f in range(vel.shape[0]):
        for m in range(vel.shape[1]):
            if np.isfinite(vel[f, m]) and vel[f, m] > threshold_mm:
                out.append((f, m, float(vel[f, m])))
    return out


def obstacle_stationarity(
    points: np.ndarray,
    labels: list[str],
    obstacle_names: tuple[str, ...] = ("OBSTACLE_L", "OBSTACLE_R"),
) -> dict[str, float]:
    """
    Std dev of position over time for obstacle markers (should be small).

    Returns dict label -> std_dev_mm (over all axes).
    """
    result = {}
    for name in obstacle_names:
        try:
            idx = next(i for i, L in enumerate(labels) if L.strip().upper() == name.upper())
        except StopIteration:
            continue
        pos = points[:, idx, :]
        valid = np.isfinite(pos).all(axis=1)
        if valid.sum() < 2:
            result[name] = np.nan
            continue
        std = np.nanstd(pos[valid], axis=0)
        result[name] = float(np.linalg.norm(std))
    return result


def run_quality_report(
    csv_path: str,
    *,
    velocity_threshold_mm: float = 100.0,
    visibility_warn_below: float = 0.80,
    max_jump_reports: int = 20,
) -> dict:
    """
    Load labeled CSV and compute quality metrics.

    Returns
    -------
    dict with keys: points, labels, rate, visibility, velocity_jumps, obstacle_std, summary_lines
    """
    points, labels, rate = load_labeled_csv(csv_path)
    n_frames, n_markers = points.shape[0], points.shape[1]
    visibility = visibility_per_marker(points)
    jumps = large_velocity_jumps(points, threshold_mm=velocity_threshold_mm)
    obstacle_std = obstacle_stationarity(points, labels)
    # Build summary lines
    lines = [
        f"Quality report: {Path(csv_path).name}",
        f"  Frames: {n_frames}, Markers: {n_markers}, Rate: {rate:.2f} Hz",
        "",
        "Visibility (fraction of frames with data):",
    ]
    low_vis = []
    for i, (lab, vis) in enumerate(zip(labels, visibility)):
        status = " (low)" if vis < visibility_warn_below else ""
        lines.append(f"  {lab}: {vis:.2%}{status}")
        if vis < visibility_warn_below:
            low_vis.append((lab, vis))
    lines.append("")
    lines.append(f"Velocity jumps > {velocity_threshold_mm} mm between consecutive frames:")
    if not jumps:
        lines.append("  None.")
    else:
        for (f, m, v) in sorted(jumps, key=lambda x: -x[2])[:max_jump_reports]:
            lines.append(f"  Frame {f} marker {labels[m]}: {v:.1f} mm")
        if len(jumps) > max_jump_reports:
            lines.append(f"  ... and {len(jumps) - max_jump_reports} more.")
    lines.append("")
    lines.append("Obstacle stationarity (std dev of position over time, mm):")
    for name, std in obstacle_std.items():
        lines.append(f"  {name}: {std:.2f} mm")
    return {
        "points": points,
        "labels": labels,
        "rate": rate,
        "visibility": visibility,
        "velocity_jumps": jumps,
        "obstacle_std": obstacle_std,
        "summary_lines": lines,
        "low_visibility": low_vis,
    }


def print_quality_report(csv_path: str, **kwargs) -> None:
    """Load CSV, run quality report, and print to stdout."""
    report = run_quality_report(csv_path, **kwargs)
    for line in report["summary_lines"]:
        print(line)


def main() -> None:
    """CLI entry point for quality inspection."""
    import argparse
    parser = argparse.ArgumentParser(description="Quality inspection for labeled CSV output.")
    parser.add_argument("csv", help="Path to labeled CSV (e.g. *_labeled.csv)")
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=100.0,
        metavar="MM",
        help="Flag velocity jumps above this (mm). Default: 100",
    )
    parser.add_argument(
        "--visibility-warn",
        type=float,
        default=0.80,
        metavar="FRAC",
        help="Warn when visibility below this. Default: 0.80",
    )
    parser.add_argument(
        "--max-jumps",
        type=int,
        default=20,
        metavar="N",
        help="Max number of velocity jumps to list. Default: 20",
    )
    args = parser.parse_args()
    print_quality_report(
        args.csv,
        velocity_threshold_mm=args.velocity_threshold,
        visibility_warn_below=args.visibility_warn,
        max_jump_reports=args.max_jumps,
    )
