"""
Quality inspection for labeled C3D/CSV output.

Computes visibility, velocity continuity, and obstacle stationarity.
"""

from __future__ import annotations

import csv

import numpy as np


def load_labeled_csv(path: str) -> tuple[np.ndarray, list[str], float]:
    """
    Load a labeled CSV (frame, time, marker_x, marker_y, marker_z per marker).

    Also supports C3D-style exports (marker name row + Frame/Sub Frame/X/Y/Z header).

    Returns
    -------
    points : (n_frames, n_markers, 3), NaN where missing
    labels : list of marker names
    rate : point frame rate (Hz), or 0 if unknown
    """
    from .trial_trim import parse_labeled_csv

    _, meta = parse_labeled_csv(path)
    points = np.asarray(meta["points"], dtype=np.float64)
    labels = [str(s) for s in meta["all_stems"]]
    rate = float(meta.get("rate") or 0.0)
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
