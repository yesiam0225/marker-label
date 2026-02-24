"""Full labeling pipeline: static + dynamic -> labeled C3D and CSV."""

from __future__ import annotations

import numpy as np

from .io import load_c3d
from .constants import PELVIS_MARKERS, DEFAULT_OBSTACLE_VISIBILITY_MIN
from .obstacle import detect_obstacle_markers
from .body_labeling import build_template_from_static, label_body_markers
from .export import build_full_trajectory_matrix, export_labeled


def run_pipeline(
    static_path: str,
    dynamic_path: str,
    out_prefix: str,
    *,
    obstacle_visibility_min: float = DEFAULT_OBSTACLE_VISIBILITY_MIN,
    use_pelvis_frame: bool = False,
    middle_start: float = 0.20,
    middle_end: float = 0.80,
    export_filled: bool = True,
    max_interp_frames: int = 10,
) -> dict:
    """
    Run the full labeling pipeline.

    Parameters
    ----------
    static_path : path to labeled static C3D
    dynamic_path : path to unlabeled dynamic C3D
    out_prefix : output path prefix (e.g. "out/trial_01")
    obstacle_visibility_min : min visibility for obstacle candidates
    use_pelvis_frame : build template in pelvis frame (default False = lab frame)
    middle_start, middle_end : frame range for best-frame selection
    export_filled : also export filled C3D/CSV
    max_interp_frames : gap-fill threshold

    Returns
    -------
    info : dict with keys (static_labels, obstacle_indices, etc.)
    """
    static = load_c3d(static_path)
    dynamic = load_c3d(dynamic_path)
    points_s = static["points"]
    labels_s = static["labels"]
    points_d = dynamic["points"]
    residual_d = dynamic.get("residual")
    rate = dynamic.get("rate") or 0.0
    first_frame = dynamic.get("first_frame") or 1

    # 1) Obstacle detection on dynamic
    obstacle_indices, obstacle_labels = detect_obstacle_markers(
        points_d,
        visibility_min=obstacle_visibility_min,
    )
    n_frames_d = points_d.shape[0]
    # Extract obstacle trajectories; default (n_frames, 2, 3) NaN if none
    if len(obstacle_indices) >= 2:
        points_obstacle = points_d[:, obstacle_indices[:2], :].copy()
    elif len(obstacle_indices) == 1:
        points_obstacle = np.full((n_frames_d, 2, 3), np.nan)
        points_obstacle[:, 0, :] = points_d[:, obstacle_indices[0], :]
    else:
        points_obstacle = np.full((n_frames_d, 2, 3), np.nan)
    # Remove obstacle points for body labeling
    if obstacle_indices:
        keep = [i for i in range(points_d.shape[1]) if i not in obstacle_indices]
        points_d_body = points_d[:, keep, :]
        residual_d_body = residual_d[:, keep] if residual_d is not None else None
    else:
        points_d_body = points_d
        residual_d_body = residual_d

    # 2) Template from static (exclude obstacle labels if present)
    body_labels_static = [l for l in labels_s if l not in ("OBSTACLE_L", "OBSTACLE_R")]
    if not body_labels_static:
        body_labels_static = list(labels_s)
    template, _origin, _R = build_template_from_static(
        points_s, labels_s, use_pelvis_frame=use_pelvis_frame
    )
    # Filter template to body only
    template_body = {k: v for k, v in template.items() if k in body_labels_static}

    # 3) Body labeling
    labels_body_out, label_per_frame = label_body_markers(
        points_d_body,
        template_body,
        residual_dynamic=residual_d_body,
        middle_start=middle_start,
        middle_end=middle_end,
    )

    # 4) Build full output: body (static order) + obstacle
    if points_obstacle.shape[1] < 2:
        obs_fill = np.full((points_d.shape[0], 2, 3), np.nan)
        obs_fill[:, : points_obstacle.shape[1], :] = points_obstacle
        points_obstacle = obs_fill
    obs_labels = ["OBSTACLE_L", "OBSTACLE_R"]
    if obstacle_indices and len(obstacle_labels) == 2:
        obs_labels = obstacle_labels
    points_full, labels_full = build_full_trajectory_matrix(
        body_labels_static,
        points_d_body,
        label_per_frame,
        points_obstacle,
        obs_labels,
    )

    # 5) Export original + filled
    export_labeled(
        points_full,
        labels_full,
        out_prefix,
        rate=rate,
        first_frame=first_frame,
        export_filled=export_filled,
        max_interp_frames=max_interp_frames,
    )

    return {
        "static_labels": labels_s,
        "body_labels_static": body_labels_static,
        "obstacle_indices": obstacle_indices,
        "obstacle_labels": obs_labels,
        "labels_out": labels_full,
        "n_frames": points_full.shape[0],
        "n_markers": points_full.shape[1],
    }
