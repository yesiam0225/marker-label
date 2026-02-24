"""Body marker labeling via subject-specific static template and temporal propagation."""

from __future__ import annotations

import numpy as np

from .constants import DEFAULT_MIDDLE_END, DEFAULT_MIDDLE_START
from .pelvis import build_pelvis_frame, points_to_pelvis_frame


def build_template_from_static(
    points_static: np.ndarray,
    labels_static: list[str],
    *,
    use_pelvis_frame: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    """
    Build body template from labeled static trial (lab or pelvis frame).

    Parameters
    ----------
    points_static : (n_frames, n_points, 3) in lab frame
    labels_static : list of str
    use_pelvis_frame : if True, template positions are in pelvis frame; else lab frame

    Returns
    -------
    template : dict label -> (3,) mean position
    origin : (3,) or None
    R : (3,3) or None
    """
    if use_pelvis_frame:
        result = build_pelvis_frame(points_static, labels_static, 0)
        if result is None:
            use_pelvis_frame = False
            origin, R = None, None
        else:
            origin, R, _ = result
            points_static = points_to_pelvis_frame(points_static, origin, R)
    else:
        origin, R = None, None
    # Mean position per label
    template = {}
    label_to_idx = {lab.strip().upper(): i for i, lab in enumerate(labels_static)}
    for label in labels_static:
        key = label.strip().upper()
        idx = label_to_idx[key]
        pos = points_static[:, idx, :]
        valid = np.isfinite(pos).all(axis=1)
        if valid.any():
            template[label] = np.nanmean(pos[valid], axis=0)
        else:
            template[label] = np.array([np.nan, np.nan, np.nan])
    return template, origin, R


def best_frame_for_matching(
    points: np.ndarray,
    residual: np.ndarray | None,
    *,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
) -> int:
    """
    Choose frame index in the middle portion with best quality (lowest mean residual or most valid).

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    residual : (n_frames, n_points) or None
    middle_start, middle_end : fraction of frames for middle portion

    Returns
    -------
    frame_idx : int
    """
    n_frames = points.shape[0]
    start = int(n_frames * middle_start)
    end = int(n_frames * middle_end)
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames))
    candidates = list(range(start, end))
    if residual is not None:
        def score(f):
            s = np.nanmean(residual[f, :])
            return s if np.isfinite(s) else np.inf
        best = min(candidates, key=score)
        return best
    # Fallback: frame with most valid markers
    n_valid = np.sum(np.isfinite(points).all(axis=2), axis=1)
    best = max(candidates, key=lambda f: n_valid[f])
    return best


def match_markers_to_template(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
) -> list[tuple[int, str]]:
    """
    Match unlabeled points to template by nearest neighbor in pelvis frame.

    Parameters
    ----------
    points_frame : (n_points, 3) one frame in pelvis frame
    template : dict label -> (3,) position

    Returns
    -------
    list of (point_idx, label) for each point assigned (greedy nearest)
    """
    n_points = points_frame.shape[0]
    labels = list(template.keys())
    positions = np.array([template[l] for l in labels])
    valid_template = np.isfinite(positions).all(axis=1)
    if not valid_template.any():
        return []
    # Greedy: assign each point to nearest template position, then remove that template
    used_label = set()
    assignments = []
    for pi in range(n_points):
        p = points_frame[pi]
        if not np.isfinite(p).all():
            continue
        best_dist = np.inf
        best_label = None
        for li, label in enumerate(labels):
            if label in used_label or not valid_template[li]:
                continue
            d = np.linalg.norm(p - positions[li])
            if d < best_dist:
                best_dist = d
                best_label = label
        if best_label is not None:
            used_label.add(best_label)
            assignments.append((pi, best_label))
    return assignments


def propagate_labels_temporal(
    points: np.ndarray,
    initial_assignments: list[tuple[int, str]],
    frame_start: int,
) -> np.ndarray:
    """
    Propagate labels forward and backward from frame_start using nearest neighbor.

    Parameters
    ----------
    points : (n_frames, n_points, 3) in pelvis frame
    initial_assignments : list of (point_idx, label) at frame_start
    frame_start : frame index where initial assignment was done

    Returns
    -------
    label_per_point_per_frame : (n_frames, n_points) dtype object, empty string for unassigned
    """
    n_frames, n_points, _ = points.shape
    # Build label list and initial state
    labels_used = [lab for _, lab in initial_assignments]
    point_to_label = {pi: lab for pi, lab in initial_assignments}
    # Array: frame -> point_idx -> label (str or '')
    result = np.empty((n_frames, n_points), dtype=object)
    result[:] = ""
    for pi, lab in initial_assignments:
        result[frame_start, pi] = lab
    # Backward
    for f in range(frame_start - 1, -1, -1):
        for pi in range(n_points):
            if result[f + 1, pi] == "":
                continue
            lab = result[f + 1, pi]
            pos_next = points[f + 1, pi]
            if not np.isfinite(pos_next).all():
                result[f, pi] = lab
                continue
            pos_curr = points[f, :, :]
            valid = np.isfinite(pos_curr).all(axis=1)
            dists = np.linalg.norm(pos_curr - pos_next, axis=1)
            dists[~valid] = np.inf
            nearest = np.argmin(dists)
            if dists[nearest] < np.inf:
                result[f, nearest] = lab
    # Forward
    for f in range(frame_start + 1, n_frames):
        for pi in range(n_points):
            if result[f - 1, pi] == "":
                continue
            lab = result[f - 1, pi]
            pos_prev = points[f - 1, pi]
            if not np.isfinite(pos_prev).all():
                result[f, pi] = lab
                continue
            pos_curr = points[f, :, :]
            valid = np.isfinite(pos_curr).all(axis=1)
            dists = np.linalg.norm(pos_curr - pos_prev, axis=1)
            dists[~valid] = np.inf
            nearest = np.argmin(dists)
            if dists[nearest] < np.inf:
                result[f, nearest] = lab
    return result


def label_body_markers(
    points_dynamic: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    residual_dynamic: np.ndarray | None = None,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
) -> tuple[list[str], np.ndarray]:
    """
    Label body markers in dynamic trial using template and temporal propagation.

    Matching and propagation are done in the same coordinate frame as the template
    (lab frame or pelvis frame).

    Parameters
    ----------
    points_dynamic : (n_frames, n_points, 3) (remaining points after obstacle removal)
    template : label -> (3,) mean position (same frame as points_dynamic)
    residual_dynamic : optional (n_frames, n_points)
    middle_start, middle_end : frame range for best frame

    Returns
    -------
    labels_out : list of str, length n_points (order of points_dynamic)
    label_per_frame : (n_frames, n_points) object array, label per point per frame
    """
    n_frames, n_points, _ = points_dynamic.shape
    best_f = best_frame_for_matching(
        points_dynamic, residual_dynamic,
        middle_start=middle_start, middle_end=middle_end,
    )
    assignments = match_markers_to_template(points_dynamic[best_f], template)
    if not assignments:
        return [""] * n_points, np.empty((n_frames, n_points), dtype=object)
    label_per_frame = propagate_labels_temporal(points_dynamic, assignments, best_f)
    labels_out = [label_per_frame[best_f, pi] for pi in range(n_points)]
    return labels_out, label_per_frame
