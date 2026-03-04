"""Body marker labeling via subject-specific static template and temporal propagation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import DEFAULT_MIDDLE_END, DEFAULT_MIDDLE_START, TRUNK_LABELS
from .pelvis import build_pelvis_frame, points_to_pelvis_frame

_TRUNK_SET = {t.upper() for t in TRUNK_LABELS}


def _trunk_labels_in_template(template: dict) -> list[str]:
    """Return template keys whose label (case-insensitive) is in TRUNK_LABELS."""
    return [k for k in template if str(k).strip().upper() in _TRUNK_SET]


def _rigid_transform_3d(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Procrustes: find R (3x3), t (3,) so that tgt ≈ R @ src + t. src, tgt (n, 3). Returns R, t."""
    n = src.shape[0]
    if n < 3:
        return np.eye(3), np.zeros(3)
    src_c = src - np.nanmean(src, axis=0)
    tgt_c = tgt - np.nanmean(tgt, axis=0)
    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = np.nanmean(tgt, axis=0) - R @ np.nanmean(src, axis=0)
    return R, t


def _apply_rigid_to_template(template: dict[str, np.ndarray], R: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Return new template with each position transformed: pos -> R @ pos + t."""
    out = {}
    for k, pos in template.items():
        if np.isfinite(pos).all():
            out[k] = (R @ pos) + t
        else:
            out[k] = pos.copy()
    return out


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
    *,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
) -> list[tuple[int, str]]:
    """
    Match unlabeled points to template (lab or pelvis frame).

    Parameters
    ----------
    points_frame : (n_points, 3) one frame
    template : dict label -> (3,) position
    use_hungarian : if True, use global optimal assignment (min total distance); else greedy
    max_match_distance : if set, reject assignment when distance > this (mm)

    Returns
    -------
    list of (point_idx, label) for each point assigned
    """
    n_points = points_frame.shape[0]
    labels = list(template.keys())
    positions = np.array([template[l] for l in labels])
    valid_template = np.isfinite(positions).all(axis=1)
    if not valid_template.any():
        return []
    n_labels = len(labels)

    if use_hungarian and n_points >= 1 and n_labels >= 1:
        try:
            # Cost matrix: (n_points, n_labels); inf for invalid
            cost = np.full((n_points, n_labels), np.inf)
            for pi in range(n_points):
                if not np.isfinite(points_frame[pi]).all():
                    continue
                for li in range(n_labels):
                    if not valid_template[li]:
                        continue
                    cost[pi, li] = np.linalg.norm(points_frame[pi] - positions[li])
            # Hungarian: assign labels to points (minimize total cost). Rows=points, cols=labels.
            row_ind, col_ind = linear_sum_assignment(cost)
            assignments = []
            for r, c in zip(row_ind, col_ind):
                d = cost[r, c]
                if not np.isfinite(d) or (max_match_distance is not None and d > max_match_distance):
                    continue
                assignments.append((r, labels[c]))
            return assignments
        except Exception:
            pass  # fall back to greedy

    # Greedy fallback
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
            if d < best_dist and (max_match_distance is None or d <= max_match_distance):
                best_dist = d
                best_label = label
        if best_label is not None:
            used_label.add(best_label)
            assignments.append((pi, best_label))
    return assignments


def best_frame_by_trunk_cost(
    points_dynamic: np.ndarray,
    template: dict[str, np.ndarray],
    residual_dynamic: np.ndarray | None,
    *,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    trunk_labels: list[str] | None = None,
) -> int:
    """
    Choose the frame in the middle range where trunk assignment cost is minimum.

    For each candidate frame we run a provisional full-body match, then sum
    distances only for pairs whose template label is in trunk_labels. The frame
    with minimum trunk cost is returned. Falls back to best_frame_for_matching
    if trunk_labels has fewer than 3 labels.
    """
    if trunk_labels is None:
        trunk_labels = _trunk_labels_in_template(template)
    n_frames = points_dynamic.shape[0]
    start = int(n_frames * middle_start)
    end = int(n_frames * middle_end)
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames))
    candidates = list(range(start, end))
    trunk_set = {str(l).strip().upper() for l in trunk_labels}
    if len(trunk_set) < 3:
        return best_frame_for_matching(
            points_dynamic, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
        )

    def trunk_cost(f: int) -> float:
        asgn = match_markers_to_template(
            points_dynamic[f], template,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
        )
        total = 0.0
        n_trunk = 0
        for pi, lab in asgn:
            if str(lab).strip().upper() not in trunk_set:
                continue
            d = np.linalg.norm(points_dynamic[f, pi] - template[lab])
            if np.isfinite(d):
                total += d
                n_trunk += 1
        if n_trunk < 3:
            return np.inf
        return total

    best = min(candidates, key=trunk_cost)
    if np.isinf(trunk_cost(best)):
        return best_frame_for_matching(
            points_dynamic, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
        )
    return best


def procrustes_fit_trunk(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    assignments: list[tuple[int, str]],
    trunk_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit rigid (R, t) from template to points using only trunk-assigned pairs.
    Returns R, t so that aligned_template = R @ template + t.
    """
    trunk_set = {str(l).strip().upper() for l in trunk_labels}
    src_list, tgt_list = [], []
    for pi, lab in assignments:
        if str(lab).strip().upper() not in trunk_set:
            continue
        pos_t = template.get(lab)
        pos_p = points_frame[pi]
        if pos_t is not None and np.isfinite(pos_t).all() and np.isfinite(pos_p).all():
            src_list.append(pos_t)
            tgt_list.append(pos_p)
    if len(src_list) < 3:
        return np.eye(3), np.zeros(3)
    src = np.array(src_list)
    tgt = np.array(tgt_list)
    return _rigid_transform_3d(src, tgt)


def _trunk_rms(
    points_frame: np.ndarray,
    template: dict[str, np.ndarray],
    assignments: list[tuple[int, str]],
    trunk_labels: list[str],
) -> float:
    """RMS distance for trunk-assigned pairs only. Returns inf if fewer than 3 trunk pairs."""
    trunk_set = {str(l).strip().upper() for l in trunk_labels}
    dists = []
    for pi, lab in assignments:
        if str(lab).strip().upper() not in trunk_set:
            continue
        pos_t = template.get(lab)
        pos_p = points_frame[pi]
        if pos_t is not None and np.isfinite(pos_t).all() and np.isfinite(pos_p).all():
            dists.append(np.linalg.norm(pos_p - pos_t))
    if len(dists) < 3:
        return np.inf
    return float(np.sqrt(np.mean(np.array(dists) ** 2)))


def propagate_labels_temporal(
    points: np.ndarray,
    initial_assignments: list[tuple[int, str]],
    frame_start: int,
    *,
    max_propagation_distance: float | None = None,
) -> np.ndarray:
    """
    Propagate labels forward and backward from frame_start using nearest neighbor.

    Parameters
    ----------
    points : (n_frames, n_points, 3)
    initial_assignments : list of (point_idx, label) at frame_start
    frame_start : frame index where initial assignment was done
    max_propagation_distance : if set, do not assign when nearest distance > this (mm);
        avoids wrong re-acquisition after dropout or cross-over

    Returns
    -------
    label_per_point_per_frame : (n_frames, n_points) dtype object, empty string for unassigned
    """
    n_frames, n_points, _ = points.shape
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
            d = dists[nearest]
            if d < np.inf and (max_propagation_distance is None or d <= max_propagation_distance):
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
            d = dists[nearest]
            if d < np.inf and (max_propagation_distance is None or d <= max_propagation_distance):
                result[f, nearest] = lab
    return result


def label_body_markers(
    points_dynamic: np.ndarray,
    template: dict[str, np.ndarray],
    *,
    residual_dynamic: np.ndarray | None = None,
    middle_start: float = DEFAULT_MIDDLE_START,
    middle_end: float = DEFAULT_MIDDLE_END,
    use_hungarian: bool = True,
    max_match_distance: float | None = None,
    max_propagation_distance: float | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Label body markers in dynamic trial using template and temporal propagation.

    Parameters
    ----------
    points_dynamic : (n_frames, n_points, 3) (remaining points after obstacle removal)
    template : label -> (3,) mean position (same frame as points_dynamic)
    residual_dynamic : optional (n_frames, n_points)
    middle_start, middle_end : frame range for best frame
    use_hungarian : use optimal assignment at best frame (default True)
    max_match_distance : reject initial assignment if distance > this mm (default None)
    max_propagation_distance : do not propagate if nearest > this mm per frame (default None)

    Returns
    -------
    labels_out : list of str, length n_points (order of points_dynamic)
    label_per_frame : (n_frames, n_points) object array, label per point per frame
    """
    n_frames, n_points, _ = points_dynamic.shape
    trunk_labels = _trunk_labels_in_template(template)
    if len(trunk_labels) >= 3:
        best_f = best_frame_by_trunk_cost(
            points_dynamic, template, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
            trunk_labels=trunk_labels,
        )
    else:
        best_f = best_frame_for_matching(
            points_dynamic, residual_dynamic,
            middle_start=middle_start, middle_end=middle_end,
        )
    assignments = match_markers_to_template(
        points_dynamic[best_f],
        template,
        use_hungarian=use_hungarian,
        max_match_distance=max_match_distance,
    )
    # Align template using trunk-only Procrustes, then re-match — unless alignment quality is bad (fallback).
    _TRUNK_RMS_MAX_MM = 150.0  # reject Procrustes if trunk RMS after align exceeds this
    if len(trunk_labels) >= 3 and assignments:
        initial_rms = _trunk_rms(
            points_dynamic[best_f], template, assignments, trunk_labels,
        )
        R, t = procrustes_fit_trunk(
            points_dynamic[best_f], template, assignments, trunk_labels,
        )
        template_aligned = _apply_rigid_to_template(template, R, t)
        assignments_aligned = match_markers_to_template(
            points_dynamic[best_f],
            template_aligned,
            use_hungarian=use_hungarian,
            max_match_distance=max_match_distance,
        )
        aligned_rms = _trunk_rms(
            points_dynamic[best_f], template_aligned, assignments_aligned, trunk_labels,
        )
        if aligned_rms < _TRUNK_RMS_MAX_MM and aligned_rms < initial_rms:
            assignments = assignments_aligned
        # else: keep original assignments (trunk matching failed or did not help)
    if not assignments:
        return [""] * n_points, np.empty((n_frames, n_points), dtype=object)
    label_per_frame = propagate_labels_temporal(
        points_dynamic,
        assignments,
        best_f,
        max_propagation_distance=max_propagation_distance,
    )
    labels_out = [label_per_frame[best_f, pi] for pi in range(n_points)]
    return labels_out, label_per_frame
