"""
Analyze the relationship between static and dynamic trials using a reference subject
with manually labeled static and dynamic C3D files.

Use this to:
- Fit per-frame rigid transform (rotation R, translation t) from static template to dynamic.
- See which dynamic frame is closest to the template (true best frame).
- Compare with pipeline's best-frame choice (residual or most-valid).
- Inspect rotation/translation magnitude and variation across frames.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .body_labeling import build_template_from_static, best_frame_for_matching
from .io import load_c3d


def _rigid_transform_3d(
    src: np.ndarray,
    tgt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find R (3x3 orthogonal) and t (3,) such that tgt ≈ R @ src + t.
    src, tgt: (n, 3). Returns R, t, and RMS residual (mm).
    """
    assert src.shape == tgt.shape and src.ndim == 2 and src.shape[1] == 3
    n = src.shape[0]
    if n < 3:
        return np.eye(3), np.zeros(3), np.nan
    src_centered = src - np.nanmean(src, axis=0)
    tgt_centered = tgt - np.nanmean(tgt, axis=0)
    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = np.nanmean(tgt, axis=0) - R @ np.nanmean(src, axis=0)
    pred = (R @ src.T).T + t
    diff = tgt - pred
    rms = np.sqrt(np.nanmean(diff ** 2)) if np.any(np.isfinite(diff)) else np.nan
    return R, t, float(rms)


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Angle in degrees of rotation from identity."""
    trace = np.trace(R)
    trace = np.clip(trace, -1.0, 3.0)
    angle_rad = math.acos((trace - 1) / 2)
    return math.degrees(angle_rad)


def get_dynamic_positions_by_label(
    points_d: np.ndarray,
    labels_d: list[str],
) -> list[dict[str, np.ndarray]]:
    """
    For each frame, return dict label -> (3,) position (NaN if missing).
    points_d: (n_frames, n_points, 3), labels_d: length n_points.
    """
    n_frames = points_d.shape[0]
    out = []
    for f in range(n_frames):
        d = {}
        for i, lab in enumerate(labels_d):
            d[lab.strip()] = points_d[f, i, :].copy()
        out.append(d)
    return out


def analyze_static_dynamic_relationship(
    static_path: str,
    dynamic_path: str,
    *,
    sample_every: int = 1,
    use_pelvis_frame: bool = False,
    static_scale: float = 1.0,
    dynamic_scale: float = 1.0,
) -> dict:
    """
    Analyze relationship between static template and labeled dynamic trial.

    Parameters
    ----------
    static_path : path to labeled static C3D
    dynamic_path : path to labeled dynamic C3D (same subject, manually labeled)
    sample_every : analyze every N frames (default 1 = all)
    use_pelvis_frame : build template in pelvis frame (default False)
    static_scale : multiply static coordinates after load (e.g. 1000 if file is in m)
    dynamic_scale : multiply dynamic coordinates after load (e.g. 1000 if file is in m)

    Returns
    -------
    dict with:
        template_labels : list of labels in template
        n_frames_dynamic : int
        n_frames_analyzed : int (after sampling)
        per_frame : list of {frame, rotation_deg, translation_mm, rms_mm, n_pairs}
        frame_closest_to_template : frame index with minimum RMS
        pipeline_best_frame : frame index from best_frame_for_matching (middle, residual/most-valid)
        summary : {mean_rotation_deg, mean_translation_mm, mean_rms_mm, ...}
    """
    static = load_c3d(static_path, scale_factor=static_scale)
    dynamic = load_c3d(dynamic_path, scale_factor=dynamic_scale)
    points_s = static["points"]
    labels_s = static["labels"]
    points_d = dynamic["points"]
    labels_d = dynamic["labels"]
    residual_d = dynamic.get("residual")
    n_frames_d = points_d.shape[0]
    if not labels_s or not labels_d:
        raise ValueError("Both static and dynamic must have point labels.")

    # Template from static (mean per label)
    template, _, _ = build_template_from_static(
        points_s, labels_s, use_pelvis_frame=use_pelvis_frame
    )
    template_labels = [l for l in labels_s if np.isfinite(template.get(l, np.nan)).all()]
    if not template_labels:
        raise ValueError("No valid template positions from static.")

    # Dynamic: positions by label per frame (use dynamic's labels and point order)
    # Build label -> point index for dynamic
    label_d_to_idx = {str(l).strip().upper(): i for i, l in enumerate(labels_d)}
    # Common labels (exist in both template and dynamic)
    common = [l for l in template_labels if str(l).strip().upper() in label_d_to_idx]
    if len(common) < 3:
        raise ValueError("Need at least 3 common labels between static and dynamic.")

    # Pipeline's best frame (on full dynamic, as pipeline would see it)
    pipeline_best = best_frame_for_matching(points_d, residual_d)

    # Per-frame: get positions for common labels, fit R and t, compute RMS
    frames_to_analyze = list(range(0, n_frames_d, sample_every))
    if frames_to_analyze[-1] != n_frames_d - 1 and n_frames_d > 1:
        frames_to_analyze.append(n_frames_d - 1)

    per_frame = []
    for f in frames_to_analyze:
        src_list = []
        tgt_list = []
        for lab in common:
            tidx = next(i for i, L in enumerate(labels_s) if str(L).strip().upper() == str(lab).strip().upper())
            didx = label_d_to_idx[str(lab).strip().upper()]
            pos_d = points_d[f, didx, :]
            if not np.isfinite(pos_d).all():
                continue
            src_list.append(template[labels_s[tidx]])
            tgt_list.append(pos_d)
        if len(src_list) < 3:
            per_frame.append({
                "frame": f,
                "rotation_deg": np.nan,
                "translation_mm": np.nan,
                "rms_mm": np.nan,
                "n_pairs": len(src_list),
            })
            continue
        src = np.array(src_list)
        tgt = np.array(tgt_list)
        R, t, rms = _rigid_transform_3d(src, tgt)
        angle_deg = _rotation_angle_deg(R)
        t_norm = float(np.linalg.norm(t))
        per_frame.append({
            "frame": f,
            "rotation_deg": angle_deg,
            "translation_mm": t_norm,
            "rms_mm": rms,
            "n_pairs": len(src_list),
        })

    # Frame closest to template (min RMS)
    valid = [p for p in per_frame if np.isfinite(p["rms_mm"])]
    if not valid:
        frame_closest = 0
    else:
        frame_closest = min(valid, key=lambda x: x["rms_mm"])["frame"]

    # Summary stats (over analyzed frames)
    rotations = [p["rotation_deg"] for p in per_frame if np.isfinite(p["rotation_deg"])]
    translations = [p["translation_mm"] for p in per_frame if np.isfinite(p["translation_mm"])]
    rms_list = [p["rms_mm"] for p in per_frame if np.isfinite(p["rms_mm"])]
    summary = {
        "mean_rotation_deg": float(np.mean(rotations)) if rotations else np.nan,
        "max_rotation_deg": float(np.max(rotations)) if rotations else np.nan,
        "mean_translation_mm": float(np.mean(translations)) if translations else np.nan,
        "max_translation_mm": float(np.max(translations)) if translations else np.nan,
        "mean_rms_mm": float(np.mean(rms_list)) if rms_list else np.nan,
        "min_rms_mm": float(np.min(rms_list)) if rms_list else np.nan,
    }

    return {
        "static_path": str(Path(static_path).resolve()),
        "dynamic_path": str(Path(dynamic_path).resolve()),
        "template_labels": template_labels,
        "common_labels_count": len(common),
        "n_frames_dynamic": n_frames_d,
        "n_frames_analyzed": len(per_frame),
        "per_frame": per_frame,
        "frame_closest_to_template": int(frame_closest),
        "pipeline_best_frame": int(pipeline_best),
        "summary": summary,
    }


def print_analysis_report(result: dict) -> None:
    """Print a human-readable summary of the analysis."""
    s = result["summary"]
    print("=== Static vs dynamic relationship (reference subject) ===\n")
    print(f"Static:  {result['static_path']}")
    print(f"Dynamic: {result['dynamic_path']}")
    print(f"Common labels: {result['common_labels_count']}  (template has {len(result['template_labels'])} labels)")
    print(f"Dynamic frames: {result['n_frames_dynamic']}  (analyzed {result['n_frames_analyzed']} frames)\n")
    print("Per-frame rigid transform (template -> dynamic):")
    print(f"  Mean rotation:    {s['mean_rotation_deg']:.2f} deg")
    print(f"  Max rotation:     {s['max_rotation_deg']:.2f} deg")
    print(f"  Mean translation: {s['mean_translation_mm']:.1f} mm")
    print(f"  Max translation:  {s['max_translation_mm']:.1f} mm")
    print(f"  Mean RMS error:   {s['mean_rms_mm']:.2f} mm  (after fit)")
    print(f"  Min RMS error:    {s['min_rms_mm']:.2f} mm\n")
    print("Best frame:")
    print(f"  Frame closest to template (min RMS):  {result['frame_closest_to_template']}")
    print(f"  Pipeline choice (residual/most-valid): {result['pipeline_best_frame']}")
    if result["frame_closest_to_template"] != result["pipeline_best_frame"]:
        print("  --> Mismatch: pipeline would use a different frame than the one that best matches the template.")
    else:
        print("  --> Match: pipeline's best frame agrees with closest-to-template frame.")
    print()


def suggested_pipeline_params_from_reference(
    report: dict,
    *,
    window_frac: float = 0.15,
    max_match_multiplier: float = 3.0,
    max_propagation_multiplier: float = 2.0,
    max_match_cap_mm: float = 500.0,
    max_propagation_cap_mm: float = 200.0,
) -> dict:
    """
    Suggest pipeline parameters from a reference analysis report.

    Uses the reference's "frame closest to template" position and RMS summary
    to suggest middle_start, middle_end (best-frame search window) and
    optionally max_match_distance, max_propagation_distance.

    Parameters
    ----------
    report : dict from analyze_static_dynamic_relationship (or loaded from -o JSON)
    window_frac : half-width of the best-frame window around reference best (default 0.15 → ±15%)
    max_match_multiplier : suggested max_match_distance = this * min_rms_mm (capped)
    max_propagation_multiplier : suggested max_propagation_distance = this * min_rms_mm (capped)
    max_match_cap_mm, max_propagation_cap_mm : caps for suggested distances

    Returns
    -------
    dict with keys:
        middle_start, middle_end : float in [0, 1]
        max_match_distance : float or None if min_rms not usable
        max_propagation_distance : float or None if min_rms not usable
    """
    n_frames = report.get("n_frames_dynamic") or 0
    frame_closest = report.get("frame_closest_to_template", 0)
    summary = report.get("summary") or {}
    min_rms = summary.get("min_rms_mm")
    if n_frames > 1:
        ref_frac = frame_closest / (n_frames - 1)
    else:
        ref_frac = 0.5
    middle_start = max(0.0, ref_frac - window_frac)
    middle_end = min(1.0, ref_frac + window_frac)
    # Ensure non-degenerate window
    if middle_end - middle_start < 0.05:
        middle_start = max(0.0, ref_frac - 0.1)
        middle_end = min(1.0, ref_frac + 0.1)

    out = {"middle_start": middle_start, "middle_end": middle_end}

    # Optional distance thresholds from reference RMS
    if min_rms is not None and np.isfinite(min_rms) and 0 < min_rms < 1000:
        max_match = min(max_match_cap_mm, max_match_multiplier * min_rms)
        max_prop = min(max_propagation_cap_mm, max_propagation_multiplier * min_rms)
        out["max_match_distance"] = max(max_match, 20.0)
        out["max_propagation_distance"] = max(max_prop, 10.0)
    else:
        out["max_match_distance"] = None
        out["max_propagation_distance"] = None

    return out


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze static–dynamic relationship using a reference subject with manually labeled static and dynamic C3D.",
    )
    parser.add_argument("static", help="Path to labeled static C3D (reference)")
    parser.add_argument("dynamic", help="Path to labeled dynamic C3D (reference)")
    parser.add_argument("--sample", type=int, default=1, metavar="N", help="Analyze every N frames (default 1)")
    parser.add_argument("--pelvis-frame", action="store_true", help="Build template in pelvis frame")
    parser.add_argument("--static-unit", choices=("mm", "m"), default="mm", help="Unit of static C3D: mm or m (default mm)")
    parser.add_argument("--dynamic-unit", choices=("mm", "m"), default="mm", help="Unit of dynamic C3D: mm or m (default mm)")
    parser.add_argument("-o", "--output", default=None, help="Write full result JSON to this path")
    args = parser.parse_args()
    unit_scale = {"mm": 1.0, "m": 1000.0}
    result = analyze_static_dynamic_relationship(
        args.static,
        args.dynamic,
        sample_every=args.sample,
        use_pelvis_frame=args.pelvis_frame,
        static_scale=unit_scale[args.static_unit],
        dynamic_scale=unit_scale[args.dynamic_unit],
    )
    print_analysis_report(result)
    if args.output:
        def _json_default(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if np.isfinite(obj) else None
            raise TypeError(type(obj))
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=_json_default)
        print(f"Full result written to {args.output}")
