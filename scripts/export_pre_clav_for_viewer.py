"""
Export "before CLAV" or "through CLAV" state for viewer: labels assigned up to head + C7/shoulders
(optionally + CLAV/RBAK, etc.), unlabeled points shown by **loaded C3D column index** (0-based,
before initial screening; matches ``load_c3d`` point order).

Usage (from repo root, with package installed or PYTHONPATH=src):
  # Pre-CLAV (head + C7/shoulders only):
  python scripts/export_pre_clav_for_viewer.py static.c3d dynamic.c3d \\
    --first-frame 169 --last-frame 752 --skip-visibility-screening --no-check-screened-count \\
    -o out/pre_clav.csv [--viewer]

  # Through CLAV/RBAK (head + C7/shoulders + CLAV + RBAK; before STRN, T10, arm/elbow):
  python scripts/export_pre_clav_for_viewer.py static.c3d dynamic.c3d ... -o out/pre_strn.csv --through-clav [--viewer]

  # Through STRN/T10/arm4 (before forearm, pelvis, wrist, finger):
  python scripts/export_pre_clav_for_viewer.py static.c3d dynamic.c3d ... -o out/pre_pelvis_arm12.csv --through-strn-t10-arm [--viewer]

  # Through pelvis/arm12 (before thigh, knee, tibia, foot):
  python scripts/export_pre_clav_for_viewer.py static.c3d dynamic.c3d ... -o out/pre_leg_foot12.csv --through-pelvis-arm12 [--viewer]

Then open the CSV in the QC viewer to see anatomical labels and point numbers for unlabeled candidates.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pre-CLAV state (head + C7/shoulders labeled, rest as point index) for QC viewer.",
    )
    parser.add_argument("static", help="Path to labeled static C3D")
    parser.add_argument("dynamic", help="Path to dynamic C3D")
    parser.add_argument(
        "-o", "--output",
        default="out/pre_clav_labeled.csv",
        help="Output CSV path (default: out/pre_clav_labeled.csv)",
    )
    parser.add_argument("--first-frame", type=int, default=None, help="Step 1.5: first frame to keep (1-based)")
    parser.add_argument("--last-frame", type=int, default=None, help="Step 1.5: last frame to keep (1-based)")
    parser.add_argument("--skip-visibility-screening", action="store_true", help="Skip Step 2 (visibility)")
    parser.add_argument("--no-check-screened-count", action="store_true", help="Do not require 41 columns")
    parser.add_argument("--through-clav", action="store_true", help="Include CLAV/RBAK assignment (stop before STRN/T10/arm)")
    parser.add_argument("--through-strn-t10-arm", action="store_true", help="Include STRN, T10, LUPA, RUPA, LELB, RELB (stop before forearm/pelvis/wrist/finger)")
    parser.add_argument("--through-pelvis-arm12", action="store_true", help="Include pelvis + forearm/wrist/finger (LASI,RASI,LPSI,RPSI,LFRM,LWRA,LWRB,LFIN,RFRM,RWRA,RWRB,RFIN); stop before leg/foot")
    parser.add_argument("--viewer", action="store_true", help="Launch QC viewer on the exported CSV after export")
    args = parser.parse_args()

    from marker_label.io import load_c3d
    from marker_label.constants import (
        DEFAULT_OBSTACLE_VISIBILITY_MIN,
        EXPECTED_SCREENED_MARKERS,
        UNLABELED_NUMERIC_LABEL_BASE,
    )
    import numpy as np
    from marker_label.screening import (
        apply_initial_screening_steps_1_and_2,
        compute_walking_direction_x,
        get_lr_ap_axes_from_walking,
        loaded_point_indices_for_body,
        ScreeningError,
    )
    from marker_label.obstacle import detect_obstacle_markers
    from marker_label.body_labeling import build_template_from_static, run_labeling_until_before_clav, get_clav_rbak_candidate_indices, assign_clav_rbak_after_c7_shoulders, assign_strn_t10_arm4_after_clav_rbak, assign_pelvis_arm12_after_strn_t10_arm, CLAVRBAKValidationError
    from marker_label.export import export_csv

    static = load_c3d(args.static)
    dynamic = load_c3d(args.dynamic)
    points_s = static["points"]
    labels_s = static["labels"]
    points_d = dynamic["points"]
    residual_d = dynamic.get("residual")
    rate = dynamic.get("rate") or 0.0
    first_frame = dynamic.get("first_frame") or 1

    points_d, residual_d, screening_keep, frame_trim_start = apply_initial_screening_steps_1_and_2(
        points_d, residual_d,
        trim_first_frame=args.first_frame,
        trim_last_frame=args.last_frame,
        skip_visibility_step=args.skip_visibility_screening,
    )
    if args.first_frame is not None and args.last_frame is not None:
        first_frame = args.first_frame
    else:
        first_frame = (first_frame or 1) + frame_trim_start

    n_screened = points_d.shape[1]
    if not args.no_check_screened_count and n_screened != EXPECTED_SCREENED_MARKERS:
        raise ScreeningError(
            2,
            f"expected {EXPECTED_SCREENED_MARKERS} columns after screening, got {n_screened}. Use --no-check-screened-count to continue.",
        )

    wdx = compute_walking_direction_x(points_d)
    d_back, d_right = get_lr_ap_axes_from_walking(wdx)

    obstacle_indices, obstacle_labels = detect_obstacle_markers(
        points_d,
        visibility_min=DEFAULT_OBSTACLE_VISIBILITY_MIN,
        lr_axis="y",
    )
    if obstacle_indices:
        keep = [i for i in range(points_d.shape[1]) if i not in obstacle_indices]
        points_d_body = points_d[:, keep, :]
        residual_d_body = residual_d[:, keep] if residual_d is not None else None
        body_to_original = loaded_point_indices_for_body(screening_keep, keep)
        # Keep obstacle trajectories for export (so viewer can show them)
        points_obstacle = points_d[:, obstacle_indices[:2], :].copy()
        if len(obstacle_indices) == 1:
            points_obstacle = np.concatenate([
                points_obstacle,
                np.full((points_d.shape[0], 1, 3), np.nan),
            ], axis=1)
        obs_labels = list(obstacle_labels) if len(obstacle_labels) >= 2 else ["OBSTACLE_L", "OBSTACLE_R"]
    else:
        keep = list(range(points_d.shape[1]))
        points_d_body = points_d
        residual_d_body = residual_d
        body_to_original = loaded_point_indices_for_body(screening_keep, keep)
        points_obstacle = np.full((points_d.shape[0], 2, 3), np.nan)
        obs_labels = ["OBSTACLE_L", "OBSTACLE_R"]

    body_labels_static = [l for l in labels_s if str(l).strip().upper() not in ("OBSTACLE_L", "OBSTACLE_R")]
    if not body_labels_static:
        body_labels_static = list(labels_s)
    template, _, _ = build_template_from_static(points_s, labels_s, use_pelvis_frame=False)
    template_body = {k: v for k, v in template.items() if k in body_labels_static}

    best_f, head_assignments, c7_shoulder_assignments = run_labeling_until_before_clav(
        points_d_body,
        template_body if template_body else template,
        d_back,
        d_right,
        residual_dynamic=residual_d_body,
    )

    head_pt_set = {pi for pi, _ in head_assignments}
    priority_pt_set = head_pt_set | {pi for pi, _ in c7_shoulder_assignments}
    lsho_idx = next((pi for pi, lab in c7_shoulder_assignments if str(lab).strip().upper() == "LSHO"), -1)
    rsho_idx = next((pi for pi, lab in c7_shoulder_assignments if str(lab).strip().upper() == "RSHO"), -1)

    clav_rbak_assignments: list[tuple[int, str]] = []
    run_clav = args.through_clav or args.through_strn_t10_arm or args.through_pelvis_arm12
    if run_clav and lsho_idx >= 0 and rsho_idx >= 0:
        try:
            clav_rbak_assignments = assign_clav_rbak_after_c7_shoulders(
                points_d_body[best_f],
                d_back,
                d_right,
                priority_pt_set,
                lsho_idx,
                rsho_idx,
                template_body if template_body else template,
            )
        except CLAVRBAKValidationError as e:
            print(f"CLAV/RBAK validation failed: {e}")
            clav_cand, rbak_cand = get_clav_rbak_candidate_indices(
                points_d_body[best_f], priority_pt_set, lsho_idx, rsho_idx,
            )
            if clav_cand is not None:
                clav_rbak_assignments.append((clav_cand, "CLAV_cand"))
            if rbak_cand is not None:
                clav_rbak_assignments.append((rbak_cand, "RBAK_cand"))
            print(f"  Exported candidates as CLAV_cand / RBAK_cand for viewer.")

    strn_t10_arm_assignments: list[tuple[int, str]] = []
    run_strn_t10_arm = args.through_strn_t10_arm or args.through_pelvis_arm12
    if run_strn_t10_arm and lsho_idx >= 0 and rsho_idx >= 0 and len(clav_rbak_assignments) >= 2:
        priority_pt_set = head_pt_set | {pi for pi, _ in c7_shoulder_assignments} | {pi for pi, _ in clav_rbak_assignments}
        strn_t10_arm_assignments = assign_strn_t10_arm4_after_clav_rbak(
            points_d_body[best_f],
            d_back,
            d_right,
            priority_pt_set,
            lsho_idx,
            rsho_idx,
            template_body if template_body else template,
        )

    pelvis_arm12_assignments: list[tuple[int, str]] = []
    if args.through_pelvis_arm12 and lsho_idx >= 0 and rsho_idx >= 0 and len(strn_t10_arm_assignments) >= 6:
        priority_pt_set = (
            head_pt_set | {pi for pi, _ in c7_shoulder_assignments}
            | {pi for pi, _ in clav_rbak_assignments} | {pi for pi, _ in strn_t10_arm_assignments}
        )
        pelvis_arm12_assignments = assign_pelvis_arm12_after_strn_t10_arm(
            points_d_body[best_f],
            d_back,
            d_right,
            priority_pt_set,
            lsho_idx,
            rsho_idx,
            template_body if template_body else template,
        )

    # CLAV/RBAK candidates (for print when not --through-clav)
    head_pt_set = {pi for pi, _ in head_assignments}
    priority_pt_set = head_pt_set | {pi for pi, _ in c7_shoulder_assignments}
    lsho_idx = next((pi for pi, lab in c7_shoulder_assignments if str(lab).strip().upper() == "LSHO"), -1)
    rsho_idx = next((pi for pi, lab in c7_shoulder_assignments if str(lab).strip().upper() == "RSHO"), -1)
    clav_cand, rbak_cand = None, None
    if lsho_idx >= 0 and rsho_idx >= 0:
        clav_cand, rbak_cand = get_clav_rbak_candidate_indices(
            points_d_body[best_f], priority_pt_set, lsho_idx, rsho_idx,
        )

    n_body = points_d_body.shape[1]
    # Use original dynamic trial point numbers for unlabeled (body index -> screened dynamic index)
    labels = [
        str(int(body_to_original[i]) + UNLABELED_NUMERIC_LABEL_BASE) for i in range(n_body)
    ]
    for pi, lab in head_assignments + c7_shoulder_assignments + clav_rbak_assignments + strn_t10_arm_assignments + pelvis_arm12_assignments:
        labels[pi] = lab

    # Append obstacle columns so viewer can show them
    points_out = np.concatenate([points_d_body, points_obstacle], axis=1)
    labels_out = labels + obs_labels

    try:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    export_csv(args.output, points_out, labels_out, rate=rate, first_frame=first_frame)
    bestframe_path = Path(args.output).with_suffix(Path(args.output).suffix + ".bestframe")
    try:
        bestframe_path.write_text(str(best_f + 1))
    except Exception:
        pass
    if args.through_pelvis_arm12:
        print(f"Exported pre-leg/foot12 state (through pelvis/arm12): {args.output}")
    elif args.through_strn_t10_arm:
        print(f"Exported pre-pelvis/arm12 state (through STRN/T10/arm4): {args.output}")
    elif args.through_clav:
        print(f"Exported pre-STRN state (through CLAV/RBAK): {args.output}")
    else:
        print(f"Exported pre-CLAV state: {args.output}")
    print(f"  Best frame: {best_f} (0-based), {best_f + 1} (1-based)")
    if not run_clav and clav_cand is not None and rbak_cand is not None:
        print(f"  CLAV candidate: point index {clav_cand}  (highest Z in shoulder Y band)")
        print(f"  RBAK candidate: point index {rbak_cand}  (next highest Z in shoulder Y band)")
    print(f"  Labeled: {[lab for _, lab in head_assignments + c7_shoulder_assignments + clav_rbak_assignments + strn_t10_arm_assignments + pelvis_arm12_assignments]}")
    print(f"  Unlabeled points: numeric labels = loaded C3D column index (0-based, before screening).")

    if args.viewer:
        from marker_label.qc_viewer import run_viewer
        run_viewer(args.output)


if __name__ == "__main__":
    main()
    sys.exit(0)
