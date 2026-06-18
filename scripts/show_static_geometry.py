#!/usr/bin/env python3
"""Print Z-band by rank and body segment geometry built from a static trial (39 markers)."""

import sys
import numpy as np
from marker_label.io import load_c3d
from marker_label.body_labeling import build_template_from_static
from marker_label.static_geometry import (
    build_z_band_by_rank_from_static,
    build_z_band_no_arm_hand_from_static,
    build_segment_geometry_from_static,
    save_z_band_to_json,
)
from marker_label.constants import (
    WHOLE_BODY_39_SET,
    Z_BAND_SIZES,
    Z_BAND_SIZES_NO_ARM_HAND,
    N_Z_BANDS,
)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    save_path = None
    if "--save" in sys.argv:
        i = sys.argv.index("--save")
        if i + 1 < len(sys.argv):
            save_path = sys.argv[i + 1]

    path = args[0] if args else "data/SUBJ01 Cal 01.c3d"
    data = load_c3d(path)
    points_s = data["points"]
    labels_s = data["labels"]
    if points_s.ndim == 3:
        points_s = points_s[0:1]
    else:
        points_s = points_s[np.newaxis, ...]
    template, _, _ = build_template_from_static(
        points_s,
        labels_s,
        use_pelvis_frame=False,
    )
    template_39 = {
        k: v for k, v in template.items()
        if str(k).strip().upper() in WHOLE_BODY_39_SET
    }

    print("=" * 60)
    print("Z-BAND BY RANK (from static, 39 markers)")
    print("=" * 60)
    label_to_band = build_z_band_by_rank_from_static(template_39)
    by_band: dict[int, list[str]] = {b: [] for b in range(N_Z_BANDS)}
    for lab, b in label_to_band.items():
        by_band[b].append(lab)
    for b in range(N_Z_BANDS - 1, -1, -1):
        labels_b = sorted(by_band[b])
        z_vals = [float(template_39[lab][2]) for lab in labels_b]
        z_str = f"Z={z_vals[0]:.0f}..{z_vals[-1]:.0f}" if z_vals else ""
        print(f"  Band {b:2d} (size {Z_BAND_SIZES[b]}): {', '.join(labels_b)}  {z_str}")
    print()

    print("=" * 60)
    print("Z-BAND BY RANK WITHOUT ARM/HAND (27 markers)")
    print("=" * 60)
    label_to_band_no_ah = build_z_band_no_arm_hand_from_static(template_39)
    by_band_ah: dict[int, list[str]] = {b: [] for b in range(N_Z_BANDS)}
    for lab, b in label_to_band_no_ah.items():
        by_band_ah[b].append(lab)
    for b in range(N_Z_BANDS - 1, -1, -1):
        labels_b = sorted(by_band_ah[b])
        if not labels_b:
            print(f"  Band {b:2d} (size {Z_BAND_SIZES_NO_ARM_HAND[b]}): (empty)")
            continue
        z_vals = [float(template_39[lab][2]) for lab in labels_b]
        z_str = f"Z={z_vals[0]:.0f}..{z_vals[-1]:.0f}"
        print(f"  Band {b:2d} (size {Z_BAND_SIZES_NO_ARM_HAND[b]}): {', '.join(labels_b)}  {z_str}")
    if save_path:
        save_z_band_to_json(label_to_band_no_ah, save_path)
        print(f"\n  Saved to: {save_path}")
    print()

    print("=" * 60)
    print("BODY SEGMENT GEOMETRY (from static, 39 markers)")
    print("=" * 60)
    segs = build_segment_geometry_from_static(template_39)
    for s in segs:
        axis = s["axis"]
        axis_str = f"[{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]" if axis is not None else "None"
        print(f"  {s['name']}:")
        print(f"    markers: {' -> '.join(s['markers'])}")
        print(f"    length_mm: {s['length_mm']:.1f}")
        print(f"    axis: {axis_str}")
        print(f"    edges: {s['edges']}")
    print(f"  Total segments: {len(segs)}")


if __name__ == "__main__":
    main()
