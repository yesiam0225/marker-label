#!/usr/bin/env python3
"""Print Z-band boundaries and label-to-band for a static C3D template (by-gap only, 39 whole-body markers)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marker_label.io import load_c3d
from marker_label.body_labeling import build_template_from_static, _z_band_boundaries_from_template
from marker_label.constants import N_Z_BANDS, WHOLE_BODY_39


def main() -> None:
    static_path = Path(__file__).resolve().parents[1] / "data" / "SUBJ01 Cal 01.c3d"
    if not static_path.exists():
        print(f"Static file not found: {static_path}")
        sys.exit(1)
    data = load_c3d(str(static_path))
    points_s = data["points"]
    labels_s = data["labels"]
    # Restrict to 39 whole-body markers only (no *39, *40, etc.)
    whole_set = {m.upper() for m in WHOLE_BODY_39}
    keep = [i for i in range(len(labels_s)) if labels_s[i].strip().upper() in whole_set]
    if not keep:
        print("No whole-body 39 markers found in static file.")
        sys.exit(1)
    points_s = points_s[:, keep, :]
    labels_s = [labels_s[i] for i in keep]
    template, _, _ = build_template_from_static(points_s, labels_s, use_pelvis_frame=False)

    n_bands = N_Z_BANDS
    print("=" * 60)
    print("Z-BAND BY GAP (39 whole-body markers only)")
    print("=" * 60)
    boundaries_gap, label_to_band_gap = _z_band_boundaries_from_template(
        template, n_bands=n_bands, z_band_by_value=False, z_band_by_gap=True
    )
    print(f"\nBoundaries (Z): {boundaries_gap}")
    band_to_labels_gap: dict[int, list[str]] = {b: [] for b in range(n_bands)}
    for lab, b in label_to_band_gap.items():
        band_to_labels_gap[b].append(lab)
    for b in range(n_bands):
        labs = band_to_labels_gap[b]
        if not labs:
            print(f"  Band {b}: (empty)")
            continue
        z_vals = [(template[lab][2], lab) for lab in labs]
        z_vals.sort(key=lambda x: x[0])
        labels_ordered = [lab for _, lab in z_vals]
        z_min = min(template[lab][2] for lab in labs)
        z_max = max(template[lab][2] for lab in labs)
        print(f"  Band {b} (Z ~ {z_min:.0f}–{z_max:.0f} mm): {labels_ordered}")


if __name__ == "__main__":
    main()
