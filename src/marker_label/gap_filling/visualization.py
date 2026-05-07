"""Summary PNG for gap-fill run."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def plot_gap_fill_summary(metrics: Mapping, save_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Gap fill summary")

    vb = metrics.get("visibility_before", {})
    va = metrics.get("visibility_after", {})
    markers = sorted(set(vb) | set(va))
    x = np.arange(len(markers))
    before = [vb.get(m, 0) for m in markers]
    after = [va.get(m, 0) for m in markers]
    axes[0].bar(x - 0.2, before, 0.4, label="before")
    axes[0].bar(x + 0.2, after, 0.4, label="after")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(markers, rotation=90, fontsize=6)
    axes[0].set_title("Visibility (finite frames)")
    axes[0].legend()

    fbm = metrics.get("fills_by_method", {})
    methods = ["rigid_body", "asis_only", "spline", "unfillable"]
    axes[1].bar(range(len(methods)), [fbm.get(m, 0) for m in methods])
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=30, ha="right")
    axes[1].set_title("Fills by method (log rows)")

    # Stacked confidence per method from fills_log not in metrics — approximate from global counts
    # Build from optional detailed dict if present
    stacked = metrics.get("fills_by_method_and_confidence") or {}
    meths2 = [m for m in ("rigid_body", "asis_only", "spline") if m in stacked]
    if not meths2:
        axes[2].text(0.5, 0.5, "No per-method confidence breakdown", ha="center", va="center")
        axes[2].set_axis_off()
    else:
        confs = ["HIGH", "MEDIUM", "LOW"]
        bottom = np.zeros(len(meths2))
        x2 = np.arange(len(meths2))
        for c in confs:
            vals = [stacked[m].get(c, 0) for m in meths2]
            axes[2].bar(x2, vals, bottom=bottom, label=c)
            bottom += np.array(vals, dtype=float)
        axes[2].set_xticks(x2)
        axes[2].set_xticklabels(meths2, rotation=30, ha="right")
        axes[2].set_title("Confidence by method (successful fills)")
        axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
