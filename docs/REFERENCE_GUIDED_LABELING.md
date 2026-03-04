# Using Manually Labeled Trials to Guide Labeling

This document describes how to use **manually labeled** static and dynamic trials from a **reference subject** to determine the static–dynamic relationship, and how to **apply** that relationship when labeling **unlabeled** dynamic trials for other subjects using their **labeled static** trials.

---

## Overview

1. **Reference subject** (e.g. BBpilot01): You have both static and dynamic **manually labeled** with the same marker names.
2. **Target subject** (e.g. BBA01): You have a **labeled static** trial and an **unlabeled dynamic** trial that you want to label.

The idea: learn the static–dynamic relationship from the reference, then use that knowledge to improve the pipeline when labeling the target subject.

---

## Step 1: Determine the relationship (reference subject)

Use `marker-label-analyze` on the reference subject’s manually labeled static and dynamic C3D files:

```bash
marker-label-analyze "path/to/BBpilot01 Cal 01.c3d" "path/to/BBpilot01 Trial 10.c3d" -o reference_report.json
```

This produces:

- **Per-frame rigid transform**: For each (sampled) dynamic frame, the rotation (deg), translation (mm), and RMS (mm) from the static template to that frame.
- **Best frame (closest to template)**: The frame index where RMS is minimum — the “true” best frame for matching when labels are known.
- **Pipeline’s best frame**: The frame the current pipeline would choose (residual-based in the middle portion).
- **Summary**: Mean/max rotation, mean/max translation, mean/min RMS over the trial.

Interpretation:

- If **“Frame closest to template”** and **“Pipeline choice”** differ, the pipeline may be matching at a suboptimal frame on the target subject.
- **Mean/min RMS** and **rotation/translation** tell you typical pose difference between static and dynamic; use them to set distance thresholds and facing (see below).

---

## Step 2: Apply the relationship to label unlabeled dynamic trials

Use the reference report to inform how you run `marker-label` on the target subject.

### 2a. Best-frame position (recommended)

The pipeline picks a “best frame” in the **middle** of the trial (default 20–80% of frames). The reference tells you where the **true** best frame was (e.g. frame 170 of 463 ≈ 37% through the trial).

- If the reference’s best frame is near the **start** or **end** of the trial, the default middle window may miss it.
- Use **`--reference-report reference_report.json`** so the pipeline searches for the best frame in a **narrow window** around the same **relative position** as in the reference (e.g. 22–52% if reference best was at 37%). No need to set `--middle-start` / `--middle-end` by hand.

Example:

```bash
marker-label "path/to/BBA01 Cal 01.c3d" "path/to/BBA01 Trial 05.c3d" -o out/BBA01_trial05 \
  --static-facing y --dynamic-facing x \
  --reference-report reference_report.json
```

### 2b. Distance thresholds

From the reference report:

- **min_rms_mm**: RMS at the frame closest to the template. A reasonable **max_match_distance** is a few times this (e.g. 2–3×) so the initial assignment rejects only very bad matches.
- **mean_rms_mm** or propagation distance over time can inform **max_propagation_distance** so you don’t propagate across large jumps (e.g. after dropout or marker crossing).

If you don’t set these, **`--reference-report`** can suggest defaults from the reference summary (e.g. `max_match_distance = 3 * min_rms_mm`, `max_propagation_distance` from typical frame-to-frame motion).

### 2c. Facing (static vs dynamic)

If in the reference the **mean rotation** from static to dynamic is large (e.g. ~90°), the subject likely turned between static and dynamic. Use **`--static-facing`** and **`--dynamic-facing`** on the target subject the same way (e.g. `--static-facing y --dynamic-facing x`). The reference analysis does not set these automatically; you choose them from the protocol or from the reference’s rotation magnitude/axis.

---

## Workflow summary

| Step | Tool | Input | Output |
|------|------|--------|--------|
| 1 | `marker-label-analyze` | Reference: labeled static + labeled dynamic | `reference_report.json` (best frame index, RMS, R/t summary) |
| 2 | `marker-label` | Target: labeled static + unlabeled dynamic; optional `--reference-report` | Labeled C3D/CSV |

Optional flags when running `marker-label`:

- **`--reference-report reference_report.json`** — Use reference to set best-frame window (and optionally default distance thresholds).
- **`--static-facing` / `--dynamic-facing`** — Set from protocol or reference rotation.
- **`--max-match-distance` / `--max-propagation-distance`** — Set from reference RMS if desired, or let `--reference-report` suggest them.

---

## What the pipeline does with the reference report

When you pass **`--reference-report`**:

1. **Best-frame window**: Compute the reference’s “frame closest to template” as a **fraction** of the reference trial length (e.g. 170/462 ≈ 0.37). For the target trial, search for the best frame only in a band around that fraction (e.g. 0.22–0.52), instead of the default 0.20–0.80. So the **relationship** “best frame is near 37% of the way through” is applied to the new trial.
2. **Optional default thresholds**: If you did not set `--max-match-distance` or `--max-propagation-distance`, the pipeline can suggest values from the reference’s `min_rms_mm` and summary stats so that labeling is consistent with the reference’s geometry.

This way, the **manually labeled** reference pair is used only to **determine** the relationship (where to match, how strict to be); the **labeling** of the target subject still uses only that subject’s **labeled static** and **unlabeled dynamic** trials.
