# Marker labeling logic

This document explains how the pipeline assigns labels to unlabeled dynamic markers using a labeled static trial. If labeling failed in your data, the sections below help narrow down where it went wrong.

---

## Overview

**Inputs**

- **Static C3D**: One frame (or multi-frame) trial with **already labeled** markers (e.g. LFHD, RASI, …). Defines the subject-specific body template.
- **Dynamic C3D**: Same subject, **unlabeled** points (Point_1, Point_2, … or generic names). Same number of points per frame; order is arbitrary and can change if the system renumbers points.

**Output**

- Labeled trajectories: each physical marker gets a consistent label (e.g. LASI, OBSTACLE_L) across all frames, exported as C3D and CSV.

**High-level steps**

1. **Obstacle detection** (on dynamic): find the 2 most “stationary” markers with high visibility → label them OBSTACLE_L / OBSTACLE_R and remove them from the body set.
2. **Template** (from static): build one 3D position per body label (mean over static frames).
3. **Body labeling** (on remaining dynamic points): pick one “best” frame in the middle, match points to template by nearest neighbor, then propagate labels forward/backward in time.
4. **Assemble**: body markers in static order + obstacle markers → full labeled matrix and export.

---

## Step 1: Obstacle detection (`obstacle.py`)

**Goal:** Find the two markers that are on the obstacle (bar) so we can label them OBSTACLE_L / OBSTACLE_R and exclude them from body matching.

**Logic**

- **Motion score** per point: mean velocity magnitude over time (frame-to-frame distance). Lower = more stationary.
- **Visibility** per point: fraction of frames where the point is valid (non-NaN).
- **Candidates:** only points with `visibility >= obstacle_visibility_min` (default **0.80**).
- **Selection:** among candidates, take the **2 with lowest motion**.
- **L/R assignment:** by **x** position (smaller x → OBSTACLE_L, larger x → OBSTACLE_R) at first valid frame (or mean over valid frames).

**Where it can fail**

- **Wrong points chosen as obstacle:** If two body markers are very still (e.g. pelvis) and more visible than the real obstacle, they can be chosen instead. → Try raising `--obstacle-visibility` (e.g. 0.9) so only very visible markers are candidates, or check that the real obstacle has high visibility.
- **L/R swapped:** L/R is purely by x. If your lab’s “left” is the larger-x side, the assignment will be reversed. → Swap is fixable in post or by changing the convention in `obstacle.py`.
- **Only one or zero obstacles found:** If fewer than 2 candidates meet the visibility threshold, no obstacles are returned; those points stay in the body set and get body labels (often wrong). → Lower `--obstacle-visibility` slightly or check static/dynamic for dropouts.

---

## Step 2: Template from static (`body_labeling.py` + `pelvis.py`)

**Goal:** One 3D position per body label from the **static** trial.

**Logic**

- Static C3D has **labels** (e.g. LFHD, RASI, …). For each label, take the **mean** of its positions over all static frames (ignoring NaN).
- Optionally transform into **pelvis frame** (LASI, RASI, LPSI, RPSI). Currently the pipeline uses **lab frame** by default (`use_pelvis_frame=False` in `pipeline.py`), so template and matching are in the same lab coordinates.
- Result: `template[label] = (3,) position`.

**Where it can fail**

- **Wrong or missing labels in static:** If the static C3D has wrong labels (e.g. L/R swapped, or Point_1, Point_2), the template is wrong and all downstream body labeling is wrong. → Inspect static in the 3D viewer; fix labels in the static file.
- **NaN everywhere for a label:** That label gets NaN in the template and is effectively skipped in matching.
- **Pelvis frame:** If you later enable pelvis frame, missing pelvis markers (LASI, RASI, LPSI, RPSI) can disable the transform and fall back to lab frame.

---

## Step 3: Body labeling (`body_labeling.py`)

**Goal:** Assign each remaining dynamic point (after removing obstacles) to a body label at every frame.

### 3a) Best frame choice

**Logic**

- Restrict to the **middle** of the trial: `middle_start` to `middle_end` (default **0.2–0.8** of total frames). Avoids start/end where pose or visibility may be bad.
- If **residual** is available (from C3D): choose the frame in that range with **lowest mean residual** (best quality).
- Else: choose the frame with **most valid (non-NaN) points**.

**Where it can fail**

- **Bad “best” frame:** If the chosen frame has a lot of occlusions, big movement, or bad residuals, the initial matching can be wrong and propagation will spread that error. → Try different `middle_start` / `middle_end` (e.g. 0.3–0.7) or add logging to see which frame was chosen and inspect it in the viewer.

### 3b) Match to template (at best frame only)

**Logic**

- At the **best frame**, dynamic points are **unlabeled** (just indices 0, 1, 2, …). Template has one position per label.
- **Greedy nearest-neighbor:** For each dynamic point, find the **closest** template position (Euclidean distance) among labels not yet assigned; assign that label to that point. Repeat until no more assignments (or no more valid template positions).
- Done in **lab frame** (same as template).

**Where it can fail**

- **Wrong initial assignment:** If two markers are close in space (e.g. LWRB and LWRA), the greedy assignment can swap them. If the best frame has unusual pose or occlusion, nearest neighbor can attach the wrong label. → Single wrong assignment at best frame is then propagated to all frames (see below).
- **Too many NaN in template or at best frame:** Fewer assignments; some body markers never get a label and stay empty in the output.

### 3c) Temporal propagation (forward and backward)

**Logic**

- **Backward** from best frame (best_f-1, best_f-2, … 0): For each point that has a label at frame f+1, find the **nearest** point at frame f (in 3D) and give it the same label.
- **Forward** from best frame (best_f+1, … end): Same idea using the previous frame.
- So the **label follows the nearest point** in the next/previous frame. This assumes point indices are **not** reordered between frames (same physical marker stays at same index).

**Where it can fail**

- **Identity swap:** If two markers cross or get very close, “nearest” can jump to the other marker; the two labels then swap and stay swapped. → Typical cause of “markers swapping” in the middle of the trial.
- **Dropout then wrong re-acquisition:** When a marker is NaN for several frames, propagation skips it. When it reappears, the “nearest” point might be a different marker, so the label can attach to the wrong trajectory. → Results in wrong assignment after long gaps.
- **Index reordering:** If the acquisition system sometimes reorders point indices (e.g. by x or by detection order), the same index may not be the same physical marker every frame. Propagation assumes one index = one physical marker; if that’s violated, labels will be inconsistent.

---

## Step 4: Assemble and export (`export.py`)

**Logic**

- **Body:** Output has one column per **static body label** (fixed order). For each frame, `label_per_frame_body` says which dynamic point index has that label; that point’s 3D position is written. If no point has that label at that frame → NaN.
- **Obstacle:** The two obstacle trajectories are appended (OBSTACLE_L, OBSTACLE_R).
- **Labels list** = body labels (static order) + obstacle labels.

**Where it can fail**

- **Wrong order or missing labels:** If static had a different set of labels, “body_labels_static” may not match what you expect (e.g. extra or missing markers). Obstacle labels come from step 1 (OBSTACLE_L, OBSTACLE_R unless detection failed).

---

## Summary: what to check when labeling failed

| Symptom | Likely step | What to check |
|--------|-------------|----------------|
| Obstacle bar is two body markers | 1. Obstacle detection | Visibility/motion: real obstacle should be very stationary and visible. Try `--obstacle-visibility`; inspect with `marker-label-inspect`. |
| OBSTACLE_L/R swapped | 1. Obstacle detection | L/R by x only; adjust convention in code or swap in post. |
| Body markers wrong from first frame | 2–3. Template + best-frame match | Static labels correct? Best frame in a “normal” pose? Try different middle range. |
| Body markers swap in the middle of trial | 3c. Propagation | Cross or near-cross of two markers; consider gap-fill or different propagation (e.g. smooth trajectory). |
| Some markers always NaN | 3. Matching or propagation | Template NaN for that label? Or never assigned at best frame? Or lost after long dropout? |
| Wrong number of markers / wrong order | 4. Assemble | Static labels and obstacle detection output; compare to expected marker list. |

---

## Code references

- **Pipeline entry:** `pipeline.run_pipeline()` in `pipeline.py`
- **Obstacle:** `obstacle.detect_obstacle_markers()` in `obstacle.py`
- **Template:** `body_labeling.build_template_from_static()` in `body_labeling.py`
- **Best frame:** `body_labeling.best_frame_for_matching()` in `body_labeling.py`
- **Match + propagate:** `body_labeling.label_body_markers()` → `match_markers_to_template()`, `propagate_labels_temporal()` in `body_labeling.py`
- **Assemble:** `export.build_full_trajectory_matrix()` in `export.py`
