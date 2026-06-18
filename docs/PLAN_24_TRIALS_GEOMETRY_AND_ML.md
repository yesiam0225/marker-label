# Detailed plan: (1) Geometry pipeline tuning and (2) ML from 24 manually labeled trials

You have **24 manually labeled dynamic trials** from **REF01** and **REF02**. Each trial has the same subject’s static (labeled) and dynamic (now manually labeled). This document describes two ways to use these 24 trials: **(1)** improve the geometry-based pipeline, and **(2)** train a machine-learning model and use it for labeling.

---

## Subject location (static vs dynamic) and what RMS reflects

### How (1) handles difference in subject location

- **Translation (position):** In the **labeling pipeline**, matching is done in **lab coordinates**: each dynamic point is compared to each template position by **Euclidean distance**. If the subject simply moved (e.g. walked 500 mm in x), the **whole** point cloud moves together; the **relative** geometry (which point is LASI, which is RASI) is unchanged. So “closest template point” to a given dynamic point is still the same label. The algorithm does **not** explicitly subtract centroids or align position — it doesn’t need to, because nearest-neighbor (and Hungarian) matching is **invariant to a global translation**. So in (1) we do **not** add a separate step to “address” subject location; the existing matching already ignores global position.

- **Orientation (facing):** We **do** correct for the subject facing a different lab axis in static vs dynamic. That is a **rotation** in the horizontal plane (e.g. static facing y, dynamic facing x). The pipeline uses `--static-facing` / `--dynamic-facing` to **rotate** the dynamic points (around lab z) so that orientations align before matching. So (1) uses the 24 trials to set **best-frame window** and **distance thresholds**; it does not add new logic for translation, and it relies on your providing facing axes when they differ.

- **Summary:** Subject **location** (translation) is handled implicitly by distance-based matching. Subject **orientation** (facing) is handled by the existing facing flags. (1) only tunes **where** we match (best frame) and **how strict** we are (max_match_distance, max_propagation_distance).

### What RMS reflects (in marker-label-analyze)

In **marker-label-analyze**, for each dynamic frame we have:

- **Template positions** (from static): one 3D point per label, in lab frame.
- **Dynamic positions** (that frame): same labels, so we know which dynamic point corresponds to which template point.

We fit a **rigid transform** (rotation **R**, translation **t**) so that **template** aligns to **dynamic** as well as possible (Procrustes):

- `pred_i = R @ template_i + t`
- **RMS** = √ mean over points of `(dynamic_i - pred_i)²` (in mm).

So RMS is the **residual after** the best rotation and translation have been applied. It therefore **does not** reflect:

- The **global translation** between static and dynamic (that is absorbed into **t**).
- The **global rotation** between static and dynamic (that is absorbed into **R**).

It **does** reflect:

- **Non-rigid shape change:** Limbs/trunk moved relative to the static pose (e.g. arms swung, leg forward). So “how much the body shape differs from the static template after we’ve aligned it as well as we can with a single rigid transform.”
- **Measurement noise** and small tracking errors.
- **Missing points:** If some points are NaN, the fit uses fewer points; RMS is over the paired points that are valid.

So **low RMS** at a frame means: “at this frame, the subject’s marker cloud is close to a rigidly moved copy of the static template” (similar pose, good data). **High RMS** means: “even after the best rigid alignment, the two clouds differ a lot” (different pose, or noisier data). The **frame with minimum RMS** is the one where the dynamic trial is “closest” to the static template in this sense; that is what we use in (1) to set the best-frame search window.

---

## Part (1): Use 24 trials to improve the geometry pipeline

**Goal:** Use the 24 trials to choose better **reference statistics** and **pipeline parameters** so that when you run `marker-label` on a new subject (e.g. SUBJ01), the geometry-based matching is more reliable.

### 1.1 Inputs

- **Per trial (×24):**
  - One **static** C3D for that subject (REF01 or REF02), already labeled.
  - One **dynamic** C3D for that subject, **manually labeled** (same marker names as static).
- **Units:** If any dynamic is in meters, use `--dynamic-unit m` when loading that file (as you did for REF01 Trial 10).
- **Assumption:** Static and dynamic use the **same label set** (e.g. same 43 or 141 marker names) so that “template from static” and “labels in dynamic” are comparable.

### 1.2 Steps

**Step 1a – Run reference analysis on every trial**

For each of the 24 trials, run:

```bash
marker-label-analyze <static.c3d> <dynamic.c3d> -o report_<trial_id>.json [--dynamic-unit m if needed]
```

So you get 24 JSON reports. Each report contains:

- `frame_closest_to_template`: frame index where RMS (template ↔ dynamic) is minimum.
- `pipeline_best_frame`: frame the current pipeline would pick (residual-based).
- `n_frames_dynamic`: number of frames in that dynamic trial.
- `summary`: e.g. `mean_rotation_deg`, `mean_translation_mm`, `mean_rms_mm`, `min_rms_mm`.

**Step 1b – Aggregate across 24 trials (and a caveat)**

**Caveat (subject- and stride-dependence):** The “best frame” fraction (where in the trial the frame closest to the template falls) can be **subject-specific** and **stride-dependent** (보폭, walking speed, trial type). So an aggregated window from REF01/02 may not transfer well to SUBJ01. Use the aggregated window as a **loose prior** (e.g. search in 0.2–0.8 but optionally bias toward the reference fraction), or prefer a **wider** middle range and rely more on **Procrustes at best frame** or a **per-trial** “closest to template” criterion (see below) when labeling a different subject.

**Aggregate anyway for reference:**

From the 24 reports, compute:

- **Best-frame position (fraction):**  
  For each trial, `best_frac = frame_closest_to_template / (n_frames_dynamic - 1)` (or similar).  
  Then take **mean** and **standard deviation** (or percentiles, e.g. 25th–75th) of these 24 fractions.  
  This tells you “on average, the best frame is around 35% of the way through the trial, with some spread.”

- **RMS and distances:**  
  Mean (and e.g. 5th/95th percentiles) of `min_rms_mm` and `mean_rms_mm` across the 24 trials.  
  These guide reasonable **max_match_distance** and **max_propagation_distance** (e.g. 2–3× `min_rms_mm` for initial match, and a value that allows normal frame-to-frame motion but rejects big jumps).

- **Optional – Pipeline vs “true” best:**  
  For each trial, check whether `pipeline_best_frame` equals `frame_closest_to_template`. If they often differ, the pipeline’s residual-based choice is suboptimal; the aggregated **best_frac** is then especially useful to drive a narrow search window.

**Step 1c – Build a single “reference” report or parameter set**

- **Option A – Synthetic reference report:**  
  Pick one representative trial (e.g. median length, median best_frac) and use its JSON as the `--reference-report` for the pipeline.  
  Or build a **single JSON** that has the **aggregated** values: e.g. set `frame_closest_to_template` to the **mean best frame index** from the 24 trials (rounded), and `n_frames_dynamic` to a typical length (e.g. median), so that `suggested_pipeline_params_from_reference()` produces a **middle_start / middle_end** window centered on the mean best fraction.

- **Option B – Fixed parameters:**  
  Skip the report and set pipeline flags directly from the aggregates, e.g.:
  - `--middle-start` / `--middle-end`: e.g. `mean_best_frac - 0.15` to `mean_best_frac + 0.15`.
  - `--max-match-distance`: e.g. 2.5 × (mean of 24 `min_rms_mm`), capped at 400 mm.
  - `--max-propagation-distance`: e.g. 1.5 × (mean of 24 `min_rms_mm`) or a fixed 120 mm, depending on your typical frame-to-frame movement.

**Step 1d – Use when labeling SUBJ01**

When you run the pipeline for the **new** subject (SUBJ01):

```bash
marker-label "SUBJ01 Cal 01.c3d" "SUBJ01 Trial 05.c3d" -o out/SUBJ01_trial05 \
  --static-facing y --dynamic-facing x \
  --reference-report out/aggregated_reference.json
```

(if you built an aggregated reference JSON), **or** pass the chosen `--middle-start`, `--middle-end`, `--max-match-distance`, `--max-propagation-distance` explicitly.

### 1.3 Outputs

- **24 analysis reports** (e.g. `report_trial01.json`, …).
- **Summary statistics:** mean/std or percentiles of best-frame fraction, min/mean RMS, rotation, translation.
- **One aggregated reference report** (optional) or a **fixed parameter set** for the pipeline.
- **Improved geometry-based labels** on SUBJ01 (or any new subject) when you run `marker-label` with these parameters.

### 1.4 What this does and does not do

- **Does:** Uses the 24 trials only to **tune the existing geometry pipeline** (best-frame window and distance thresholds). No ML; same algorithm, better parameters.
- **Does not:** Learn a new mapping from point cloud to labels; it only refines **where** and **how strictly** the current matching runs.

---

## Part (2): Train ML on the 24 trials and use it for labeling

**Goal:** Train a model that takes a **single frame’s 3D point cloud** (and optionally context) and predicts **which label each point has**. Use the 24 trials as training (and validation) data, then apply the model to **new trials** (e.g. SUBJ01) or to held-out trials for evaluation.

### 2.1 Inputs

- **Same 24 trials:** For each trial, you need:
  - **Dynamic C3D (or exported CSV)** with 3D positions per frame and **per-point labels** (from manual labeling).
- **Consistent label set:** All 24 trials (and, for application, SUBJ01) should use the **same marker names** (e.g. same 43 labels). If REF01/02 have 141 markers and SUBJ01 has 43, you must either:
  - Restrict to the **common 43 labels** for training and application, or
  - Define a mapping (e.g. “body part” or “reduced set”) so that the model’s output can be interpreted on SUBJ01.
- **Point order:** In C3D/CSV, point order can differ across trials or frames. The model must treat the set of points as **unordered** (permutation-invariant) so that the same physical configuration always gets the same predicted labels regardless of index order.

### 2.2 Model design (high level)

- **Input:** One frame: a set of 3D points. Represent as:
  - **Matrix:** `(N, 3)` coordinates (N = number of markers in that frame; can vary if you allow missing markers, but usually N is fixed per trial).
  - Optionally: extra features per point (e.g. residual, or velocity from previous frame).
- **Output:** For each of the N points, a **label** (or distribution over labels). So output is either:
  - **N × num_labels** logits/probabilities, or
  - **N** label indices (or one-hot), with a fixed vocabulary of labels (e.g. 43 classes).
- **Permutation invariance:** The same 3D configuration should get the same labeling even if the rows of the point matrix are permuted. Common approaches:
  - **Set encoder:** e.g. PointNet-style: per-point MLP → max (or mean) over points → global feature → per-point head that uses both global and local feature. Or use a small **transformer** over the set of points with position encoding from coordinates.
  - **Pairwise / graph:** Encode pairwise distances (or relative vectors) and use a network that is invariant to permuting the points (e.g. message-passing on a fully connected graph over points).
- **Training:** Supervised. For each frame in the 24 trials, you have (points, labels). Loss: cross-entropy (or similar) between predicted and true labels for each point. You can mask loss on missing (NaN) points if needed.

### 2.3 Data preparation

- **Per trial:** Load dynamic C3D (or CSV) and labels; get `(T, N, 3)` and `(T, N)` label indices (T = frames, N = markers).
- **Split options:**
  - **By trial:** e.g. 18 trials for training, 6 for validation (or 20 train / 4 val). Ensures the model is evaluated on whole trials it has not seen.
  - **Leave-one-subject-out:** Train on all trials from REF01, validate on all from REF02 (or vice versa). Repeat for the other subject. This measures **cross-subject** generalization.
  - **Leave-one-trial-out:** Train on 23 trials, test on 1; rotate so each trial is left out once. Gives an estimate of “same-subject, new trial” performance.
- **Frames:** Use all frames (or subsample, e.g. every 5th frame) to get many training samples. Data augmentation (e.g. random small rotation, translation, or noise) can help if the same subject appears in multiple trials.
- **Output format:** Either (1) one training example per frame: `(N, 3)` → `(N,)` label indices, or (2) batched frames with the same N.

### 2.4 Training and validation

- **Loss:** Cross-entropy over the N points (optionally masked for NaN).
- **Metric:** Accuracy or macro F1 over points (and optionally over frames). You can also report “frame-level accuracy” (fraction of frames where all labels are correct).
- **Validation:** Run on the chosen validation split (by trial or by subject). If you do leave-one-subject-out, you get two numbers: performance when generalizing from REF01→REF02 and from REF02→REF01. That tells you how well the model transfers between these two subjects.

### 2.5 Application to SUBJ01 (or a new trial)

- **At inference:** For each frame of SUBJ01’s unlabeled dynamic trial, you have `(N, 3)` points (N may be 43 or whatever the trial has). Run the model → get predicted label per point. No need for static or template; the model directly maps point cloud → labels.
- **Consistency over time:** The model is frame-independent, so you may get label “jitter” (same physical marker predicted as different labels on adjacent frames). You can:
  - **Temporal smoothing:** e.g. majority vote or HMM over a short window (e.g. 5 frames), or
  - **Propagate from high-confidence frames:** Use the model’s confidence (e.g. softmax max) to pick a few frames where prediction is confident, then propagate those labels in time (similar to the geometry pipeline’s propagation) and fill the rest with the model or a hybrid.

### 2.6 Outputs

- **Trained model** (weights) and training/validation curves.
- **Validation metrics:** e.g. point accuracy and (if applicable) leave-one-subject-out performance.
- **Predicted labels** for SUBJ01 (or any new trial) when you run the model on that trial’s frames; optionally **smoothed or hybrid** with propagation.

### 2.7 What this does and does not do

- **Does:** Learns a **direct mapping** from (point cloud) → (labels) from the 24 trials. Can capture pose-dependent patterns that the geometry pipeline might miss. With 2 subjects and 24 trials, you get a first idea of **cross-subject** generalization.
- **Does not:** Replace the need for consistent labels and units; if SUBJ01 has a different marker set, you still need a common subset or mapping. It also does not use the static trial at inference (unless you add it as an optional input to the model).

---

## Summary table

| Aspect | (1) Geometry pipeline tuning | (2) ML from 24 trials |
|--------|------------------------------|-------------------------|
| **Input** | 24 × (static + manually labeled dynamic) | Same 24 dynamic trials (points + labels per frame) |
| **Main use** | Compute best-frame and RMS stats; set middle_start/end, max_match_distance, max_propagation_distance | Train permutation-invariant (point cloud → labels) model |
| **Output** | Aggregated reference report or fixed CLI parameters; better geometry-based labels on SUBJ01 | Trained model; predicted labels on SUBJ01 or held-out trials |
| **Evaluation** | Compare labeling quality on SUBJ01 (or leave-one-out on 24) when using new params vs old | Validation accuracy; leave-one-subject-out or leave-one-trial-out |
| **When to use** | To improve the existing pipeline without adding ML code | To try a data-driven predictor and optionally combine with (1) (e.g. geometry + ML refinement) |

If you tell me your preferred format (e.g. CSV vs C3D for the 24 trials, and whether you want to implement (1) aggregation in this repo or (2) a minimal ML script elsewhere), I can outline concrete file layouts and code steps next.
