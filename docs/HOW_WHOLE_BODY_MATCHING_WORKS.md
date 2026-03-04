# How whole-body matching works (template from static → unlabeled dynamic C3D)

This document explains how the pipeline assigns **body labels** to **unlabeled** 3D points in a dynamic C3D using a **template** built from the labeled static trial.

---

## 1. Inputs

**Template (from static trial)**

- Built by `build_template_from_static(static_points, static_labels)`.
- One 3D position per **label**: `template[label] = (x, y, z)` (mean over static frames).
- Example: `template["LASI"] = [1200, 500, 900]`, `template["LFHD"] = [1100, 400, 1600]`, … (in mm, lab frame or pelvis frame).
- So we have **N_labels** positions, each with a **name** (LASI, RASI, C7, …).

**Unlabeled dynamic (one frame)**

- One frame from the dynamic C3D **after** obstacle removal: `points_frame` shape `(n_points, 3)`.
- Each row is one 3D point; **no labels** — only indices 0, 1, 2, …, n_points−1.
- Same coordinate system as the template (lab frame; if facing was used, dynamic points are already rotated to align with static).
- Typically `n_points` ≈ number of body markers (e.g. 41), and `N_labels` from static is similar (e.g. 41 body labels).

---

## 2. Cost matrix (one frame)

We define a **cost** between every **dynamic point** and every **template label**:

- **Cost(pi, label)** = Euclidean distance between:
  - position of dynamic point **pi**: `points_frame[pi, :]`
  - position of template for **label**: `template[label]`

So we get a matrix of size **(n_points × N_labels)**. Invalid entries (NaN in point or template) are set to **infinity** so they are never chosen.

Example (conceptually):

- Dynamic point 0 might be at (1205, 502, 898); template LASI at (1200, 500, 900) → cost ≈ 8 mm.
- Dynamic point 0 to template LFHD might be 700 mm.
- So point 0 is “closer” to LASI than to LFHD.

---

## 3. Assignment (who gets which label)

We need to assign **each** dynamic point to **at most one** template label, and **each** template label to **at most one** dynamic point, so that the **total cost** (sum of distances of assigned pairs) is as small as possible.

**Method 1: Hungarian algorithm (default)**

- `scipy.optimize.linear_sum_assignment(cost)` finds the assignment that **minimizes total cost**.
- Result: for each row (dynamic point) we get a column (template label), i.e. a list of pairs (point_idx, label).
- Any pair with cost **greater than max_match_distance** (if set) is **rejected**: that point keeps no label at this stage.

**Method 2: Greedy (fallback)**

- For each dynamic point (in order), assign it to the **closest** template label that is not yet assigned.
- Also reject if distance > max_match_distance.

So after this step we have a list of **(point_idx, label)** pairs, e.g. `[(0, "LASI"), (3, "RASI"), (7, "C7"), …]`. Some points may have no label (rejected or not enough labels).

---

## 4. Where this is done: “best frame” only

The pipeline does **not** match every frame independently. It:

1. Chooses **one** frame in the middle of the trial (“best frame”) — e.g. by lowest mean residual or most valid points.
2. Runs the **whole-body matching** (cost matrix + Hungarian/greedy) **only at that frame**.
3. Uses the result as **initial_assignments** for that frame.

So “match whole body in unlabeled C3D” is, in practice: **match whole body at the best frame** using the template from the static trial.

---

## 5. Filling the rest of the trial: temporal propagation

For **all other frames**, we do **not** match again to the template. We **propagate** the labels from the best frame:

- **Backward** (frame best−1, best−2, …, 0): For each point that has a label at frame f+1, find the **nearest** point at frame f (in 3D) and give it the **same** label.
- **Forward** (frame best+1, …, end): Same idea using the **previous** frame.

So the **identity** of each marker over time is “same physical marker = same point index,” and the label is copied along that trajectory. This assumes point indices do not reorder between frames.

---

## 6. Summary flow

| Step | What happens |
|------|----------------|
| 1. Template | Static (labeled) → one 3D position per label. |
| 2. Best frame | Pick one frame in the middle of the dynamic trial (residual or validity). |
| 3. Cost matrix | At that frame: cost(pi, label) = distance(dynamic point pi, template[label]). |
| 4. Assignment | Hungarian (or greedy): assign points to labels so total cost is minimized; reject pairs with distance > max_match_distance. |
| 5. Result at best frame | List of (point_idx, label) = “whole-body match” at that frame. |
| 6. Propagation | Copy labels to all other frames by “nearest point in previous/next frame.” |

So **matching the whole body in the unlabeled C3D** = **steps 3–5** at the best frame, using the template from the static trial; then propagation fills the rest.
