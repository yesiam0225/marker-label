# Head labeling logic (rule-based)

How LFHD, RFHD, LBHD, RBHD are assigned from the top-4-Z head candidates. No per-trial flags; same rules for all trials.

---

## 1. Pipeline: L/R and A/P axes (same frame as points)

**File:** `src/marker_label/pipeline.py` (around 199–225)

1. **Facing rotation (optional)**  
   If `static_facing_axis` and `dynamic_facing_axis` are set, dynamic body points are rotated so orientations align:
   - `points_d_body_for_matching = points_d_body @ R.T`
   - `R_facing = R` is stored for the next step.

2. **Walking direction → d_back, d_right**  
   - `wdx = compute_walking_direction_x(points_d)` (+1: X increases, -1: X decreases, 0: unclear).
   - `d_back, d_right = get_lr_ap_axes_from_walking(wdx, left_side_positive_lr=...)`  
     Lab-frame unit vectors: posterior direction and right-side direction.

3. **Centroid (in matching frame)**  
   - `centroid_xy` = mean of `points_d_body_for_matching[:, :, :2]` over frames and points.

4. **Rule: same frame as points**  
   If facing rotation was applied, **rotate d_back and d_right** into the matching frame so head (and rest of labeling) use the same coordinate system as the points:
   ```python
   if R_facing is not None:
       R_2d = R_facing[:2, :2]
       d_back = R_2d @ d_back
       d_right = R_2d @ d_right
   lr_ap_from_walking = (d_back, d_right, centroid_xy)
   ```

---

## 2. Axes from walking direction

**File:** `src/marker_label/body_labeling.py` — `get_lr_ap_axes_from_walking` (81–104)

- **wdx > 0** (subject walks toward +X):  
  `d_back = [-1, 0]` (posterior = -X), `d_right = [0, -1]` (right = smaller Y).
- **wdx < 0**:  
  `d_back = [1, 0]`, `d_right = [0, 1]`.
- **wdx == 0**:  
  Default `d_back = [-1, 0]`, `d_right = [0, -1]`.
- **left_side_positive_lr True**:  
  `d_right = -d_right` (positive L/R axis = left).

---

## 3. Head candidate points (top 4 by Z)

**File:** `src/marker_label/body_labeling.py` — `_match_head_markers_first` (797–818)

- **use_top4_z True (default):**  
  Among all points with finite Z, take the **4 with largest Z** → `point_indices` (length 4).
- **use_top4_z False:**  
  Restrict to points in head band(s), then take top 4 by Z within that set.

`pts_xy` = 2D coordinates of these 4 points (columns X, Y), shape (4, 2).

---

## 4. Fore/Back (A/P) — rule-based split

**File:** `src/marker_label/body_labeling.py` (828–854)

**Rule 1 (preferred):** When **d_back** and **centroid_xy** are both available:

- Project each of the 4 points onto the posterior direction (relative to centroid):
  ```text
  dot_back[i] = (pts_xy[i] - centroid_xy) · d_back
  ```
- **Smaller dot_back** → more anterior; **larger dot_back** → more posterior.
- Sort by `dot_back` ascending:
  - **anterior 2:** indices with the **two smallest** `dot_back` → `ant_idx`
  - **posterior 2:** indices with the **two largest** `dot_back` → `post_idx`

This uses the same frame as the points (and, if applicable, the rotated d_back from the pipeline), so it stays correct with or without facing rotation.

**Rule 2 (fallback):** If d_back is set but centroid is not:

- Use X only: `order_by_x = argsort(pts_xy[:, 0])` descending.
- If `d_back[0] > 0`: smaller X = anterior → `ant_idx = order_by_x[2:]`, `post_idx = order_by_x[:2]`.
- Else: larger X = anterior → `ant_idx = order_by_x[:2]`, `post_idx = order_by_x[2:]`.

**Rule 3:** If d_back is not set:

- Use `head_anterior_smaller_x` / `head_anterior_larger_x` or default (larger X = anterior) with the same `order_by_x`.

---

## 5. Left/Right within anterior and posterior

**File:** `src/marker_label/body_labeling.py` (856–869)

For each pair (anterior 2, posterior 2), L/R is decided by **Y** only:

- **assign_lr(two_indices):**
  - Compare `pts_xy[i0, 1]` and `pts_xy[i1, 1]`.
  - **left_side_positive_lr True:** larger Y = left.
  - **left_side_positive_lr False:** smaller Y = left.
  - Returns `(left_i, right_i)`.

Then:

- `lfhd_i, rfhd_i = assign_lr(ant_idx)`  (anterior left → LFHD, anterior right → RFHD)
- `lbhd_i, rbhd_i = assign_lr(post_idx)`  (posterior left → LBHD, posterior right → RBHD)

---

## 6. Assign labels to point indices

**File:** `src/marker_label/body_labeling.py` (871–879)

- `(point_indices[lfhd_i], LFHD)`, `(point_indices[rfhd_i], RFHD)`,
- `(point_indices[lbhd_i], LBHD)`, `(point_indices[rbhd_i], RBHD)`  
  are appended to the assignments (only for labels present in the template).

Optional **head_swap_lfhd_rbhd**: if True, swap the labels LFHD and RBHD only after the above (not used in the default rule-based pipeline).

---

## Summary flow

```text
Pipeline:
  points for matching (rotated if facing set)
  → wdx → d_back, d_right (lab)
  → if rotation: d_back, d_right rotated by R[:2,:2]
  → centroid_xy from points_for_matching
  → lr_ap_from_walking = (d_back, d_right, centroid_xy)

Head:
  top 4 by Z → pts_xy (4×2)
  → dot_back = (pts_xy - centroid_xy) @ d_back
  → ant_idx = 2 smallest dot_back, post_idx = 2 largest dot_back
  → assign_lr(ant_idx) → LFHD, RFHD
  → assign_lr(post_idx) → LBHD, RBHD
  → (point_index, label) for each
```

All of this is rule-based; no trial-specific flags are required for correct frame alignment or A/P split.
