# All Possible Reasons LFHD/RBHD Can Be Switched (Code Analysis)

Analysis of the **current code** only. No changes made. Locations are in `src/marker_label/` unless noted.

---

## BEFORE head labeling

### 1. Walking direction sign (pipeline + body_labeling)

- **Where**: `pipeline.py` line 210: `wdx = compute_walking_direction_x(points_d)`  
  `body_labeling.py` 47–77: `compute_walking_direction_x`; 81–104: `get_lr_ap_axes_from_walking`.
- **Logic**: Centroid X over time → `mean_dx > 5` → +1, `mean_dx < -5` → -1, else 0.  
  +1 → `d_back = [-1,0]` (larger X = anterior).  
  -1 → `d_back = [1,0]` (smaller X = anterior).
- **Effect**: If `wdx` is -1 when the subject actually walks +X, head A/P uses “smaller X = anterior” and LFHD/RBHD are assigned to the wrong points (diagonal swap).
- **Note**: `points_d` is after visibility trim but **before** obstacle removal; if obstacles change centroid trend, `wdx` can still be wrong.

### 2. Frame mismatch: walking axes vs. matching coordinates (pipeline)

- **Where**: `pipeline.py` 199–217.  
  When `static_facing_axis` and `dynamic_facing_axis` are set, `points_d_body_for_matching = points_d_body @ R.T` (rotated).  
  But `wdx = compute_walking_direction_x(points_d)` and `d_back, d_right = get_lr_ap_axes_from_walking(wdx, ...)` use **unrotated** `points_d`.
- **Logic**: `d_back` is in the **original** lab frame. Head matching uses **rotated** `points_frame` (X, Y in rotated frame). A/P is decided by `d_back[0]` and “X” of the head points; that X is rotated, so “larger X / smaller X” in the rotated frame does not correspond to the same anterior/posterior as `d_back` in the original frame.
- **Effect**: With facing-axis rotation, head A/P can be wrong and LFHD/RBHD can be switched even when walking direction is correct in the original frame.

### 3. `points_d` vs. body-only points (pipeline)

- **Where**: `pipeline.py` 210: `wdx = compute_walking_direction_x(points_d)`.  
  `points_d` may still include obstacle columns (before body extraction). After obstacle removal we use `points_d_body` / `points_d_body_for_matching` for labeling.
- **Effect**: If obstacle columns are included in `points_d`, centroid X trend can differ from the body-only trend and change `wdx`, hence `d_back` and head A/P.

### 4. Threshold 5.0 mm/frame (body_labeling)

- **Where**: `body_labeling.py` 73–76: `if mean_dx > 5.0: return 1` etc.
- **Effect**: If `|mean_dx| <= 5`, `wdx = 0` and code uses default `d_back = [-1,0]` (larger X = anterior). So borderline trials can flip between +1 and 0 (or -1 and 0) with small changes in data or threshold, and head A/P can change.

---

## DURING head labeling (_match_head_markers_first)

### 5. A/P from `d_back` only when `d_back` is not None (body_labeling)

- **Where**: `body_labeling.py` 827–846.  
  If `d_back is not None`, head uses only `d_back` for A/P (`d_back[0] > 0` → smaller X = anterior, else larger X = anterior).  
  So any wrong `d_back` (from 1–4 above) directly causes wrong anterior/posterior and can swap LFHD/RBHD.

### 6. `order_by_x` and indexing (body_labeling)

- **Where**: `body_labeling.py` 828–846.  
  `order_by_x = np.argsort(pts_xy[:, 0])[::-1]` (descending X).  
  `ant_idx = list(order_by_x[:2])` or `order_by_x[2:]`, `post_idx` the other two.
- **Possible bug**: `order_by_x[:2]` and `order_by_x[2:]` are correct for “two largest X” and “two smallest X”. No obvious index swap in the current code.

### 7. `assign_lr` and L/R (body_labeling)

- **Where**: `body_labeling.py` 848–862.  
  `assign_lr(two_local_indices)` returns `(left_i, right_i)` by Y and `left_side_positive_lr`.  
  `lfhd_i, rfhd_i = assign_lr(ant_idx)`, `lbhd_i, rbhd_i = assign_lr(post_idx)`.
- **Effect**: If `left_side_positive_lr` is wrong for this trial (e.g. left = smaller Y but flag True), L/R within anterior and within posterior are flipped. That would swap LFHD↔RFHD and LBHD↔RBHD, not only LFHD↔RBHD, unless combined with another effect.

### 8. Assignment order and `head_key` (body_labeling)

- **Where**: `body_labeling.py` 864–866.  
  `assignments.append((point_indices[lfhd_i], head_key["LFHD"]))` etc.  
  `head_key` is built from template labels (`.strip().upper()` → key).
- **Effect**: If the static template uses a different string (e.g. `"LFHD "`) than `body_labels_static` (e.g. `"LFHD"`), export uses `label_per_frame_body[f, pi] == label` (export.py 45); an exact string mismatch would cause missing column, not a swap. So this is more about label identity than LFHD/RBHD swap.

### 9. `head_swap_lfhd_rbhd` (body_labeling)

- **Where**: `body_labeling.py` 872–877.  
  If True, labels LFHD and RBHD are explicitly swapped after assignment.
- **Effect**: Default is False; only relevant if the flag is passed. When used, it directly swaps LFHD/RBHD.

---

## AFTER head labeling (same best-frame assignments)

### 10. `head_align_to_shoulders` (body_labeling)

- **Where**: `body_labeling.py` 1574–1598 (inside `match_markers_to_template_z_bands`).  
  After C7/shoulders, if `head_align_to_shoulders` is True and `side_lfhd < 0`, all four head labels are flipped L↔R: LFHD↔RFHD, LBHD↔RBHD.
- **Effect**: Default is False. When True, this flips L/R for all four head markers; can produce an effective LFHD/RBHD swap when combined with geometry (e.g. diagonal swap).

### 11. Second matching after Procrustes (body_labeling)

- **Where**: `body_labeling.py` 2322–2359.  
  After trunk Procrustes, `match_markers_to_template_z_bands(pts_f, template_aligned, ...)` is called again. That runs `_match_head_markers_first` again with the **aligned** template and same `lr_ap_from_walking` (same `d_back`). Then, if `aligned_rms` is better, either:
  - `assignments = assignments_aligned` (when `head_align_to_shoulders` is True), or  
  - head is preserved from the first call and the rest from `assignments_aligned` (when `head_align_to_shoulders` is False).
- **Effect**:  
  - When head is **not** preserved: second run can assign head differently (e.g. different `max_match_distance` vs. `template_aligned` dropping one head point, or band loop filling the 4th head label). That can change which point gets LFHD vs RBHD.  
  - When head **is** preserved (current default): first-call head is kept; second call can still produce a different head set in `assignments_aligned`, but those head entries are discarded. So the only way to get a wrong head after this block is if the “preserve head” logic is wrong (e.g. wrong set `HEAD_MARKER_SET` or list order).

### 12. Band loop re-assigning band 11 (body_labeling)

- **Where**: `body_labeling.py` 1677–1732.  
  For each band `b`, `labels_in_band = [lab for lab in template if label_to_band.get(lab,-1) == b and lab not in assigned_lab_global]`.  
  When `match_head_first` is True, head is already in `assigned_lab_global`, so for band 11 `labels_in_band` is empty and the loop does not reassign head.
- **Effect**: With `match_head_first=True`, band loop does **not** overwrite head. If `match_head_first` were False and head were assigned in the band loop, L/R and A/P there use `_classify_point_lr_ap` and `_classify_label_lr_ap` (and cost matrix); a bug there could in theory swap labels, but with current flow head is not assigned in the band loop when head-first is used.

### 13. `propagate_labels_temporal` (body_labeling)

- **Where**: `body_labeling.py` 2093–2154.  
  Propagates (point_index, label) from `frame_start` forward/backward by nearest neighbor. Same label is copied to the nearest point at the next frame; labels are not swapped or renamed.
- **Effect**: Does **not** swap LFHD and RBHD; only propagates whatever (pi, lab) came from the initial assignments.

### 14. `build_full_trajectory_matrix` and export (export.py, pipeline)

- **Where**: `export.py` 41–46: for each label in `body_labels_static`, find `pi` such that `label_per_frame_body[f, pi] == label` and write `points_dynamic_body[f, pi, :]` to that label’s column.  
  `body_labels_static` order comes from static (pipeline 177–188).
- **Effect**: Output column order is by `body_labels_static`; no swapping of LFHD/RBHD. Only way to get “swapped” in the file is if `label_per_frame` already has the wrong label on the wrong point index (i.e. the bug is before this step).

---

## Summary table

| # | Phase        | Location / cause                                      | Can cause LFHD/RBHD swap? |
|---|-------------|--------------------------------------------------------|----------------------------|
| 1 | Before      | `compute_walking_direction_x` returns wrong sign       | Yes (wrong A/P)            |
| 2 | Before      | Facing rotation: `d_back` original frame, points rotated | Yes (A/P axis mismatch)  |
| 3 | Before      | `points_d` includes obstacles → wrong centroid/wdx     | Yes (wrong A/P)            |
| 4 | Before      | Threshold 5.0 → wdx=0 vs ±1 borderline                  | Yes (wrong A/P)            |
| 5 | During      | Head A/P strictly from `d_back` when set               | Amplifies 1–4              |
| 6 | During      | `order_by_x` / ant_idx / post_idx                      | Unlikely (logic correct)   |
| 7 | During      | `assign_lr` / `left_side_positive_lr`                  | Would swap L/R pairs       |
| 8 | During      | Template vs static label string mismatch               | Unlikely (wrong column)    |
| 9 | During      | `head_swap_lfhd_rbhd` True                             | Yes (explicit swap)        |
|10 | After       | `head_align_to_shoulders` True                         | Yes (L/R flip)             |
|11 | After       | Second match overwrites head (or wrong preserve logic) | Yes                        |
|12 | After       | Band loop band 11                                      | No (head already assigned) |
|13 | After       | `propagate_labels_temporal`                            | No                         |
|14 | After       | `build_full_trajectory_matrix` / export               | No                         |

---

## Most likely causes for “LFHD/RBHD still switched” with current defaults

1. **Walking direction** (1, 3, 4): Wrong or borderline `wdx` → wrong `d_back` → wrong head A/P.  
2. **Frame mismatch** (2): If `--static-facing` / `--dynamic-facing` are used, `d_back` and head X are in different frames.  
3. **Second matching** (11): With `head_align_to_shoulders=False`, head is preserved from the first call; if the first call already had LFHD/RBHD wrong (e.g. due to 1 or 2), the preserved head keeps that swap.

No code changes were made; this is an analysis-only report.
