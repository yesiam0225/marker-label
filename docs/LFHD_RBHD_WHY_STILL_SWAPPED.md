# Why LFHD and RBHD Can Still Be Swapped (No Flips)

## Possible reason: wrong anterior/posterior from walking direction

L/R is from walking direction and we do **no flip** in labeling. The only way LFHD and RBHD get swapped (without swapping all four head labels) is **anterior vs posterior** being reversed for the head.

### How head A/P is decided (`_match_head_markers_first`)

- **Larger X = anterior** (front) when:
  - `head_anterior_larger_x` is True, or
  - `d_back` is not used for “smaller X = anterior”, i.e. `d_back[0] <= 0`.
- **Smaller X = anterior** when:
  - `head_anterior_smaller_x` is True, or
  - `d_back[0] > 0` (posterior direction is +X).

`d_back` comes from **walking direction**:

- `compute_walking_direction_x(points)` uses centroid X over time:
  - `mean_dx > 5` mm/frame → walking toward +X → `d_back = [-1, 0]` → **larger X = anterior**.
  - `mean_dx < -5` → walking toward -X → `d_back = [1, 0]` → **smaller X = anterior**.
  - Otherwise → unclear → default `d_back = [-1, 0]` → **larger X = anterior**.

So if the trial actually has **subject walking +X** (larger X = anterior) but centroid X **decreases** (e.g. short trial, noise, or **body-only centroid** in the pipeline after obstacle removal), then:

- `compute_walking_direction_x` returns **-1**,
- `d_back[0] > 0` → code uses **smaller X = anterior**,
- Result: (smaller X, left) gets LFHD and (larger X, right) gets RBHD → **LFHD and RBHD are swapped** compared to anatomy.

### What to do

1. **Check walking direction** for the dynamic trial:
   ```bash
   python scripts/report_lr_ap_axes.py "data/BBA01 Trial 05.c3d"
   ```
   If it reports “-X (subject walks toward decreasing X)” but you know the subject walks +X, the pipeline is using the wrong A/P for the head.

2. **Force larger X = anterior for head** (override `d_back` for head only):
   ```bash
   python -m marker_label.cli 'data/BBA01 Cal 01.c3d' 'data/BBA01 Trial 05.c3d' \
     --z-band-by-rank --left-side-positive-y --head-anterior-larger-x -o out/BBA01_trial05
   ```
   This makes head use **larger X = anterior** regardless of `d_back`, so LFHD/RBHD should match anatomy when the subject walks +X.

### Code locations

- **Walking direction**: `body_labeling.compute_walking_direction_x` (centroid X trend).
- **d_back**: `body_labeling.get_lr_ap_axes_from_walking(walking_direction_x, ...)`.
- **Head A/P**: `body_labeling._match_head_markers_first` (around 827–838): `head_anterior_smaller_x`, `head_anterior_larger_x`, `d_back[0] > 0` → `ant_idx` / `post_idx` → LFHD/RFHD vs LBHD/RBHD.

No other flip or L/R logic changes head labels after this; the only remaining cause for a pure LFHD↔RBHD swap is this A/P choice.

---

## Why "LFHD/RBHD swapped" appeared *after* we started fixing trunk L/R

**Symptom:** LBHD and RFHD are correct; only LFHD and RBHD are switched.

**Why trunk "fix" can be related (without any code that directly swaps LFHD↔RBHD):**

1. **Trunk L/R fix does not touch head A/P directly**  
   Trunk L/R is fixed with `--left-side-positive-y` (or similar), which only flips `d_right`. Head A/P uses `d_back` from walking direction; `d_back` is **not** changed by `left_side_positive_lr`. So there is no code path where "fixing trunk L/R" directly swaps only LFHD and RBHD.

2. **What can still cause LFHD/RBHD swap when you changed the pipeline for trunk**  
   - **Facing-axis rotation**  
     If you added `--static-facing` / `--dynamic-facing` when fixing trunk, dynamic points are **rotated** for matching, but `wdx` and `d_back` are still computed from **unrotated** `points_d`. So head A/P is decided using `d_back` (lab frame) while head point X is in the **rotated** frame. That mismatch can make "anterior vs posterior" wrong and produce exactly the "LFHD and RBHD swapped" pattern (see *LFHD_RBHD_ALL_POSSIBLE_REASONS.md* §2).  
   - **Walking direction**  
     Any change that affects which points go into `points_d` or centroid (e.g. obstacle handling, trimming) can change `wdx`. Wrong or borderline `wdx` → wrong `d_back` → wrong head A/P → LFHD/RBHD swap (same doc §1, 3, 4).

3. **Why only LFHD and RBHD swap (LBHD and RFHD correct)**  
   Head logic: split 4 points by **X** into anterior (2) and posterior (2), then assign L/R by **Y** in each group.  
   If the A/P split is wrong in a "cross" way (one anterior and one posterior point swap groups), you get:
   - Anterior group = [anterior-right, posterior-right] → L/R gives RFHD correct, LFHD on posterior-right (wrong).
   - Posterior group = [posterior-left, anterior-left] → L/R gives LBHD correct, RBHD on anterior-left (wrong).  
   So **LBHD and RFHD stay correct; LFHD and RBHD are swapped**.

**What to do**

- If you use facing rotation: try without it for this trial, or add `--head-anterior-larger-x` / `--head-anterior-smaller-x` to override head A/P.
- Check walking direction (e.g. `scripts/report_lr_ap_axes.py`) and, if wrong, use `--head-anterior-larger-x` when subject walks +X.
- Quick fix for this trial: `--head-swap-lfhd-rbhd` to swap only LFHD and RBHD after assignment.
