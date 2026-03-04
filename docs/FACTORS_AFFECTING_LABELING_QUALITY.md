# Factors that affect marker labeling quality in the geometry pipeline

When marker labeling is **completely off** (almost all body markers wrong, not just a few swaps), the cause is usually one of the factors below. This document lists them in rough order of impact and how to check or fix each.

---

## 1. Unit mismatch (static vs dynamic)

**What happens:** If static C3D is in **mm** and dynamic is in **meters** (or vice versa), coordinates differ by 1000×. The template (from static) and dynamic points are on completely different scales. Every “closest” match is wrong; labeling is nonsensical.

**Check:** Look at numeric values in the static vs dynamic C3D (or CSV). Typical lab data in mm: hundreds to a few thousand. In meters: 0.3–2.0.

**Fix:** Use `--static-unit m` and/or `--dynamic-unit m` when the corresponding file is in meters. The pipeline then scales to mm internally so static and dynamic match.

**In your setup:** Only the **manually labeled dynamic** trials (e.g. reference: BBpilot01 Trial 10) are in meters. For **marker-label** on BBA01, both BBA01 Cal 01 (static) and BBA01 Trial 05 (dynamic) are in mm, so no `--static-unit` or `--dynamic-unit` is needed. Use `--dynamic-unit m` only when running **marker-label-analyze** on a reference pair whose dynamic file is in m.

---

## 2. Facing axis (orientation) wrong or not set

**What happens:** In static the subject faces one lab axis (e.g. **y**); in dynamic they face another (e.g. **x**). The template and dynamic point clouds are **rotated** relative to each other in the horizontal plane. Nearest-neighbor in lab frame then assigns “left” body markers to “right” template positions (or anterior/posterior swapped), so labels are systematically wrong.

**Check:** In your capture protocol, which way did the subject face during static vs dynamic? If they turned 90° (or 180°) between trials, you must set facing.

**Fix:** Use `--static-facing y --dynamic-facing x` (or the correct axes for your lab). Values: `x`, `-x`, `y`, `-y` (direction the subject faces). If you use the wrong direction (e.g. `-y` instead of `y`), labels can still be wrong or swapped.

---

## 3. Static C3D has wrong or generic labels

**What happens:** The template is built from the **labels** in the static file. If those labels are wrong (e.g. L/R swapped, or Point_1, Point_2, …), the template is wrong and every assignment is wrong. If the static has no real labels (only generic names), the pipeline still runs but the output names are meaningless.

**Check:** Open the static C3D in the viewer and confirm that each marker position matches the label (e.g. LFHD is on the left head, RASI on the right ASIS).

**Fix:** Correct the static C3D labels in your acquisition/editing software before running the pipeline. The pipeline does not verify static labels.

**In your setup:** Static trial labeling has been checked and is correct; this factor is ruled out for body labeling failure.

---

## 4. Obstacle detection picks body markers instead of the real obstacle

**What happens:** The pipeline finds the **2 most stationary** markers with high visibility and labels them OBSTACLE_L / OBSTACLE_R, then **removes them** from the body set. If two **body** markers (e.g. pelvis or trunk) are more stationary and visible than the real obstacle bar, they get removed. Then: (a) body matching has 2 fewer points and wrong correspondences; (b) two body markers are wrongly labeled OBSTACLE_L/R. Result can be a cascade of wrong assignments.

**Check:** After labeling, inspect the output: are the two “obstacle” markers actually on the bar, or on the body? If on the body, obstacle detection failed.

**Fix:** Raise `--obstacle-visibility` (e.g. 0.9) so only very visible markers are candidates; ensure the real obstacle has high visibility. Or lower it slightly if the real obstacle has some dropouts. You can also skip obstacle detection in code (e.g. force no obstacles) if you don’t have an obstacle in the trial.

---

## 5. Best frame is a bad frame for matching

**What happens:** The pipeline picks **one** “best” frame (in the middle range, by lowest residual or most valid points) and does **all** initial labeling there. If that frame has unusual pose (arms up, one leg forward), heavy occlusion, or bad residuals, the **initial assignment** is wrong. That wrong assignment is then **propagated** to every other frame, so the whole trial is wrong.

**Check:** Run with a fixed middle range or use `--reference-report` from a known-good analysis. Optionally add logging to print the chosen best frame index and inspect that frame in the viewer (compare to static pose).

**Fix:** Narrow the search window with `--middle-start` / `--middle-end` (e.g. 0.3–0.5) so the best frame is in a “neutral” part of the trial. Use `--reference-report` with an aggregated report from 24 trials so the window is centered where “template-closest” frames usually fall. If your pipeline exposes it, try a different criterion (e.g. frame closest to template by Procrustes RMS) instead of residual-only.

---

## 6. Initial match: pose too different from static

**What happens:** Even with correct units and facing, if at the best frame the subject’s **pose** is very different from the static (e.g. walking vs standing T-pose), the **relative** positions of markers don’t match the template. Nearest-neighbor (or Hungarian) then assigns the wrong label to many points (e.g. wrist to elbow). One wrong assignment propagates.

**Check:** Compare the best frame’s point cloud to the static template in the compare viewer (`marker-label-compare`). If the body shape is clearly different (arms/legs in different positions), matching in raw lab frame will be unreliable.

**Fix:** Ensure the best frame is chosen from a moment when pose is close to static (e.g. standing or neutral stride). Add **Procrustes alignment** at the best frame: fit a rigid transform from template to the dynamic cloud (e.g. from an initial assignment), then re-match in the aligned space so global rotation/translation doesn’t drive wrong matches.

---

## 7. Distance thresholds too loose or too tight

**What happens:**  
- **max_match_distance** too high: Clearly wrong point–template pairs are accepted at the best frame, so wrong labels are assigned and propagated.  
- **max_match_distance** too low: Valid assignments are rejected, so many markers get no label or a wrong one by exclusion.  
- **max_propagation_distance** too high: After a dropout or when two markers cross, the label “jumps” to the wrong point and stays wrong.  
- **max_propagation_distance** too low: Normal frame-to-frame movement is rejected, so labels don’t propagate and you get many NaNs.

**Check:** Inspect the output: many wrong labels from the first frame → initial match problem (units, facing, best frame, or max_match_distance). Wrong labels only after a certain time or after a gap → propagation problem (max_propagation_distance).

**Fix:** Set `--max-match-distance` (e.g. 200–350 mm) and `--max-propagation-distance` (e.g. 80–150 mm) from reference analysis (e.g. 2–3× min RMS from marker-label-analyze) or by trial and error. Use `--reference-report` so suggested values are filled when the reference has sensible RMS.

---

## 8. Point index identity across frames

**What happens:** The pipeline assumes that **point index i** in the dynamic C3D is the **same physical marker** on every frame. If the acquisition system **reorders** points (e.g. by x position or by detection order), index 5 at frame 0 might be a different marker than index 5 at frame 100. Propagation “follows” index, so labels become inconsistent or wrong.

**Check:** In the raw dynamic C3D, does point order stay fixed across frames (only positions change), or do indices get reordered? Some systems output a stable order; others don’t.

**Fix:** Preprocess the dynamic C3D so that point order is stable (e.g. sort by mean position or use system-specific tools). The pipeline does not reorder points.

---

## 9. Different marker set (static vs dynamic)

**What happens:** If the static has 43 markers and the dynamic has 50 (or different names), the template has 43 positions and the dynamic has 50 points. Matching assigns 43 labels to 50 points (or the pipeline uses body_labels_static and some dynamic points get no label). Mismatch in **which** markers were placed (e.g. static has LFHD but dynamic doesn’t) can cause wrong assignments when the closest template point is not the right one.

**Check:** Compare the marker list (or count) in the static vs dynamic. Ensure the same physical markers are present and that the static has correct labels for them.

**Fix:** Use a static that matches the dynamic trial’s marker set (same number and same anatomy). If some markers are optional, the pipeline can still run but NaNs or wrong matches may increase.

---

## 10. Residual or quality not available / misleading

**What happens:** Best frame is chosen by **lowest mean residual** (if C3D provides it) or by **most valid points**. If residual is missing or wrong (e.g. always zero), the pipeline falls back to “most valid points,” which might pick a frame with many points but bad pose. If residual is noisy or not representative of true quality, the “best” frame can be bad.

**Check:** Does your C3D contain a residual (4th column in point data)? Is it in the same unit as x,y,z (often mm)?

**Fix:** Rely more on **middle_start / middle_end** and reference report (best-frame fraction from 24 trials) so the chosen frame is in a known-good range rather than residual alone.

---

## Quick checklist when labeling is completely off

| Check | If wrong, likely cause |
|-------|------------------------|
| Static and dynamic in same unit (mm vs m)? | **Unit mismatch** → use --static-unit / --dynamic-unit |
| Subject facing same or corrected (--static-facing / --dynamic-facing)? | **Facing** → set axes correctly |
| Static C3D has correct, non-generic labels? | **Bad template** → fix static labels |
| Obstacle = real bar, not body markers? | **Obstacle detection** → adjust --obstacle-visibility or data |
| Best frame in a “neutral” pose? | **Best frame** → narrow middle range or use --reference-report |
| Best frame pose similar to static? | **Pose mismatch** → better best frame or Procrustes at match |
| max_match_distance / max_propagation_distance set? | **Thresholds** → set from reference or tune |
| Point order stable across frames in dynamic? | **Index reordering** → fix in preprocessing |
| Same marker set in static and dynamic? | **Marker set** → align marker lists |

---

## Summary

The most common reasons for **completely wrong** labeling in this geometry pipeline are:

1. **Unit mismatch** (m vs mm) between static and dynamic.  
2. **Facing axis** not set or wrong when the subject turned between static and dynamic.  
3. **Wrong or generic labels in the static** C3D.  
4. **Obstacle detection** taking two body markers instead of the real obstacle.  
5. **Bad best frame** (wrong pose or quality), leading to wrong initial assignment that propagates everywhere.  
6. **Pose at best frame** too different from static so that nearest-neighbor in lab frame gives wrong labels.  
7. **Distance thresholds** (max_match_distance, max_propagation_distance) too loose or too tight.  
8. **Point index** not stable across frames in the dynamic C3D.

Fixing (1)–(2) and verifying (3)–(4) usually addresses “completely off” results; (5)–(8) improve remaining quality.
