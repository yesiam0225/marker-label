# Static T-pose vs dynamic walking: how to match when poses differ

When the **static** trial is captured with **arms spread to the sides** (T-pose) and the **dynamic** trial is **walking**, there is no frame in the dynamic trial where the subject has the same pose as the static. So “find the frame closest to the template” is ill-posed for the whole body. This document suggests ways to overcome that.

---

## Why it’s a problem

- **Template** = mean positions from static → arms out (T-pose), trunk upright, **legs typically together or neutral**.
- **Dynamic** = walking → arms swing, trunk may lean slightly, **legs spread forward and backward** (one leg in front, one behind).
- If we pick a “best” frame by **lowest residual** or **minimum distance to full template**, we are still comparing a standing template to a walking frame; **arm** and **leg** positions never match well, so:
  - Best-frame choice can be driven by noise or by one body part.
  - Initial assignment at that frame can be wrong for arms and legs (and wrong labels then propagate).

So we should **not** require any frame in the dynamic trial to look like the full static pose (arms out, legs together).

---

## Main idea: trunk-first, then propagate limbs

- **Trunk** (pelvis, spine, head, optionally shoulders) has **similar relative geometry** in T-pose and in walking (same person, upright). So we can still “align” the trunk between template and a walking frame.
- **Limbs** (arms, legs) change a lot between static and walking: arms go from out to swinging; **legs go from together to spread (one forward, one back)**. We don’t need a frame where limbs match the static pose; we need a frame where **trunk** matches well, then we **match the whole body once** at that frame and **propagate** limb labels over time (so limbs get the right label by continuity, not by pose similarity).

Concrete suggestions:

---

## 1. Best frame by “trunk only” (not full body)

**Current behaviour:** Best frame = lowest mean residual or most valid points in the middle range. That does not target “trunk closest to template.”

**Proposed:** For each candidate frame in the middle range:

1. Run a **provisional** match (e.g. Hungarian) between template and that frame’s points.
2. From that assignment, keep only pairs where the **template label** is in a **trunk set** (e.g. pelvis + spine + head: LASI, RASI, LPSI, RPSI, C7, CLAV, STRN, T10, LFHD, RFHD, RBHD, LBHD — or a subset you have).
3. Compute a **trunk cost** for that frame (e.g. sum of distances for those pairs, or RMS of those distances).
4. Choose the frame with **minimum trunk cost** as the best frame.

So we choose “the frame where the **trunk** aligns best with the template,” even if arms and legs are wrong in the provisional match. We never require arms to be in T-pose or legs to be together.

**Implementation note:** You need a fixed list of “trunk” labels (e.g. in `constants.py` as `TRUNK_LABELS`). Only template labels in that set participate in the trunk cost.

---

## 2. Procrustes at best frame fitted on trunk only

**Current behaviour:** No Procrustes; matching is in lab frame (with optional facing rotation).

**Proposed:** At the chosen best frame:

1. Get an **initial** assignment (e.g. Hungarian, template vs dynamic points).
2. From that assignment, take only the **trunk** pairs (template label in trunk set).
3. Fit **Procrustes** (R, t) from **template trunk positions** to **assigned dynamic positions** for those pairs.
4. Apply that **same** R, t to the **full** template (all labels).
5. **Re-match** full template (now aligned) to dynamic points (e.g. Hungarian again).
6. Optionally iterate once more (re-fit Procrustes on trunk from new assignment, transform, re-match).

So the rigid transform is driven by **trunk** only; arms and legs in the template move with the body but their static positions (T-pose arms, legs together) are just rotated/translated. The re-match then assigns each dynamic point to the nearest transformed template point. **Limbs** (arms and legs) may still match to the correct side and segment because the **pelvis/trunk is already aligned**: in the aligned frame, “left” and “right” are consistent, so the forward leg (e.g. left) stays on the left side of the body and tends to be closest to the left-leg template points; the back leg (right) to the right-leg template points. So we do **not** require the legs to be in the same pose as static (together); we only require the trunk to align, then leg assignment follows by nearest neighbor in that aligned space.

---

## 3. Rely on temporal propagation for limbs

Once we have **one** frame with a reasonable assignment (thanks to trunk-based best frame and trunk-based Procrustes), **temporal propagation** carries every label forward and backward. So arm and leg labels are maintained by “same index = same marker over time,” not by finding a frame where arms are in T-pose or legs are together. We only need the **trunk** to be well aligned at the best frame; limbs (arms and legs) follow by propagation.

---

## 4. Optional: two-stage matching (trunk then limbs)

If one global match at the trunk-aligned frame still mixes up limbs, a more involved option is:

- **Stage 1:** Match only **trunk** labels (e.g. restrict template to trunk set; assign only those; or use a RANSAC-style fit of trunk template to a subset of dynamic points).
- **Stage 2:** Fix those trunk assignments; for the remaining dynamic points, assign **limb** labels (e.g. by nearest neighbour to the remaining template positions in the aligned space, or by segment: “points near this shoulder are arm,” etc.).

This is more complex and only worth it if (1)+(2)+(3) still fail on limbs.

---

## Summary

| Step | Change |
|------|--------|
| Best frame | Choose frame where **trunk** assignment cost is minimum (not full body, not residual). |
| At best frame | Fit Procrustes on **trunk** pairs only; apply R,t to full template; re-match full body. |
| After that | Use existing temporal propagation; no need for any frame to have T-pose arms or legs together. |

This way we **don’t** require the same pose (T-pose arms, legs together) in the dynamic trial; we only require that **trunk** geometry is consistent enough to align once, and limbs (arms and legs) get their labels by propagation and by nearest-neighbor in the trunk-aligned space at the best frame.
