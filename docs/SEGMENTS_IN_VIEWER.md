# How segments are composed in the video (marker-label-view)

The 3D viewer draws **markers** (spheres) and **segment lines** (sticks) between certain markers. This document describes what the lines represent and which markers are linked.

---

## What the lines indicate

- Each **line** connects **two markers** that belong to the same **body segment** (e.g. upper arm, thigh, pelvis).
- Together, the lines form a **stick figure** (skeleton) over the marker cloud: they show the **segment structure** (which marker is connected to which) in a Vicon-style full-body model.
- Lines are drawn **only** when both endpoints have **valid** (non-NaN) positions in that frame; if a marker is missing, that edge is skipped for that frame.
- The **color** of the lines is set by `--segment-color` (default: darkblue). It is the same for all segments.

So: **lines = edges of the segment model**; they indicate “this marker is linked to that marker” for visualization (and later analysis). They do **not** indicate velocity, force, or any derived quantity—only connectivity.

---

## Which markers are linked (segment definitions)

Segments are defined in `marker_label.segments.SEGMENTS`. Each segment is an **ordered list** of marker names; **consecutive** markers in the list are connected by a line. Order is **proximal to distal** (e.g. shoulder → elbow → wrist), or a **closed loop** (first and last marker are the same so the polygon closes).

The table below lists every segment, the marker chain, and the resulting **links** (which markers are connected by a line).

| Segment name   | Marker chain (order) | Lines drawn (consecutive pairs) |
|----------------|----------------------|----------------------------------|
| **Head**       | LFHD → RFHD → RBHD → LBHD → LFHD | LFHD–RFHD, RFHD–RBHD, RBHD–LBHD, LBHD–LFHD (closed) |
| **Thorax**     | C7 → CLAV → STRN → T10 → RBAK → C7 | C7–CLAV, CLAV–STRN, STRN–T10, T10–RBAK, RBAK–C7 (closed) |
| **Pelvis**     | LASI → RASI → RPSI → LPSI → LASI | LASI–RASI, RASI–RPSI, RPSI–LPSI, LPSI–LASI (closed) |
| **L_UpperArm** | LSHO → LUPA → LELB | LSHO–LUPA, LUPA–LELB |
| **L_Forearm**  | LELB → LFRM → LWRA → LWRB | LELB–LFRM, LFRM–LWRA, LWRA–LWRB |
| **L_Hand**     | LWRA → LWRB → LFIN | LWRA–LWRB, LWRB–LFIN |
| **R_UpperArm** | RSHO → RUPA → RELB | RSHO–RUPA, RUPA–RELB |
| **R_Forearm**  | RELB → RFRM → RWRA → RWRB | RELB–RFRM, RFRM–RWRA, RWRA–RWRB |
| **R_Hand**     | RWRA → RWRB → RFIN | RWRA–RWRB, RWRB–RFIN |
| **L_Thigh**    | LASI → LTHI → LKNE | LASI–LTHI, LTHI–LKNE |
| **L_Shank**    | LKNE → LTIB → LANK | LKNE–LTIB, LTIB–LANK |
| **L_Foot**     | LANK → LHEE → LTOE → LANK | LANK–LHEE, LHEE–LTOE, LTOE–LANK (closed) |
| **R_Thigh**    | RASI → RTHI → RKNE | RASI–RTHI, RTHI–RKNE |
| **R_Shank**    | RKNE → RTIB → RANK | RKNE–RTIB, RTIB–RANK |
| **R_Foot**     | RANK → RHEE → RTOE → RANK | RANK–RHEE, RHEE–RTOE, RTOE–RANK (closed) |
| **Obstacle_Bar** | OBSTACLE_L → OBSTACLE_R | OBSTACLE_L–OBSTACLE_R |

---

## Marker name abbreviations (Vicon-style)

- **Head:** LFHD (left front head), RFHD, RBHD (right back), LBHD.
- **Thorax:** C7 (7th cervical), CLAV (clavicle), STRN (sternum), T10 (10th thoracic).
- **Pelvis:** LASI, RASI (anterior superior iliac spine), LPSI, RPSI (posterior).
- **Arms:** LSHO/RSHO (shoulder), LUPA/RUPA (upper arm), LELB/RELB (elbow), LFRM/RFRM (forearm), LWRA/LWRB, RWRA/RWRB (wrist), LFIN/RFIN (finger).
- **Legs:** LTHI/RTHI (thigh), LKNE/RKNE (knee), LTIB/RTIB (tibia), LANK/RANK (ankle), LHEE/RHEE (heel), LTOE/RTOE (toe).
- **Obstacle:** OBSTACLE_L, OBSTACLE_R (the two detected obstacle markers).

---

## How it’s used in the viewer

- For **each frame**, the viewer calls `segment_lines_for_frame(points[f], labels)`.
- It looks up each marker name in the segment chains in the **labels** of the loaded file; if a marker is not present, that edge is skipped.
- So if your labeled C3D/CSV uses the same marker names as in `SEGMENTS`, you see the corresponding sticks; if some names differ or markers are missing, only the edges that exist and are valid are drawn.

No lines are drawn for markers that are not part of any segment definition or that have no valid position in that frame.
