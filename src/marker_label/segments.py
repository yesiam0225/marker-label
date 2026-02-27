"""
Body segment definitions for visualization (and later analysis).

How to create segments
---------------------
A segment is an ordered list of marker names. Consecutive markers are
connected by line segments (sticks). Order = proximal to distal, or a
closed loop (e.g. pelvis: LASI -> RASI -> RPSI -> LPSI -> LASI).

1. Add a key to SEGMENTS (e.g. "MySegment") and set the value to a list
   of marker labels that exist in your C3D/CSV (e.g. ["LSHO", "LUPA", "LELB"]).
2. Only markers present in your data are drawn; missing markers skip that edge.
3. For a closed polygon, repeat the first marker at the end (e.g. ["A", "B", "C", "A"]).

Example: "L_Thigh" -> ["LASI", "LTHI", "LKNE"] draws LASI–LTHI and LTHI–LKNE.
"""

# Vicon full-body style: each segment is proximal -> distal (or closed loop).
# Use only marker names that exist in your labeled C3D/CSV; unknown markers are skipped.
SEGMENTS = {
    "Head": ["LFHD", "RFHD", "RBHD", "LBHD", "LFHD"],  # closed
    "Thorax": ["C7", "CLAV", "STRN", "T10", "C7"],     # closed
    "Pelvis": ["LASI", "RASI", "RPSI", "LPSI", "LASI"],  # closed
    "L_UpperArm": ["LSHO", "LUPA", "LELB"],
    "L_Forearm": ["LELB", "LFRM", "LWRA", "LWRB"],
    "L_Hand": ["LWRA", "LFIN"],  # or LWRB->LFIN
    "R_UpperArm": ["RSHO", "RUPA", "RELB"],
    "R_Forearm": ["RELB", "RFRM", "RWRA", "RWRB"],
    "R_Hand": ["RWRA", "RFIN"],
    "L_Thigh": ["LASI", "LTHI", "LKNE"],
    "L_Shank": ["LKNE", "LTIB", "LANK"],
    "L_Foot": ["LANK", "LHEE", "LTOE", "LANK"],  # closed
    "R_Thigh": ["RASI", "RTHI", "RKNE"],
    "R_Shank": ["RKNE", "RTIB", "RANK"],
    "R_Foot": ["RANK", "RHEE", "RTOE", "RANK"],  # closed
    # Bar between the two obstacle markers
    "Obstacle_Bar": ["OBSTACLE_L", "OBSTACLE_R"],
}


def segment_lines_for_frame(
    points: "np.ndarray",
    labels: list[str],
    segments: dict[str, list[str]] | None = None,
) -> tuple["np.ndarray", list[list[int]]]:
    """
    Build line geometry for one frame from segment definitions.

    Parameters
    ----------
    points : (n_markers, 3) one frame of marker positions (NaN = missing)
    labels : list of marker names, same order as points
    segments : segment name -> list of marker names; default SEGMENTS

    Returns
    -------
    line_points : (n_vertices, 3) concatenated endpoints for all drawn edges
    line_cells : list of [2, i, j] for each edge (VTK_LINE)
    """
    import numpy as np

    if segments is None:
        segments = SEGMENTS
    # Normalize: strip labels so "LFHD  " matches "LFHD"
    label_to_idx = {str(lab).strip(): i for i, lab in enumerate(labels)}
    line_points_list = []
    line_cells = []
    offset = 0
    for _name, marker_chain in segments.items():
        for k in range(len(marker_chain) - 1):
            a, b = marker_chain[k].strip(), marker_chain[k + 1].strip()
            ia = label_to_idx.get(a)
            ib = label_to_idx.get(b)
            if ia is None or ib is None:
                continue
            pa = points[ia]
            pb = points[ib]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            line_points_list.append(pa)
            line_points_list.append(pb)
            line_cells.append([2, offset, offset + 1])
            offset += 2
    if not line_points_list:
        return np.empty((0, 3), dtype=np.float64), np.array([], dtype=np.int32)
    line_points = np.array(line_points_list, dtype=np.float64)
    # VTK connectivity: [n_pts, id0, id1, n_pts, id2, id3, ...] for LINE cells
    cells_flat = np.array([x for cell in line_cells for x in cell], dtype=np.int32)
    return line_points, cells_flat
