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

import numpy as np

# Vicon full-body style: each segment is proximal -> distal (or closed loop).
# Use only marker names that exist in your labeled C3D/CSV; unknown markers are skipped.
SEGMENTS = {
    "Head": ["LFHD", "RFHD", "RBHD", "LBHD", "LFHD"],  # closed
    "Thorax": ["C7", "CLAV", "STRN", "T10", "RBAK", "C7"],  # closed: C7, T10, RBAK, CLAV, STRN
    "Pelvis": ["LASI", "RASI", "RPSI", "LPSI", "LASI"],  # closed
    "L_UpperArm": ["LSHO", "LUPA", "LELB"],
    "L_Forearm": ["LELB", "LFRM", "LWRA", "LWRB"],
    "L_Forearm_FRM_WRB": ["LFRM", "LWRB"],  # extra line: FRM–WRB
    "L_Hand": ["LWRA", "LWRB", "LFIN"],
    "L_Hand_WRA_FIN": ["LWRA", "LFIN"],  # extra line: WRA–FIN
    "R_UpperArm": ["RSHO", "RUPA", "RELB"],
    "R_Forearm": ["RELB", "RFRM", "RWRA", "RWRB"],
    "R_Forearm_FRM_WRB": ["RFRM", "RWRB"],  # extra line: FRM–WRB
    "R_Hand": ["RWRA", "RWRB", "RFIN"],
    "R_Hand_WRA_FIN": ["RWRA", "RFIN"],  # extra line: WRA–FIN
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


def segment_lines_for_frame_by_segment(
    points: "np.ndarray",
    labels: list[str],
    segments: dict[str, list[str]] | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Build line geometry per segment for one frame. Returns list of (segment_name, line_points, line_cells)
    so the viewer can draw each segment with a different color.
    """
    import numpy as np

    if segments is None:
        segments = SEGMENTS
    label_to_idx = {str(lab).strip(): i for i, lab in enumerate(labels)}
    out = []
    for seg_name, marker_chain in segments.items():
        line_points_list = []
        line_cells = []
        offset = 0
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
        if line_points_list:
            line_points = np.array(line_points_list, dtype=np.float64)
            cells_flat = np.array([x for cell in line_cells for x in cell], dtype=np.int32)
            out.append((seg_name, line_points, cells_flat))
    return out


# Default colors per segment for the viewer (name -> color string or hex).
# Segments not listed get "gray". PyVista accepts "red", "blue", "#aabbcc", etc.
SEGMENT_COLORS = {
    "Head": "#1f77b4",
    "Thorax": "#ff7f0e",
    "Pelvis": "#2ca02c",
    "L_UpperArm": "#d62728",
    "L_Forearm": "#9467bd",
    "L_Forearm_FRM_WRB": "#9467bd",
    "L_Hand": "#8c564b",
    "L_Hand_WRA_FIN": "#8c564b",
    "R_UpperArm": "#e377c2",
    "R_Forearm": "#7f7f7f",
    "R_Forearm_FRM_WRB": "#7f7f7f",
    "R_Hand": "#bcbd22",
    "R_Hand_WRA_FIN": "#bcbd22",
    "L_Thigh": "#17becf",
    "L_Shank": "#aec7e8",
    "L_Foot": "#ffbb78",
    "R_Thigh": "#98df8a",
    "R_Shank": "#ff9896",
    "R_Foot": "#c5b0d5",
    "Obstacle_Bar": "#c7c7c7",
}
