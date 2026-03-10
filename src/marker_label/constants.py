"""Constants for the marker labeling pipeline."""

# -----------------------------------------------------------------------------
# Whole-body 39 markers. Only these are used for Z-band by rank and body segment
# geometry built from static trial.
# -----------------------------------------------------------------------------
WHOLE_BODY_39 = (
    "RHEE", "LHEE", "LTOE", "RTOE", "LANK", "RANK", "LTIB", "RTIB",
    "RKNE", "LKNE", "LTHI", "RTHI", "RASI", "LASI", "RPSI", "LPSI",
    "LWRB", "RWRB", "LWRA", "RWRA", "RFIN", "LFIN", "RFRM", "RELB",
    "LFRM", "T10", "STRN", "LELB", "RUPA", "LUPA", "RBAK", "CLAV",
    "RSHO", "LSHO", "C7", "RBHD", "LBHD", "LFHD", "RFHD",
)
WHOLE_BODY_39_SET = frozenset(m.upper() for m in WHOLE_BODY_39)

# Arm and hand markers excluded from "no arm/hand" Z-band (12 markers).
ARM_HAND_MARKERS_EXCLUDED = (
    "LWRA", "LWRB", "LFIN", "RWRA", "RWRB", "RFIN",  # hand
    "LUPA", "LELB", "LFRM", "RUPA", "RELB", "RFRM",  # arm
)
ARM_HAND_EXCLUDED_SET = frozenset(m.upper() for m in ARM_HAND_MARKERS_EXCLUDED)

# Whole-body without arm/hand (27 markers) for Z-band by rank without arm/hand.
WHOLE_BODY_NO_ARM_HAND = tuple(
    m for m in WHOLE_BODY_39 if m.upper() not in ARM_HAND_EXCLUDED_SET
)
WHOLE_BODY_NO_ARM_HAND_SET = frozenset(m.upper() for m in WHOLE_BODY_NO_ARM_HAND)

# Z-band sizes for 27 markers (no arm/hand): band 6 = 0, band 8 = 1 (RBAK only), etc.
Z_BAND_SIZES_NO_ARM_HAND = (6, 2, 2, 2, 2, 2, 0, 2, 1, 1, 3, 4)

# Anatomical groups for no-arm-hand Z-band: assign by Z rank *within* each group so
# pelvis/leg/foot stay in correct bands. Each entry: (list of (band, size), list of labels).
# Pelvis 4 markers -> band 5 (top 2 by Z), band 4 (next 2). Leg/foot each get one band.
Z_BAND_GROUPS_NO_ARM_HAND = (
    # (bands with sizes), labels; bands in order high->low, labels sorted by Z desc within group
    (((11, 4),), ("LFHD", "RFHD", "RBHD", "LBHD")),
    (((10, 3),), ("C7", "LSHO", "RSHO")),
    (((9, 1),), ("CLAV",)),
    (((8, 1),), ("RBAK",)),
    (((7, 2),), ("STRN", "T10")),
    (((5, 2), (4, 2)), ("LPSI", "RPSI", "LASI", "RASI")),  # pelvis: higher Z -> band 5, next -> band 4
    (((3, 2),), ("LTHI", "RTHI")),   # thigh
    (((2, 2),), ("LKNE", "RKNE")),   # knee
    (((1, 2),), ("LTIB", "RTIB")),   # shank
    (((0, 6),), ("LANK", "RANK", "LHEE", "RHEE", "LTOE", "RTOE")),  # foot/ankle
)

# Z-band by rank: 12 bands from foot (0) to head (11). Band sizes used when
# building label_to_band from static (sort labels by Z descending, assign by size).
Z_BAND_SIZES = (6, 2, 2, 2, 2, 2, 6, 6, 3, 1, 3, 4)
N_Z_BANDS = 12

# Vicon Plug-in Gait pelvis markers (full-body set)
PELVIS_MARKERS = ("LASI", "RASI", "LPSI", "RPSI")
PELVIS_MARKER_OPTIONAL = ("SACR",)

# Obstacle marker labels
OBSTACLE_LABELS = ("OBSTACLE_L", "OBSTACLE_R")

# Default visibility threshold for obstacle candidates (fraction of frames)
DEFAULT_OBSTACLE_VISIBILITY_MIN = 0.80

# Default frame range for "middle" of trial (fraction of total frames)
DEFAULT_MIDDLE_START = 0.20
DEFAULT_MIDDLE_END = 0.80

# Dynamic trial initial screening (see docs/DYNAMIC_TRIAL_INITIAL_SCREENING_PLAN.md)
SCREENING_Y_MIN_MM = -1500
SCREENING_Y_MAX_MM = 1500
SCREENING_VISIBILITY_MIN = 0.50  # Step 2: drop columns with visibility < this
# Expected marker count after screening: 39 body + 2 obstacles (raise if different)
EXPECTED_SCREENED_MARKERS = 41  # 39 body + 2 obstacles
SCREENING_BEST_FRAME_VISIBILITY_MIN = 0.95  # Step 5: frame must have >= this fraction valid
SCREENING_FOOT_PROXY_N_SMALLEST_Z = 6  # Step 5: mean of N smallest Z as foot-height proxy

# Head markers (4): assigned at best frame by top 4 Z and L/R (Y), A/P (X).
HEAD_MARKERS = ("LFHD", "RFHD", "LBHD", "RBHD")
HEAD_MARKERS_SET = frozenset(m.upper() for m in HEAD_MARKERS)

# C7 and shoulders: assigned after head; among remaining points, top 1 Z = C7, next 2 Z = LSHO/RSHO (L/R by Y).
C7_SHOULDER_MARKERS = ("C7", "LSHO", "RSHO")
C7_SHOULDER_MARKERS_SET = frozenset(m.upper() for m in C7_SHOULDER_MARKERS)

# CLAV and RBAK: after C7/shoulders; next 2 Z among points with Y between shoulders ± offset; anterior = CLAV, posterior = RBAK.
CLAV_RBAK_MARKERS = ("CLAV", "RBAK")
CLAV_RBAK_MARKERS_SET = frozenset(m.upper() for m in CLAV_RBAK_MARKERS)
CLAV_RBAK_Y_OFFSET_MM = 15  # Allow Y in [min(LSHO_y, RSHO_y) - 15, max(...) + 15] to avoid arm markers.

# STRN, T10, LUPA, RUPA, LELB, RELB: next 6 highest Z after head/C7/shoulders/CLAV/RBAK; direct x/y/z comparison (no centroid).
STRN_T10_ARM_MARKERS = ("STRN", "T10", "LUPA", "RUPA", "LELB", "RELB")
STRN_T10_ARM_MARKERS_SET = frozenset(m.upper() for m in STRN_T10_ARM_MARKERS)

# Pelvis (4) + arm/hand (8): next 12 highest Z; Y in shoulder band → LASI,RASI,LPSI,RPSI (A/P then L/R); left/right 4 each → LFRM,LWRA,LWRB,LFIN and RFRM,RWRA,RWRB,RFIN.
PELVIS_ARM12_MARKERS = (
    "LASI", "RASI", "LPSI", "RPSI",
    "LFRM", "LWRA", "LWRB", "LFIN",
    "RFRM", "RWRA", "RWRB", "RFIN",
)
PELVIS_ARM12_MARKERS_SET = frozenset(m.upper() for m in PELVIS_ARM12_MARKERS)
# Error codes for pelvis/arm12 step (for logging / diagnostics)
PELVIS_ARM12_TOO_FEW_POINTS = "PELVIS_ARM12_TOO_FEW_POINTS"
PELVIS_BAND_NOT_4 = "PELVIS_BAND_NOT_4"
PELVIS_LEFT_NOT_4 = "PELVIS_LEFT_NOT_4"
PELVIS_RIGHT_NOT_4 = "PELVIS_RIGHT_NOT_4"
PELVIS_ARM_DUPLICATE_INDEX = "PELVIS_ARM_DUPLICATE_INDEX"
PELVIS_ARM_MISSING_GROUP = "PELVIS_ARM_MISSING_GROUP"
ARM_SIDE_TOO_FEW_FOR_Z = "ARM_SIDE_TOO_FEW_FOR_Z"
ARM_SIDE_INVALID_Z_ORDER = "ARM_SIDE_INVALID_Z_ORDER"

# Leg/foot (12): from remaining points, top 12 by Z; split L/R by CLAV Y; per-side Z order THI,KNE,TIB,ANK; remaining 2 per side A/P -> HEE,TOE.
LEG_FOOT12_MARKERS = (
    "LTHI", "RTHI", "LKNE", "RKNE", "LTIB", "RTIB", "LANK", "RANK",
    "LHEE", "LTOE", "RHEE", "RTOE",
)
LEG_FOOT12_MARKERS_SET = frozenset(m.upper() for m in LEG_FOOT12_MARKERS)
LEG_FOOT12_TOO_FEW_POINTS = "LEG_FOOT12_TOO_FEW_POINTS"
LEG_FOOT12_CLAV_INVALID = "LEG_FOOT12_CLAV_INVALID"
LEG_FOOT12_LR_NOT_6_6 = "LEG_FOOT12_LR_NOT_6_6"
LEG_FOOT12_SIDE_TOO_FEW = "LEG_FOOT12_SIDE_TOO_FEW"
LEG_FOOT12_SIDE_Z_INVALID = "LEG_FOOT12_SIDE_Z_INVALID"
LEG_FOOT12_FOOT_NOT_2 = "LEG_FOOT12_FOOT_NOT_2"
LEG_FOOT12_FOOT_AP_INVALID = "LEG_FOOT12_FOOT_AP_INVALID"
LEG_FOOT12_THIGH_PAIR_INVALID = "LEG_FOOT12_THIGH_PAIR_INVALID"
LEG_FOOT12_KNEE_PAIR_INVALID = "LEG_FOOT12_KNEE_PAIR_INVALID"
LEG_FOOT12_TIBIA_PAIR_INVALID = "LEG_FOOT12_TIBIA_PAIR_INVALID"
LEG_FOOT12_ANKLE_PAIR_INVALID = "LEG_FOOT12_ANKLE_PAIR_INVALID"
LEG_FOOT12_FOOT_NOT_4 = "LEG_FOOT12_FOOT_NOT_4"
LEG_FOOT12_FOOT_LR_SPLIT = "LEG_FOOT12_FOOT_LR_SPLIT"

# Trunk markers: pelvis + thorax + head (similar geometry in static T-pose and walking).
# Used for trunk-only best-frame selection and Procrustes alignment (see STATIC_T_POSE_VS_DYNAMIC_WALKING.md).
TRUNK_LABELS = (
    "LASI", "RASI", "LPSI", "RPSI", "SACR",  # pelvis
    "C7", "CLAV", "STRN", "T10",              # thorax
    "LFHD", "RFHD", "RBHD", "LBHD",          # head
)
