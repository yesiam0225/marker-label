"""Constants for the marker labeling pipeline."""

# -----------------------------------------------------------------------------
# Whole-body marker set (39 markers) — canonical set for whole-body labeling.
# See docs/WHOLE_BODY_39_MARKERS.md and analysis out/BBpilot01_whole_body_39_behavior.md.
# -----------------------------------------------------------------------------
WHOLE_BODY_39 = (
    "RHEE", "LHEE", "LTOE", "RTOE", "LANK", "RANK", "LTIB", "RTIB",
    "RKNE", "LKNE", "LTHI", "RTHI", "RASI", "LASI", "RPSI", "LPSI",
    "LWRB", "RWRB", "LWRA", "RWRA", "RFIN", "LFIN", "RFRM", "RELB",
    "LFRM", "T10", "STRN", "LELB", "RUPA", "LUPA", "RBAK", "CLAV",
    "RSHO", "LSHO", "C7", "RBHD", "LBHD", "LFHD", "RFHD",
)

# Anchor markers: most rigid in static→dynamic (BBpilot01 analysis).
# Use for establishing orientation / rigid fit first; trust these matches more.
WHOLE_BODY_ANCHOR_MARKERS = (
    "RSHO", "LSHO", "STRN", "T10", "CLAV", "RBAK", "C7",
    "RUPA", "LUPA", "LPSI", "RPSI",
)

# Reference RMS from BBpilot01 (39 markers only, manually labeled static + dynamic).
# Use to set distance thresholds when matching static→dynamic (e.g. max_match_distance).
REFERENCE_39_MIN_RMS_MM = 173.5   # min over frames (best frame)
REFERENCE_39_MEAN_RMS_MM = 189.0  # mean over frames

# Suggested max match distance (mm): allow ~2.3× reference min RMS for initial assignment.
SUGGESTED_MAX_MATCH_DISTANCE_39_MM = 400.0

# Minimum caps (mm) for tiered matching so that small s does not over-reject (e.g. head markers).
# Effective cap = max(alpha * s, MIN_CAP_*_MM).
MIN_CAP_ANCHOR_MM = 180.0
MIN_CAP_MID_MM = 380.0   # head/neck/upper trunk often far after anchor-only alignment
MIN_CAP_DISTAL_MM = 380.0

# Fallback: when tiered match leaves labels unassigned, assign to closest point within this (mm).
# Head markers can be far from template after anchor-only Procrustes.
FALLBACK_CAP_MM = 600.0

# Z as first criterion when matching points to labels within a band:
# cost = Z_WEIGHT_FOR_LABELING * |Z_point - Z_template| + distance (so Z dominates, then L/R, midline, A/P).
Z_WEIGHT_FOR_LABELING = 10.0  # 1 mm Z difference = 10 mm in cost

# Scalable rule: multipliers for subject-adaptive scale s (see docs/SCALABLE_LABELING_RULES_XYZ.md).
# Match/propagation caps = alpha * s where s = RMS at best frame from anchor fit.
ALPHA_ANCHOR = 1.2   # strict for anchor labels
ALPHA_MID = 2.3      # default for mid-tier
ALPHA_DISTAL = 3.0   # loose for distal (hand, foot, etc.)
ALPHA_PROP = 1.4     # max_propagation_distance = ALPHA_PROP * s

# Tier for each of the 39: anchor / mid / distal (for tiered max_match_distance).
WHOLE_BODY_ANCHOR_SET = frozenset(m.upper() for m in WHOLE_BODY_ANCHOR_MARKERS)
WHOLE_BODY_DISTAL_LABELS = frozenset((
    "LWRB", "RWRB", "LWRA", "RWRA", "RFIN", "LFIN", "RFRM", "LFRM",
    "RTOE", "LTOE", "RTIB", "LTIB", "LTHI", "RTHI", "RANK", "LANK",
    "RKNE", "LKNE", "LHEE", "RHEE",
))
# Mid = rest of 39 that are not anchor and not distal
WHOLE_BODY_MID_LABELS = frozenset(
    m.upper() for m in WHOLE_BODY_39
    if m.upper() not in WHOLE_BODY_ANCHOR_SET and m.upper() not in WHOLE_BODY_DISTAL_LABELS
)

# -----------------------------------------------------------------------------
# Vicon Plug-in Gait pelvis markers (full-body set)
# -----------------------------------------------------------------------------
PELVIS_MARKERS = ("LASI", "RASI", "LPSI", "RPSI")
PELVIS_MARKER_OPTIONAL = ("SACR",)

# Obstacle marker labels
OBSTACLE_LABELS = ("OBSTACLE_L", "OBSTACLE_R")

# Default visibility threshold for obstacle candidates (fraction of frames)
DEFAULT_OBSTACLE_VISIBILITY_MIN = 0.80

# Dynamic trial: drop markers (points) visible in less than this fraction of frames (default 0.5 = 50%)
DEFAULT_DYNAMIC_VISIBILITY_MIN = 0.50

# Default frame range for "middle" of trial (fraction of total frames)
DEFAULT_MIDDLE_START = 0.20
DEFAULT_MIDDLE_END = 0.80

# Best frame: only consider frames where at least this fraction of markers (points) are visible/valid
DEFAULT_VISIBILITY_MIN_FRACTION = 0.90

# Z-band matching: number of vertical bands (head → foot). Points and template labels
# are assigned to bands by Z; matching is done only within each band.
# Use 12 bands so the former single band (arms, trunk, CLAV) splits into more bands.
N_Z_BANDS = 12

# Fixed Z-band assignment for labeling: band number -> labels (39 whole-body markers).
# Z values differ across subjects; only band index and label membership are fixed.
# Used so the same band structure applies to every subject; boundaries are computed
# per subject from template Z (max Z in band b and min Z in band b+1).
Z_BAND_LABELS = (
    ("RHEE", "LHEE", "LTOE", "RTOE", "LANK", "RANK"),           # 0: foot/ankle
    ("LTIB", "RTIB"),                                           # 1: shank
    ("RKNE", "LKNE"),                                           # 2: knee
    ("LTHI", "RTHI"),                                           # 3: thigh
    ("RASI", "LASI"),                                           # 4: pelvis
    ("RPSI", "LPSI"),                                           # 5: pelvis
    ("LWRB", "RWRB", "LWRA", "RWRA", "RFIN", "LFIN"),           # 6: wrist/hand
    ("RFRM", "RELB", "LFRM", "T10", "STRN", "LELB"),            # 7: forearm/trunk
    ("RUPA", "LUPA", "RBAK"),                                   # 8: upper arm/back
    ("CLAV",),                                                  # 9: clavicle
    ("RSHO", "LSHO", "C7"),                                     # 10: shoulder/neck
    ("RBHD", "LBHD", "LFHD", "RFHD"),                           # 11: head
)

# Label -> band index for the 39 whole-body set (for lookup during labeling).
Z_BAND_LABEL_TO_BAND = {
    lab: b for b, labels in enumerate(Z_BAND_LABELS) for lab in labels
}

# Trunk markers: pelvis + thorax + head (similar geometry in static T-pose and walking).
# Used for trunk-only best-frame selection and Procrustes alignment (see STATIC_T_POSE_VS_DYNAMIC_WALKING.md).
TRUNK_LABELS = (
    "LASI", "RASI", "LPSI", "RPSI", "SACR",  # pelvis
    "C7", "CLAV", "STRN", "T10",              # thorax
    "LFHD", "RFHD", "RBHD", "LBHD",          # head
)

# For RBAK-based L/R and anterior/posterior: labels used to find thorax band (PCA for RBAK/T10)
THORAX_LABELS_FOR_RBAK = ("RBAK", "T10", "STRN", "C7", "CLAV")
# Head markers (highest Z); can be matched first using X for L/R and Y for A/P (forward walking).
HEAD_MARKER_LABELS = ("LFHD", "RFHD", "LBHD", "RBHD")
HEAD_MARKER_SET = frozenset(m.upper() for m in HEAD_MARKER_LABELS)
# Next band below head in Z-band-by-gap: match using midline from head geometry.
NECK_SHOULDER_LABELS = ("RSHO", "LSHO", "C7")
NECK_SHOULDER_SET = frozenset(m.upper() for m in NECK_SHOULDER_LABELS)
# Same band may include CLAV (smaller Z than shoulders/C7); match CLAV first by min Z.
NECK_SHOULDER_CLAV_LABELS = ("RSHO", "LSHO", "C7", "CLAV")
NECK_SHOULDER_CLAV_SET = frozenset(m.upper() for m in NECK_SHOULDER_CLAV_LABELS)
# Anatomically posterior (back) and anterior (front) for candidate filtering
POSTERIOR_LABELS = frozenset(("RBAK", "T10", "RPSI", "LPSI", "RBHD", "LBHD"))
ANTERIOR_LABELS = frozenset(("STRN", "CLAV", "LASI", "RASI", "LFHD", "RFHD"))

# Trunk markers matched after C7 and shoulders (Z-band order: CLAV band 9, RBAK 8, T10/STRN 7)
TRUNK_AFTER_SHOULDERS_LABELS = ("CLAV", "RBAK", "STRN", "T10")
TRUNK_AFTER_SHOULDERS_SET = frozenset(m.upper() for m in TRUNK_AFTER_SHOULDERS_LABELS)

# Arm and hand markers: matched by geometry (proximal-to-distal chain from shoulder), not by static Z
ARM_HAND_LABELS = frozenset((
    "LUPA", "RUPA", "LELB", "RELB", "LFRM", "RFRM",
    "LWRA", "RWRA", "LWRB", "RWRB", "LFIN", "RFIN",
))
# Proximal-to-distal order for chain matching (shoulder already matched in phase 1)
LEFT_ARM_CHAIN_ORDER = ("LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN")
RIGHT_ARM_CHAIN_ORDER = ("RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN")
