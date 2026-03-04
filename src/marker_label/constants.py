"""Constants for the marker labeling pipeline."""

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

# Trunk markers: pelvis + thorax + head (similar geometry in static T-pose and walking).
# Used for trunk-only best-frame selection and Procrustes alignment (see STATIC_T_POSE_VS_DYNAMIC_WALKING.md).
TRUNK_LABELS = (
    "LASI", "RASI", "LPSI", "RPSI", "SACR",  # pelvis
    "C7", "CLAV", "STRN", "T10",              # thorax
    "LFHD", "RFHD", "RBHD", "LBHD",          # head
)
