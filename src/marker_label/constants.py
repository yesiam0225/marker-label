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
