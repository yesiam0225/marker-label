"""Machine-readable pipeline errors (code + step) for CLI and automation."""

from __future__ import annotations

# Drop loaded columns / auto-drop almost-empty (before screening)
ERR_DROP_COLUMN_OUT_OF_RANGE = "DROP_COLUMN_OUT_OF_RANGE"
ERR_DROP_COLUMN_REMOVE_ALL = "DROP_COLUMN_REMOVE_ALL"
ERR_AUTO_DROP_REMOVE_ALL = "AUTO_DROP_REMOVE_ALL"
ERR_AUTO_DROP_INVALID_THRESHOLD = "AUTO_DROP_INVALID_THRESHOLD"
ERR_EXTRA_STATIONARY_INVALID = "EXTRA_STATIONARY_INVALID"
ERR_EXTRA_STATIONARY_REMOVE_ALL = "EXTRA_STATIONARY_REMOVE_ALL"

# Skip label names (after column drop; applied during labeling / export)
ERR_SKIP_LABEL_UNKNOWN = "SKIP_LABEL_UNKNOWN"
ERR_SKIP_LABEL_OBSTACLE = "SKIP_LABEL_OBSTACLE"
ERR_SKIP_LABEL_EMPTY_TEMPLATE = "SKIP_LABEL_EMPTY_TEMPLATE"


class LabelingPipelineError(Exception):
    """
    Configuration or pipeline failure with a stable ``error_code`` and ``step`` id.

    Parameters
    ----------
    error_code : short identifier (e.g. ``SKIP_LABEL_UNKNOWN``)
    message : human-readable detail
    step : pipeline region, e.g. ``drop_columns``, ``labeling``, ``export``
    """

    def __init__(self, error_code: str, message: str, *, step: str = "pipeline") -> None:
        self.error_code = error_code
        self.step = step
        self.message = message
        super().__init__(f"[{error_code}] ({step}) {message}")
