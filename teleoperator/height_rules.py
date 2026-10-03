"""Rules for 3D targets near the minimum height; pure functions of the model and the pose."""

import numpy as np

from kinematics import Kinematics

# Targets stop this far above the height limit, so solver rounding cannot cross it (metres).
TARGET_HEIGHT_MARGIN = 0.001
# The arm sags this far below the limit while holding; such a tool still moves normally (metres).
SAG_ALLOWANCE = 0.010
# Largest rise per target below the height limit, about 20 mm/s at the control rate: a jump would
# leave the joints behind and trip the tracking check (metres).
MAX_RISE_PER_TARGET = 0.0007


def must_rise_first(model: Kinematics, angles: np.ndarray) -> bool:
    """Return True far below the height limit, as at the rest pose: only lifting straight up works."""
    return model.forward(angles)[2, 3] < model.min_height - SAG_ALLOWANCE


def allowed_target(model: Kinematics, angles: np.ndarray, position: np.ndarray) -> np.ndarray | None:
    """Return the allowed version of a 3D target (metres, changed in place), or None to ignore it."""
    tool = model.forward(angles)[:3, 3]
    if must_rise_first(model, angles):
        # Model height says little about table clearance down here, so nothing moves sideways.
        if position[2] <= tool[2]:
            return None
        position[:2] = tool[:2]
    else:
        # Reaching the height limit stops the tool there; it is not an error.
        position[2] = max(position[2], model.min_height + TARGET_HEIGHT_MARGIN)
    if tool[2] < model.min_height:
        position[2] = min(position[2], tool[2] + MAX_RISE_PER_TARGET)
    return position
