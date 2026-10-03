"""Path back to the rest pose that does not sweep over the table."""

import numpy as np

try:
    from .kinematics import JOINT_NAMES
except ImportError:
    from kinematics import JOINT_NAMES

# Beyond this reach the tool first rises and pulls back high above any cubes (metres).
FAR_REACH = 0.15
# From far out the tool rises straight up to at least this height (metres).
SAFE_HEIGHT = 0.10
# High on the way back: tool about 230 mm out and 115 mm up (degrees).
HIGH_POSE = {"shoulder_lift": -50.0, "elbow_flex": 40.0, "wrist_flex": 60.0}
# Only a tool further out than the high pose goes through it (metres).
HIGH_POSE_REACH = 0.23
# Folded above the rest pose, where the base turns: tool about 130 mm out and 30 mm up (degrees).
TUCKED_POSE = {"shoulder_lift": -70.0, "elbow_flex": 70.0}
# A base this close to its rest angle needs no turn before folding (degrees).
BASE_TOLERANCE = 3.0


def _rise_step(model, angles, tool):
    """Return joint targets straight above the tool, or None when none are needed or reachable."""
    if tool[2] >= SAFE_HEIGHT:
        return None
    try:
        risen = model.solve([tool[0], tool[1], SAFE_HEIGHT], angles, current_angles=angles)
    except ValueError:
        return None
    return dict(zip(JOINT_NAMES, map(float, np.rad2deg(risen))))


def plan_return(model, angles, rest):
    """Return ordered joint targets (degrees, gripper 0..100) from angles (radians) to rest.

    Each target names only the joints it moves; the arm walks to each one in a straight joint line.
    """
    tool = model.forward(angles)[:3, 3]
    reach = np.hypot(tool[0], tool[1])
    is_far = reach > FAR_REACH
    steps = []
    if is_far:
        rise = _rise_step(model, angles, tool)
        if rise is not None:
            steps.append(rise)
    if reach > HIGH_POSE_REACH:
        steps.append({**HIGH_POSE, "wrist_roll": rest["wrist_roll"]})
    tucked = {
        **TUCKED_POSE,
        "shoulder_pan": rest["shoulder_pan"],
        "wrist_flex": rest["wrist_flex"],
        "wrist_roll": rest["wrist_roll"],
        "gripper": rest["gripper"],
    }
    tucked_angles = np.deg2rad([tucked[name] for name in JOINT_NAMES])
    is_beside_rest = abs(np.rad2deg(angles[0]) - rest["shoulder_pan"]) > BASE_TOLERANCE
    is_above_tucked = tool[2] > model.forward(tucked_angles)[2, 3]
    if is_far or is_beside_rest or is_above_tucked:
        steps.append(tucked)
    steps.append(dict(rest))
    return steps
