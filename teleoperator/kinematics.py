"""Position kinematics for the bundled SO-101 model (metres and radians)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MODEL_PATH = Path(__file__).parent / "models" / "so101_new_calib.urdf"
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TOOL_FRAME = "gripper_frame_link"
WRIST_FLEX_INDEX = 3
WRIST_ROLL_INDEX = 4
MAX_GRIPPER_OPENING = 100.0  # percent
DEFAULT_MIN_HEIGHT = 0.015  # metres
MAX_REACH = 0.6  # metres
MAX_SOLVE_ERROR = 0.003  # metres
ANGLE_TOLERANCE = 1e-6  # radians
SEED_PREFERENCE_WEIGHT = 0.001
MAX_SOLVER_EVALUATIONS = 45
GRIPPER_ZERO_DEGREES = -10.0
GRIPPER_DEGREES_PER_PERCENT = 1.1


class BelowMinimumHeightError(ValueError):
    """The pose is within the joint limits, but the tool would be too low."""


def _parse_vector(text):
    return np.fromstring(text, sep=" ")


def _joint_origin(joint):
    """Return the 4x4 transform of a URDF joint relative to its parent link."""
    transform = np.eye(4)
    origin = joint.find("origin")
    if origin is None:
        return transform
    transform[:3, 3] = _parse_vector(origin.get("xyz", "0 0 0"))
    transform[:3, :3] = Rotation.from_euler("xyz", _parse_vector(origin.get("rpy", "0 0 0"))).as_matrix()
    return transform


def _joint_axis(joint):
    axis = joint.find("axis")
    return np.zeros(3) if axis is None else _parse_vector(axis.get("xyz", "1 0 0"))


def _load_chain(joints):
    """Return (origin, axis, index in JOINT_NAMES or None) for each joint from the base to the tool."""
    by_child = {joint.find("child").attrib["link"]: joint for joint in joints.values()}
    chain = []
    child = TOOL_FRAME
    while child in by_child:
        joint = by_child[child]
        name = joint.attrib["name"]
        index = JOINT_NAMES.index(name) if name in JOINT_NAMES else None
        chain.append((_joint_origin(joint), _joint_axis(joint), index))
        child = joint.find("parent").attrib["link"]
    chain.reverse()
    return chain


class Kinematics:
    def __init__(self, path: Path = MODEL_PATH, min_height: float = DEFAULT_MIN_HEIGHT) -> None:
        root = ET.parse(path).getroot()
        joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
        self.lower = np.array([float(joints[name].find("limit").attrib["lower"]) for name in JOINT_NAMES])
        self.upper = np.array([float(joints[name].find("limit").attrib["upper"]) for name in JOINT_NAMES])
        self.min_height = min_height
        self.chain = _load_chain(joints)

    def forward(self, angles: ArrayLike) -> np.ndarray:
        """Return the tool pose in the base frame."""
        transform = np.eye(4)
        for origin, axis, index in self.chain:
            transform = transform @ origin
            if index is not None:
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis * angles[index]).as_matrix()
                transform = transform @ rotation
        return transform

    def _allowed_range(self, current_angles):
        """Return (lower, upper, lowest height), widened to include a pose already past a limit."""
        if current_angles is None:
            return self.lower, self.upper, self.min_height
        return (
            np.minimum(self.lower, current_angles),
            np.maximum(self.upper, current_angles),
            min(self.min_height, self.forward(current_angles)[2, 3]),
        )

    def validate(self, angles: ArrayLike, *, current_angles: np.ndarray | None = None) -> np.ndarray:
        """Check joint limits and tool height.

        With current_angles, a pose already past a limit, as at the folded rest pose or after
        sagging under the arm's weight, may stay there or move back but not go further out.
        """
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (5,) or not np.isfinite(angles).all():
            raise ValueError("Expected five finite joint angles")
        lower, upper, lowest_height = self._allowed_range(current_angles)
        if np.any(angles < lower - ANGLE_TOLERANCE) or np.any(angles > upper + ANGLE_TOLERANCE):
            raise ValueError("Pose exceeds model or calibrated joint limits")
        if self.forward(angles)[2, 3] < lowest_height - ANGLE_TOLERANCE:
            raise BelowMinimumHeightError("Gripper is below the minimum height")
        return angles

    def solve(
        self,
        position: ArrayLike,
        seed: ArrayLike,
        wrist_roll: float | None = None,
        *,
        current_angles: np.ndarray | None = None,
        keep_wrist_flex: bool = False,
    ) -> np.ndarray:
        """Solve near the current pose; reject targets more than 3 mm away.

        With current_angles, the solution may stay as far past a limit as that pose is.
        With keep_wrist_flex, the wrist flex stays at its seed angle.
        """
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Expected a finite 3D position")
        lower, upper, lowest_height = self._allowed_range(current_angles)
        if position[2] < lowest_height or np.linalg.norm(position) > MAX_REACH:
            raise ValueError("Target is outside the workspace")
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (5,) or not np.isfinite(seed).all():
            raise ValueError("Expected five finite seed angles")
        seed = np.clip(seed, lower, upper)
        roll = seed[WRIST_ROLL_INDEX] if wrist_roll is None else float(wrist_roll)
        if not np.isfinite(roll) or not lower[WRIST_ROLL_INDEX] <= roll <= upper[WRIST_ROLL_INDEX]:
            raise ValueError("Wrist roll exceeds joint limits")

        if keep_wrist_flex:
            lower, upper = lower.copy(), upper.copy()
            lower[WRIST_FLEX_INDEX] = seed[WRIST_FLEX_INDEX] - ANGLE_TOLERANCE
            upper[WRIST_FLEX_INDEX] = seed[WRIST_FLEX_INDEX] + ANGLE_TOLERANCE

        angles = self._closest_angles(position, seed, roll, lower, upper)
        error = float(np.linalg.norm(self.forward(angles)[:3, 3] - position))
        if error > MAX_SOLVE_ERROR:
            raise ValueError(f"Target is unreachable (remaining error {error * 1000:.1f} mm)")
        self.validate(angles, current_angles=current_angles)
        return angles

    def _closest_angles(self, position, seed, roll, lower, upper):
        """Return the five angles whose tool comes closest to position, with the wrist roll fixed."""

        def residual(values):
            angles = np.append(values, roll)
            error = self.forward(angles)[:3, 3] - position
            # A weak preference for the previous pose keeps the elbow continuous.
            return np.concatenate((error, SEED_PREFERENCE_WEIGHT * (values - seed[:WRIST_ROLL_INDEX])))

        result = least_squares(
            residual,
            seed[:WRIST_ROLL_INDEX],
            bounds=(lower[:WRIST_ROLL_INDEX], upper[:WRIST_ROLL_INDEX]),
            max_nfev=MAX_SOLVER_EVALUATIONS,
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-7,
        )
        return np.append(result.x, roll)

    @staticmethod
    def visual_config(angles: np.ndarray, gripper: float) -> dict[str, float]:
        """Jaw percentage is only approximately mapped to the visual joint."""
        config = dict(zip(JOINT_NAMES, map(float, angles), strict=False))
        config["gripper"] = float(np.deg2rad(GRIPPER_ZERO_DEGREES + GRIPPER_DEGREES_PER_PERCENT * gripper))
        return config
