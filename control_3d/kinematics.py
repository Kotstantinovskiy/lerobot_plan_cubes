"""Position kinematics for the bundled SO-101 model (metres and radians)."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MODEL_PATH = Path(__file__).parent / "models" / "so101_new_calib.urdf"
JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
)
TOOL_FRAME = "gripper_frame_link"


class Kinematics:
    def __init__(self, path=MODEL_PATH, min_height=0.015):
        root = ET.parse(path).getroot()
        joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
        self.lower = np.array([float(joints[n].find("limit").attrib["lower"]) for n in JOINT_NAMES])
        self.upper = np.array([float(joints[n].find("limit").attrib["upper"]) for n in JOINT_NAMES])
        self.min_height = min_height
        by_child = {joint.find("child").attrib["link"]: joint for joint in joints.values()}
        chain = []
        child = TOOL_FRAME
        while child in by_child:
            joint = by_child[child]
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            name = joint.attrib["name"]
            index = JOINT_NAMES.index(name) if name in JOINT_NAMES else None
            axis_node = joint.find("axis")
            axis = np.fromstring(axis_node.get("xyz", "1 0 0"), sep=" ") if axis_node is not None else np.zeros(3)
            chain.append((transform, axis, index))
            child = joint.find("parent").attrib["link"]
        self.chain = list(reversed(chain))

    def forward(self, angles):
        """Return the tool pose in the base frame."""
        transform = np.eye(4)
        for origin, axis, index in self.chain:
            transform = transform @ origin
            if index is not None:
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis * angles[index]).as_matrix()
                transform = transform @ rotation
        return transform

    def validate(self, angles):
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (5,) or not np.isfinite(angles).all():
            raise ValueError("Expected five finite joint angles")
        if np.any(angles < self.lower - 1e-6) or np.any(angles > self.upper + 1e-6):
            raise ValueError("Pose exceeds model or calibrated joint limits")
        if self.forward(angles)[2, 3] < self.min_height - 1e-6:
            raise ValueError("Gripper is below the minimum height")
        return angles

    def solve(self, position, seed, wrist_roll=None):
        """Solve near the current pose; reject targets more than 3 mm away."""
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Expected a finite 3D position")
        if position[2] < self.min_height or np.linalg.norm(position) > 0.6:
            raise ValueError("Target is outside the workspace")
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (5,) or not np.isfinite(seed).all():
            raise ValueError("Expected five finite seed angles")
        seed = np.clip(seed, self.lower, self.upper)
        roll = seed[4] if wrist_roll is None else float(wrist_roll)
        if not np.isfinite(roll) or not self.lower[4] <= roll <= self.upper[4]:
            raise ValueError("Wrist roll exceeds joint limits")

        def residual(values):
            angles = np.append(values, roll)
            error = self.forward(angles)[:3, 3] - position
            # A weak preference for the previous pose keeps the elbow continuous.
            return np.concatenate((error, 0.001 * (values - seed[:4])))

        result = least_squares(
            residual, seed[:4], bounds=(self.lower[:4], self.upper[:4]),
            max_nfev=45, ftol=1e-5, xtol=1e-5, gtol=1e-7,
        )
        angles = np.append(result.x, roll)
        error = float(np.linalg.norm(self.forward(angles)[:3, 3] - position))
        if error > 0.003:
            raise ValueError(f"Target is unreachable (remaining error {error * 1000:.1f} mm)")
        self.validate(angles)
        return angles

    @staticmethod
    def visual_config(angles, gripper):
        """Jaw percentage is only approximately mapped to the visual joint."""
        config = dict(zip(JOINT_NAMES, map(float, angles)))
        config["gripper"] = float(np.deg2rad(-10 + 1.1 * gripper))
        return config
