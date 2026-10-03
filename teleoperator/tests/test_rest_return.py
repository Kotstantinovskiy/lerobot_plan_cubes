"""Planned way back to the rest pose, checked against the arm model."""

import numpy as np
import pytest

from kinematics import JOINT_NAMES, Kinematics
from rest_return import FAR_REACH, SAFE_HEIGHT, plan_return

REST = {
    "shoulder_pan": -2.1,
    "shoulder_lift": -104.8,
    "elbow_flex": 103.6,
    "wrist_flex": 79.4,
    "wrist_roll": -57.2,
    "gripper": 0.4,
}
# Stretched out over the table, as the arm was left after a phone session (degrees).
FAR_POSE = (10.5, 7.7, 34.6, -5.0, -2.2)


@pytest.fixture
def model():
    return Kinematics()


def tool_positions(model, start_degrees, steps):
    """Return tool positions along the straight joint lines through the planned steps."""
    pose = np.array(start_degrees, dtype=float)
    positions = []
    for step in steps:
        target = pose.copy()
        for index, name in enumerate(JOINT_NAMES):
            if name in step:
                target[index] = step[name]
        for fraction in np.linspace(0.0, 1.0, 40):
            angles = np.deg2rad(pose + fraction * (target - pose))
            positions.append(model.forward(angles)[:3, 3].copy())
        pose = target
    return positions


def test_far_pose_first_rises_straight_up(model):
    steps = plan_return(model, np.deg2rad(FAR_POSE), REST)
    start = model.forward(np.deg2rad(FAR_POSE))[:3, 3]
    positions = tool_positions(model, FAR_POSE, steps[:1])
    assert positions[-1][:2] == pytest.approx(start[:2], abs=0.001)
    assert positions[-1][2] == pytest.approx(SAFE_HEIGHT, abs=0.001)


def test_far_pose_stays_above_the_cubes_until_it_is_close(model):
    steps = plan_return(model, np.deg2rad(FAR_POSE), REST)
    start_height = model.forward(np.deg2rad(FAR_POSE))[2, 3]
    far_heights = [tool[2] for tool in tool_positions(model, FAR_POSE, steps) if np.hypot(tool[0], tool[1]) > FAR_REACH]
    assert min(far_heights) >= min(start_height, 0.03)


def test_base_keeps_its_angle_until_the_arm_pulls_back(model):
    steps = plan_return(model, np.deg2rad(FAR_POSE), REST)
    assert steps[0]["shoulder_pan"] == pytest.approx(FAR_POSE[0], abs=0.01)
    assert "shoulder_pan" not in steps[1]


def test_path_ends_at_the_rest_pose(model):
    assert plan_return(model, np.deg2rad(FAR_POSE), REST)[-1] == REST


def test_moderately_far_pose_skips_the_high_pose(model):
    moderate = model.solve([0.18, 0.0, 0.06], np.deg2rad([0.0, -60.0, 60.0, 60.0, -57.0]))
    steps = plan_return(model, moderate, REST)
    assert len(steps) == 3
    assert steps[0]["shoulder_pan"] == pytest.approx(np.rad2deg(moderate[0]), abs=0.01)
    assert steps[1]["shoulder_lift"] == -70.0
    assert steps[2] == REST


def test_pose_just_above_rest_folds_directly(model):
    near_rest = np.deg2rad([-2.0, -95.0, 95.0, 79.4, -57.2])
    assert plan_return(model, near_rest, REST) == [REST]


def test_close_pose_with_a_turned_base_turns_it_at_the_tucked_pose(model):
    turned = np.deg2rad([25.0, -95.0, 95.0, 79.4, -57.2])
    steps = plan_return(model, turned, REST)
    assert len(steps) == 2
    assert steps[0]["shoulder_pan"] == REST["shoulder_pan"]
    assert steps[0]["shoulder_lift"] == -70.0


def test_unreachable_rise_is_skipped(model):
    def refuse_target(*args, **kwargs):
        raise ValueError("Target is unreachable")

    model.solve = refuse_target
    steps = plan_return(model, np.deg2rad(FAR_POSE), REST)
    assert "shoulder_pan" not in steps[0]
    assert steps[-1] == REST
