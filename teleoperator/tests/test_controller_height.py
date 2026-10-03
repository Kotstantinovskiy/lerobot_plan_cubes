"""The height limit and the folded rest pose past the model limits."""

import numpy as np
import pytest

from controller import Controller
from fakes import REST_POSE, FakeHardware, connected_controller
from kinematics import JOINT_NAMES, Kinematics

# Within the joint limits, with the tool about 2 mm above the base plane: below the height limit.
LOW_POSE = {"shoulder_pan": -2.5, "shoulder_lift": -85.0, "elbow_flex": 85.0, "wrist_flex": 79.7, "wrist_roll": -56.9}


# Tool about 12 mm above the base plane, as after the arm sags while holding at the height limit.
SAGGED_POSE = {
    "shoulder_pan": -2.5,
    "shoulder_lift": -77.3,
    "elbow_flex": 78.8,
    "wrist_flex": 79.7,
    "wrist_roll": -56.9,
}


def tool_height(controller):
    return controller.model.forward(controller.angles)[2, 3]


def enabled_at_rest():
    hardware = FakeHardware()
    hardware.joint_limits = dict.fromkeys(JOINT_NAMES, (-106.0, 106.0))
    hardware.positions.update(REST_POSE)
    controller = Controller(Kinematics(), hardware)
    controller.connect()
    controller.enable()
    return controller, hardware


def test_target_below_the_height_limit_stops_there_and_keeps_motion_enabled():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_target([0.25, 0.0, -0.1])
    for _ in range(300):
        controller.tick(0.1)
    assert controller.enabled
    assert tool_height(controller) == pytest.approx(controller.model.min_height, abs=0.003)
    assert tool_height(controller) >= controller.model.min_height


def test_pose_that_sagged_below_the_height_limit_can_enable_and_only_rises():
    controller, hardware = connected_controller()
    hardware.positions.update(SAGGED_POSE)
    controller.enable()
    low_height = tool_height(controller)
    assert controller.enabled
    assert low_height < controller.model.min_height
    assert not controller.must_rise_first
    controller.tick(0.05)
    assert controller.enabled
    position = controller.model.forward(controller.angles)[:3, 3]
    previous_height = low_height
    for _ in range(100):
        controller.set_target([position[0], position[1], low_height - 0.01])
        controller.tick(1 / 30)
        assert tool_height(controller) >= previous_height - 1e-6
        assert tool_height(controller) <= previous_height + 0.001
        previous_height = tool_height(controller)
    assert controller.enabled
    assert tool_height(controller) >= controller.model.min_height


def test_joint_command_below_the_height_limit_is_ignored_without_pausing():
    controller, hardware = connected_controller()
    controller.enable()
    previous_target = controller.target_angles.copy()
    controller.set_joints(np.deg2rad([LOW_POSE[name] for name in JOINT_NAMES[:5]]))
    assert controller.enabled
    assert np.array_equal(controller.target_angles, previous_target)


def test_motion_step_below_the_height_limit_holds_without_pausing():
    controller, hardware = connected_controller()
    controller.enable()
    position = controller.model.forward(controller.angles)[:3, 3]
    lower_target = controller.model.solve([position[0], position[1], position[2] - 0.05], controller.angles)
    controller.model.min_height = position[2] - 0.0001
    controller.target_angles = lower_target
    held_angles = controller.angles.copy()
    controller.tick(0.1)
    assert controller.enabled
    assert np.array_equal(controller.angles, held_angles)


def test_rest_pose_past_the_model_limits_can_enable_and_hold():
    controller, hardware = enabled_at_rest()
    assert controller.enabled
    assert controller.must_rise_first
    held_angles = controller.angles.copy()
    controller.tick(0.05)
    assert controller.enabled
    assert np.array_equal(controller.angles, held_angles)


def test_rest_pose_lifts_straight_up():
    controller, hardware = enabled_at_rest()
    start = controller.model.forward(controller.angles)[:3, 3].copy()
    while controller.must_rise_first:
        tool = controller.model.forward(controller.angles)[:3, 3]
        controller.set_target([tool[0], tool[1], tool[2] + 0.0005])
        controller.tick(1 / 30)
        assert controller.enabled
    tool = controller.model.forward(controller.angles)[:3, 3]
    assert tool[:2] == pytest.approx(start[:2], abs=0.001)
    assert tool[2] > start[2] + 0.03


def test_rest_pose_ignores_sideways_and_downward_targets():
    controller, hardware = enabled_at_rest()
    tool = controller.model.forward(controller.angles)[:3, 3].copy()
    previous_target = controller.target_angles.copy()
    controller.set_target([tool[0] + 0.02, tool[1], tool[2]])
    controller.set_target([tool[0], tool[1] + 0.02, tool[2]])
    controller.set_target([tool[0], tool[1], tool[2] - 0.01])
    assert controller.enabled
    assert np.array_equal(controller.target_angles, previous_target)


def test_large_target_above_a_low_tool_rises_one_small_step():
    controller, hardware = enabled_at_rest()
    tool = controller.model.forward(controller.angles)[:3, 3].copy()
    controller.set_target([tool[0] + 0.05, tool[1], tool[2] + 0.1])
    target = controller.model.forward(controller.target_angles)[:3, 3]
    assert target[0] == pytest.approx(tool[0], abs=0.0005)
    assert 0 < target[2] - tool[2] < 0.0015


def test_joint_past_the_model_limit_may_move_back_but_not_further():
    controller, hardware = enabled_at_rest()
    back = controller.angles.copy()
    back[2] -= np.deg2rad(1)
    controller.set_joints(back)
    assert controller.enabled
    assert np.array_equal(controller.target_angles, back)
    further = controller.angles.copy()
    further[1] -= np.deg2rad(1)
    with pytest.raises(ValueError, match="limits"):
        controller.set_joints(further)
    assert not controller.enabled
