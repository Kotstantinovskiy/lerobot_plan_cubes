"""Controller transitions and limits without opening any serial ports."""

import numpy as np
import pytest

from controller import Controller
from fakes import FakeHardware, connected_controller
from kinematics import Kinematics


def test_preview_speed_is_bounded_even_after_long_delay():
    controller = Controller(Kinematics(), speed=20)
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.set_gripper(100)
    controller.tick(10)
    assert np.rad2deg(controller.angles[0]) == pytest.approx(2)
    assert controller.gripper == pytest.approx(52.5)
    assert not controller.connected


def test_configured_but_disconnected_hardware_still_allows_preview():
    hardware = FakeHardware()
    controller = Controller(Kinematics(), hardware)
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    assert np.rad2deg(controller.angles[0]) == pytest.approx(2)
    assert not hardware.sent
    assert not hardware.connected


def test_enable_discards_preview_target_and_latches_measured_pose():
    controller, hardware = connected_controller()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.set_gripper(100)
    hardware.positions["shoulder_pan"] = 7
    hardware.positions["gripper"] = 15
    controller.enable()
    controller.tick(0.05)
    assert hardware.sent[-1]["shoulder_pan"] == pytest.approx(7)
    assert hardware.sent[-1]["gripper"] == 15
    assert controller.enabled


def test_motion_permission_distinguishes_preview_from_an_unexpected_lost_connection():
    controller = Controller(Kinematics())
    assert controller.motion_allowed
    controller, hardware = connected_controller()
    assert not controller.motion_allowed
    controller.enable()
    assert controller.motion_allowed
    hardware.connected = False
    assert not controller.motion_allowed


def test_tracking_error_sends_current_hold_and_requires_reenable():
    controller, hardware = connected_controller()
    controller.enable()
    hardware.positions["shoulder_pan"] = -15
    controller.tick(0.05)
    assert not controller.enabled
    assert hardware.armed
    assert hardware.sent[-1]["shoulder_pan"] == -15
    assert "not following" in controller.status
    assert np.rad2deg(controller.measured_angles[0]) == pytest.approx(-15)
    controller.tick(0.05)
    assert len(hardware.sent) == 1


@pytest.mark.parametrize("joint", ["shoulder_pan", "gripper"])
def test_stalled_joint_pauses_even_when_driver_clips_targets(joint):
    controller, hardware = connected_controller()
    controller.enable()
    hardware.follow = False
    hardware.relative_limit = 3.0
    initial = hardware.positions[joint]
    if joint == "gripper":
        controller.set_gripper(100)
    else:
        controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    for _ in range(60):
        controller.tick(1 / 30)
        if not controller.enabled:
            break
    assert not controller.enabled
    assert "not following" in controller.status
    assert hardware.armed
    assert all(abs(command[joint] - initial) <= 3.0 for command in hardware.sent)
    if joint == "gripper":
        # Pausing keeps the existing bounded gripping command, rather than
        # replacing it with the blocked jaw's measured opening.
        assert hardware.sent[-1][joint] == hardware.sent[-2][joint]
        assert controller.target_gripper == controller.gripper
    else:
        assert hardware.sent[-1][joint] == initial


@pytest.mark.parametrize("kind", ["joints", "target", "gripper"])
def test_invalid_target_stops_movement(kind):
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    with pytest.raises(ValueError):
        if kind == "joints":
            controller.set_joints(np.deg2rad([120, 0, 0, 0, 0]))
        elif kind == "target":
            controller.set_target([3, 0, 0.1])
        else:
            controller.set_gripper(float("nan"))
    assert not controller.enabled
    assert hardware.sent[-1]["shoulder_pan"] == 0
    assert np.array_equal(controller.target_angles, controller.angles)


def test_disconnect_restores_model_limits_and_releases_torque():
    controller, hardware = connected_controller()
    assert controller.model.upper[0] == pytest.approx(np.deg2rad(90))
    controller.enable()
    controller.disconnect()
    assert not hardware.armed
    assert not controller.connected
    assert controller.model.upper[0] == pytest.approx(np.deg2rad(110), abs=1e-5)


def test_unsafe_measured_pose_can_display_but_cannot_enable():
    controller, hardware = connected_controller()
    hardware.positions["wrist_flex"] = 120
    controller.tick(0.05)
    assert np.rad2deg(controller.angles[3]) == pytest.approx(120)
    with pytest.raises(ValueError):
        controller.enable()
    assert not controller.enabled
    assert not hardware.armed


def test_preview_invalid_target_freezes_existing_motion():
    controller = Controller(Kinematics())
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    with pytest.raises(ValueError):
        controller.set_target([3, 0, 0.1])
    previous = controller.angles.copy()
    controller.tick(0.1)
    assert np.array_equal(controller.angles, previous)


@pytest.mark.parametrize("value", [-1, 0, 46, float("nan"), float("inf"), None])
def test_speed_changes_are_validated_without_replacing_the_last_valid_speed(value):
    controller = Controller(Kinematics(), speed=20)
    with pytest.raises(ValueError, match="speed"):
        controller.speed = value
    assert controller.speed == 20
