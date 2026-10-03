"""Pausing and holding: the arm keeps its pose and a grasp stays closed."""

import numpy as np
import pytest

from fakes import connected_controller


def test_pause_holds_measured_pose_and_preserves_torque():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    hardware.positions["shoulder_pan"] = 4
    controller.pause()
    assert not controller.enabled
    assert hardware.armed
    assert hardware.sent[-1]["shoulder_pan"] == 4
    assert controller.target_angles[0] == pytest.approx(np.deg2rad(4))


def test_pause_retains_a_loaded_grasp_and_stops_further_closing():
    controller, hardware = connected_controller()
    controller.enable()
    hardware.follow = False
    controller.gripper = 11
    controller.target_gripper = 8
    hardware.positions["gripper"] = 16
    controller.pause()
    assert not controller.enabled
    assert hardware.armed
    assert controller.gripper == 11
    assert controller.target_gripper == 11
    assert controller.measured_gripper == 16
    assert hardware.sent[-1]["gripper"] == 11


def test_paused_polling_and_reenable_do_not_replace_the_owned_grasp_with_its_opening():
    controller, hardware = connected_controller()
    controller.enable()
    hardware.follow = False
    controller.gripper = 11
    controller.target_gripper = 8
    hardware.positions["gripper"] = 16
    controller.pause()
    controller.tick(0.1)
    assert controller.gripper == 11
    assert controller.target_gripper == 11
    # A paused target must still be discarded when explicitly re-enabling.
    controller.set_gripper(75)
    controller.enable()
    controller.tick(0.1)
    assert controller.enabled
    assert controller.gripper == 11
    assert controller.target_gripper == 11
    assert hardware.sent[-1]["gripper"] == 11


def test_pause_freezes_the_current_gripper_command_during_a_longer_close():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_gripper(100)
    controller.tick(0.04)
    command = controller.gripper
    controller.pause()
    assert controller.gripper == command
    assert controller.target_gripper == command
    assert hardware.sent[-1]["gripper"] == command
    controller.tick(0.1)
    assert controller.gripper == command


def test_pause_without_owned_torque_uses_the_measured_jaw_and_sends_nothing():
    controller, hardware = connected_controller()
    hardware.positions["gripper"] = 16
    controller.gripper = 11
    controller.target_gripper = 8
    controller.pause()
    assert controller.gripper == 16
    assert controller.target_gripper == 16
    assert not hardware.sent


def test_keyboard_hold_preserves_enabled_state_after_a_successful_measured_hold():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    hardware.positions["shoulder_pan"] = 4
    controller.hold_motion()
    assert controller.enabled
    assert controller.motion_allowed
    assert hardware.armed
    assert hardware.sent[-1]["shoulder_pan"] == 4
    assert controller.target_angles == pytest.approx(controller.measured_angles)


def test_keyboard_hold_never_enables_a_paused_controller():
    controller, hardware = connected_controller()
    controller.hold_motion()
    assert not controller.enabled
    assert not controller.motion_allowed
    assert not hardware.armed
    assert not hardware.sent


def test_failed_keyboard_hold_cannot_restore_enabled_state():
    controller, hardware = connected_controller()
    controller.enable()

    def fail_read():
        raise OSError("serial link lost")

    hardware.read_positions = fail_read
    with pytest.raises(RuntimeError, match="serial link lost"):
        controller.hold_motion()
    assert not controller.enabled
    assert not controller.motion_allowed
    assert controller.faulted


def test_keyboard_hold_does_not_relax_a_gripper_holding_an_object():
    controller, hardware = connected_controller()
    controller.enable()
    controller.gripper = 11
    controller.target_gripper = 8
    hardware.positions["gripper"] = 16
    controller.hold_motion()
    assert controller.enabled
    assert controller.gripper == 11
    assert controller.target_gripper == 8
    assert controller.measured_gripper == 16
    assert hardware.sent[-1]["gripper"] == 11


def test_keyboard_hold_preserves_an_in_progress_gripper_target_without_jumping_to_it():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_gripper(100)
    controller.tick(0.04)
    command = controller.gripper
    controller.hold_motion()
    assert controller.gripper == command
    assert controller.target_gripper == 100
    assert hardware.sent[-1]["gripper"] == command
    controller.tick(0.1)
    assert controller.gripper == pytest.approx(command + 2.5)
