"""Controller transitions and limits without opening any serial ports."""

import numpy as np
import pytest

from controller import Controller
from kinematics import JOINT_NAMES, Kinematics


class FakeHardware:
    def __init__(self):
        self.connected = False
        self.armed = False
        self.positions = dict.fromkeys(JOINT_NAMES, 0.0)
        self.positions["gripper"] = 50.0
        self.joint_limits = dict.fromkeys(JOINT_NAMES, (-90.0, 90.0))
        self.sent = []
        self.follow = True
        self.relative_limit = None

    def connect(self):
        self.connected = True
        return self.read_positions()

    def read_positions(self):
        return self.positions.copy()

    def arm(self):
        self.armed = True
        return self.read_positions()

    def send_positions(self, positions):
        assert self.armed
        if self.relative_limit is not None:
            positions = {
                name: self.positions[name] + float(np.clip(
                    value - self.positions[name], -self.relative_limit, self.relative_limit
                ))
                for name, value in positions.items()
            }
        self.sent.append(positions.copy())
        if self.follow:
            self.positions = positions.copy()
        return positions.copy()

    def disarm(self):
        self.armed = False

    def close(self):
        self.disarm()
        self.connected = False


def connected_controller():
    hardware = FakeHardware()
    controller = Controller(Kinematics(), hardware)
    controller.connect()
    return controller, hardware


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


def test_failed_serial_read_stops_further_updates():
    controller, hardware = connected_controller()
    controller.enable()

    def fail_read():
        raise OSError("serial link lost")

    hardware.read_positions = fail_read
    with pytest.raises(RuntimeError, match="serial link lost"):
        controller.tick(0.05)
    assert not controller.enabled
    assert not hardware.sent


def test_preview_invalid_target_freezes_existing_motion():
    controller = Controller(Kinematics())
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    with pytest.raises(ValueError):
        controller.set_target([0, 0, -1])
    previous = controller.angles.copy()
    controller.tick(0.1)
    assert np.array_equal(controller.angles, previous)


def test_lost_connection_cannot_turn_enabled_motion_into_preview_or_resume_it():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    hardware.connected = False
    with pytest.raises(RuntimeError, match="connection lost"):
        controller.tick(0.1)
    assert not controller.enabled
    assert controller.faulted
    frozen = controller.angles.copy()
    sent_count = len(hardware.sent)
    # A recovered serial link must not resume an old mouse target.
    hardware.connected = True
    controller.tick(0.1)
    assert np.array_equal(controller.angles, frozen)
    assert len(hardware.sent) == sent_count
    recovered_position = hardware.positions["shoulder_pan"]
    controller.enable()
    assert not controller.faulted
    controller.tick(0.1)
    assert hardware.sent[-1]["shoulder_pan"] == hardware.positions["shoulder_pan"]
    assert np.rad2deg(controller.target_angles[0]) == pytest.approx(recovered_position)


@pytest.mark.parametrize("enabled", [False, True])
def test_serial_failure_is_latched_instead_of_retried_on_every_tick(enabled):
    controller, hardware = connected_controller()
    if enabled:
        controller.enable()
    reads = []

    def fail_read():
        reads.append(True)
        raise OSError("serial link lost")

    hardware.read_positions = fail_read
    with pytest.raises(RuntimeError, match="serial link lost"):
        controller.tick(0.1)
    first_attempts = len(reads)
    for _ in range(30):
        controller.tick(1 / 30)
    assert len(reads) == first_attempts
    assert controller.faulted
    assert not controller.enabled


def test_failed_hold_freezes_last_measured_pose_and_requires_explicit_reenable():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    # The command is ahead of the observation sampled at the beginning of tick.
    assert controller.angles[0] > controller.measured_angles[0]

    def fail_read():
        raise OSError("serial link lost")

    hardware.read_positions = fail_read
    with pytest.raises(RuntimeError, match="hold failed"):
        controller.pause()
    assert controller.faulted
    assert not controller.enabled
    assert np.array_equal(controller.angles, controller.measured_angles)
    assert np.array_equal(controller.target_angles, controller.measured_angles)


def test_reenable_with_invalid_measured_pose_holds_instead_of_leaving_old_goal():
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    hardware.positions["wrist_flex"] = 100
    with pytest.raises(ValueError, match="limits"):
        controller.enable()
    assert not controller.enabled
    assert hardware.armed
    assert hardware.sent[-1] == hardware.positions


def test_lost_connection_cannot_report_successful_torque_release():
    controller, hardware = connected_controller()
    controller.enable()
    hardware.connected = False
    with pytest.raises(RuntimeError, match="torque release not confirmed"):
        controller.release()
    assert controller.faulted
    assert not controller.enabled


@pytest.mark.parametrize("value", [-1, 0, 46, float("nan"), float("inf"), None])
def test_speed_changes_are_validated_without_replacing_the_last_valid_speed(value):
    controller = Controller(Kinematics(), speed=20)
    with pytest.raises(ValueError, match="speed"):
        controller.speed = value
    assert controller.speed == 20


@pytest.mark.parametrize("port_stays_open", [False, True])
def test_failed_disconnect_latches_fault_and_does_not_claim_release(port_stays_open):
    controller, hardware = connected_controller()
    controller.enable()
    controller.set_joints(np.deg2rad([30, 0, 0, 0, 0]))
    controller.tick(0.1)
    close = hardware.close

    def fail_close():
        hardware.connected = port_stays_open
        raise OSError("torque disable failed")

    hardware.close = fail_close
    with pytest.raises(RuntimeError, match="torque release not confirmed"):
        controller.disconnect()
    assert controller.faulted
    assert not controller.enabled
    assert controller.connected == port_stays_open
    assert "Disconnect failed" in controller.status
    assert np.array_equal(controller.angles, controller.measured_angles)
    assert np.array_equal(controller.target_angles, controller.measured_angles)
    sent_count = len(hardware.sent)
    controller.tick(0.1)
    assert len(hardware.sent) == sent_count
    # Motor limits still apply while the failed disconnect leaves a live port.
    expected_limit = 90 if port_stays_open else 110
    assert np.rad2deg(controller.model.upper[0]) == pytest.approx(expected_limit, abs=0.001)
    hardware.close = close
    controller.disconnect()
    assert not controller.faulted
    assert not controller.connected
