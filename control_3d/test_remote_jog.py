"""Five-axis thumb control and lost-touch stopping without physical devices."""

import json

import numpy as np
import pytest

from controller import Controller
from kinematics import JOINT_NAMES, Kinematics
from remote_jog import RemoteJog
from test_controller import REST_POSE, FakeHardware
from test_keyboard_jog import Clock, point


def remote(hardware=None):
    clock = Clock()
    controller = Controller(Kinematics(), hardware)
    return RemoteJog(controller, clock), controller, clock


def packet(jog, rates=(0, 1, 0, 0, 0), sequence=0):
    return [*rates, sequence, jog.epoch]


def advance(jog, controller, clock, rates, steps=30):
    for sequence in range(steps):
        assert jog.update(packet(jog, rates, sequence))
        jog.tick(1 / 30)
        controller.tick(1 / 30)
        clock.advance(1 / 30)


def test_two_thumbs_can_move_xyz_roll_and_gripper_in_the_same_tick():
    jog, controller, clock = remote()
    initial = point(controller)
    advance(jog, controller, clock, (0.3, 0.4, 0.5, 0.5, -0.5))
    assert point(controller) - initial == pytest.approx((0.006, 0.008, 0.01), abs=0.0001)
    assert np.rad2deg(controller.angles[4]) == pytest.approx(10.0)
    assert controller.gripper == pytest.approx(37.5)


def test_flat_table_movement_preserves_height_and_has_analog_speed():
    jog, controller, clock = remote()
    initial = point(controller)
    advance(jog, controller, clock, (0, 0.25, 0, 0, 0))
    assert point(controller) - initial == pytest.approx((0, 0.005, 0), abs=0.0001)


def test_release_discards_unfinished_motion_and_retains_the_commanded_grasp():
    jog, controller, clock = remote()
    advance(jog, controller, clock, (0, 0, 0, 1, -1), steps=3)
    assert controller.gripper < 50
    commanded_grip = controller.gripper
    jog.tick(0.05)
    assert controller.target_gripper < commanded_grip
    assert jog.update(packet(jog, (0, 0, 0, 0, 0), sequence=4))
    frozen = controller.angles.copy()
    assert controller.target_gripper == commanded_grip
    for _ in range(10):
        jog.tick(1 / 30)
        controller.tick(1 / 30)
    assert controller.angles == pytest.approx(frozen)
    assert controller.gripper == commanded_grip
    assert not jog.active


def test_releasing_roll_and_grip_does_not_cancel_a_thumb_still_moving_xy():
    jog, controller, clock = remote()
    advance(jog, controller, clock, (0, 0.4, 0, 1, -1), steps=3)
    roll, opening, y = controller.angles[4], controller.gripper, point(controller)[1]
    assert jog.update(packet(jog, (0, 0.4, 0, 0, 0), sequence=4))
    jog.tick(0.05)
    controller.tick(0.05)
    assert jog.active
    assert controller.angles[4] == roll
    assert controller.gripper == opening
    assert point(controller)[1] > y


@pytest.mark.parametrize("sign", [-1, 1])
def test_wrist_and_jaw_limits_do_not_stop_valid_translation(sign):
    jog, controller, clock = remote()
    controller.angles[4] = controller.model.upper[4] if sign == 1 else controller.model.lower[4]
    controller.gripper = 100.0 if sign == 1 else 0.0
    initial = point(controller)
    advance(jog, controller, clock, (0, 0.4, 0, sign, sign), steps=3)
    assert jog.active
    assert controller.gripper == (100.0 if sign == 1 else 0.0)
    assert controller.angles[4] == (controller.model.upper[4] if sign == 1 else controller.model.lower[4])
    assert point(controller)[1] > initial[1]


def test_wrist_rate_respects_the_configured_joint_speed():
    jog, controller, clock = remote()
    controller.speed = 3
    advance(jog, controller, clock, (0, 0, 0, 1, 0))
    assert np.rad2deg(controller.angles[4]) == pytest.approx(3)


def test_delayed_control_tick_is_capped_for_every_axis():
    jog, controller, clock = remote()
    initial = point(controller)
    jog.update(packet(jog, (0, 1, 0, 1, -1)))
    jog.tick(10)
    target_position = controller.model.forward(controller.target_angles)[:3, 3]
    assert np.linalg.norm(target_position - initial) <= 0.00101
    assert np.rad2deg(controller.target_angles[4]) == pytest.approx(1)
    assert controller.target_gripper == pytest.approx(48.75)


def test_input_timeout_stops_all_axes_and_invalidates_delayed_packets():
    jog, controller, clock = remote()
    old = packet(jog, (0, 1, 0, 1, -1))
    jog.update(old)
    jog.tick(0.05)
    controller.tick(0.05)
    frozen = controller.angles.copy()
    opening = controller.gripper
    clock.advance(0.26)
    assert jog.tick(0.05)
    assert not jog.active
    assert jog.epoch == old[-1] + 1
    assert not jog.update([0, 1, 0, 1, -1, 1, old[-1]])
    controller.tick(0.1)
    assert controller.angles == pytest.approx(frozen)
    assert controller.gripper == opening


def test_repeated_sequence_cannot_keep_a_held_thumb_alive():
    jog, controller, clock = remote()
    message = packet(jog)
    assert jog.update(message)
    clock.advance(0.2)
    assert not jog.update(message)
    clock.advance(0.06)
    jog.tick(0.01)
    assert not jog.active


def test_old_arrival_cannot_start_a_gesture():
    jog, controller, clock = remote()
    assert not jog.update(packet(jog), received_at=clock() - 0.4)
    assert not jog.active


def test_new_packet_cannot_hide_an_expired_lease_before_the_next_tick():
    jog, controller, clock = remote()
    jog.update(packet(jog))
    epoch = jog.epoch
    clock.advance(0.26)
    assert not jog.update(packet(jog, sequence=1))
    assert not jog.active
    assert jog.epoch == epoch + 1


def test_paused_hardware_cannot_enable_itself_from_a_thumb_packet():
    hardware = FakeHardware()
    jog, controller, clock = remote(hardware)
    controller.connect()
    assert not jog.update(packet(jog))
    assert not jog.active
    assert not controller.enabled
    assert not hardware.armed
    assert not hardware.sent


def test_release_preserves_enabled_hardware_and_does_not_relax_a_loaded_grasp():
    hardware = FakeHardware()
    jog, controller, clock = remote(hardware)
    controller.connect()
    controller.enable()
    advance(jog, controller, clock, (0, 0, 0, 0, -1), steps=3)
    commanded_grip = controller.gripper
    hardware.positions["gripper"] = commanded_grip + 10
    jog.update(packet(jog, (0, 0, 0, 0, 0), sequence=4))
    assert controller.enabled
    assert hardware.armed
    assert controller.gripper == commanded_grip
    assert controller.target_gripper == commanded_grip
    assert hardware.sent[-1]["gripper"] == commanded_grip


def test_stick_held_down_stops_at_the_height_limit_and_stays_active():
    jog, controller, clock = remote()
    advance(jog, controller, clock, (0, 0, -1, 0, 0), steps=330)
    assert jog.active
    assert point(controller)[2] == pytest.approx(controller.model.min_height, abs=0.003)
    assert point(controller)[2] >= controller.model.min_height


def remote_at_rest():
    hardware = FakeHardware()
    hardware.joint_limits = dict.fromkeys(JOINT_NAMES, (-106.0, 106.0))
    hardware.positions.update(REST_POSE)
    jog, controller, clock = remote(hardware)
    controller.connect()
    controller.enable()
    return jog, controller, clock


def test_stick_up_lifts_straight_out_of_the_rest_pose():
    jog, controller, clock = remote_at_rest()
    start = point(controller)
    advance(jog, controller, clock, (0, 0, 1, 0, 0), steps=45)
    assert controller.enabled
    assert point(controller)[:2] == pytest.approx(start[:2], abs=0.001)
    assert point(controller)[2] == pytest.approx(start[2] + 0.03, abs=0.001)


def test_lift_from_rest_crosses_up_to_the_height_limit_without_a_jump():
    jog, controller, clock = remote_at_rest()
    previous_height = point(controller)[2]
    for sequence in range(150):
        assert jog.update(packet(jog, (0, 0, 1, 0, 0), sequence))
        jog.tick(1 / 30)
        controller.tick(1 / 30)
        clock.advance(1 / 30)
        height = point(controller)[2]
        assert height - previous_height < 0.0015
        previous_height = height
    assert controller.enabled
    assert point(controller)[2] > controller.model.min_height


@pytest.mark.parametrize("rates", [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, -1, 0, 0)])
def test_other_stick_directions_do_nothing_at_the_rest_pose(rates):
    jog, controller, clock = remote_at_rest()
    start = point(controller)
    advance(jog, controller, clock, rates, steps=30)
    assert controller.enabled
    assert point(controller) == pytest.approx(start, abs=0.0005)


def test_unreachable_translation_cancels_every_axis():
    jog, controller, clock = remote()
    jog.update(packet(jog, (0, 1, 0, 1, -1)))
    def refuse_target(*args, **kwargs):
        raise ValueError("Target is outside the workspace")
    controller.model.solve = refuse_target
    with pytest.raises(ValueError):
        jog.tick(0.05)
    assert not jog.active
    assert jog.epoch > 0
    assert controller.target_gripper == controller.gripper


def test_failed_hold_cannot_leave_a_remote_gesture_active():
    jog, controller, clock = remote()
    jog.update(packet(jog))
    def fail():
        raise OSError("USB unavailable")
    controller.hold_motion = fail
    with pytest.raises(OSError):
        jog.cancel(hold=True)
    assert not jog.active
    assert jog.epoch == 1


@pytest.mark.parametrize("payload", [
    {}, [], [0] * 6, [0] * 8, [0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 1.1, 0, 0, 0], [0, 0, 0, 0, -1.1, 0, 0],
    [0, 0, float("nan"), 0, 0, 0, 0], [0, 0, 0, float("inf"), 0, 0, 0],
    [True, 0, 0, 0, 0, 0, 0], [0, "1", 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0.5, 0], [0, 1, 0, 0, 0, -1, 0], [0, 1, 0, 0, 0, 0, -1],
])
def test_invalid_packets_are_rejected_without_activating_input(payload):
    jog, controller, clock = remote()
    with pytest.raises(ValueError):
        jog.update(payload)
    assert not jog.active


def test_json_packet_uses_the_same_validation():
    jog, controller, clock = remote()
    assert jog.update(json.dumps(packet(jog)))
    assert jog.active
