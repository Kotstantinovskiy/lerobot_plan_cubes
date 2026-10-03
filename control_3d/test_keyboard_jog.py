"""Keyboard Cartesian motion and expired input without opening hardware."""

import json

import numpy as np
import pytest

from controller import Controller
from keyboard_jog import KeyboardJog
from kinematics import Kinematics
from test_controller import FakeHardware


class Clock:
    def __init__(self):
        self.value = 1.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def keyboard(hardware=None):
    clock = Clock()
    controller = Controller(Kinematics(), hardware)
    return KeyboardJog(controller, clock), controller, clock


def packet(jog, direction, sequence=0):
    return [*direction, sequence, jog.epoch]


def point(controller):
    return controller.model.forward(controller.angles)[:3, 3].copy()


def test_horizontal_hold_preserves_height_and_moves_twenty_millimetres_per_second():
    jog, controller, clock = keyboard()
    initial = point(controller)
    opening = controller.gripper
    roll = controller.angles[4]
    for index in range(30):
        jog.update(packet(jog, (0, 1, 0), index))
        jog.tick(1 / 30)
        controller.tick(1 / 30)
        clock.advance(1 / 30)
    current = point(controller)
    assert current[1] - initial[1] == pytest.approx(0.02, abs=0.0001)
    assert current[0] == pytest.approx(initial[0], abs=0.0001)
    assert current[2] == pytest.approx(initial[2], abs=0.0001)
    assert controller.gripper == opening
    assert controller.angles[4] == roll


@pytest.mark.parametrize("sign", [-1, 1])
def test_vertical_input_changes_only_height(sign):
    jog, controller, clock = keyboard()
    initial = point(controller)
    for index in range(15):
        jog.update(packet(jog, (0, 0, sign), index))
        jog.tick(1 / 30)
        controller.tick(1 / 30)
        clock.advance(1 / 30)
    current = point(controller)
    assert current[:2] == pytest.approx(initial[:2], abs=0.0001)
    assert current[2] - initial[2] == pytest.approx(sign * 0.01, abs=0.0001)


def test_release_holds_once_and_discards_pending_mouse_motion():
    jog, controller, clock = keyboard()
    controller.set_joints(np.deg2rad([25, -20, 45, -20, 0]))
    initial = controller.angles.copy()
    jog.update(packet(jog, (0, 1, 0)))
    assert controller.target_angles == pytest.approx(initial)
    jog.tick(1 / 30)
    controller.tick(1 / 30)
    jog.update(packet(jog, (0, 0, 0), 1))
    assert not jog.active
    frozen = controller.angles.copy()
    for _ in range(10):
        clock.advance(1 / 30)
        jog.tick(1 / 30)
        controller.tick(1 / 30)
    assert controller.angles == pytest.approx(frozen)
    assert jog.epoch == 0


def test_expired_heartbeat_holds_and_invalidates_old_key_messages():
    jog, controller, clock = keyboard()
    jog.update(packet(jog, (0, 1, 0)))
    old_epoch = jog.epoch
    jog.tick(0.05)
    clock.advance(0.26)
    assert jog.tick(0.05)
    assert not jog.active
    assert jog.epoch > old_epoch
    assert not jog.update([0, 1, 0, 1, old_epoch])


def test_a_queued_old_heartbeat_cannot_start_motion_after_a_stalled_loop():
    jog, controller, clock = keyboard()
    assert not jog.update(packet(jog, (0, 1, 0)), received_at=clock() - 0.4)
    assert not jog.active


def test_replayed_sequence_does_not_extend_the_input_lease():
    jog, controller, clock = keyboard()
    message = packet(jog, (0, 1, 0), 3)
    assert jog.update(message)
    clock.advance(0.2)
    assert not jog.update(message)
    clock.advance(0.06)
    jog.tick(0.01)
    assert not jog.active


def test_large_frame_cannot_request_more_than_one_millimetre():
    jog, controller, clock = keyboard()
    initial = point(controller)
    jog.update(packet(jog, (0, 1, 0)))
    jog.tick(10)
    target = controller.model.forward(controller.target_angles)[:3, 3]
    assert np.linalg.norm(target - initial) <= 0.00101


def test_paused_hardware_cannot_start_jog_or_enable_motors():
    hardware = FakeHardware()
    jog, controller, clock = keyboard(hardware)
    controller.connect()
    assert not jog.update(packet(jog, (0, 1, 0)))
    assert not jog.active
    assert not hardware.armed
    assert not hardware.sent


def test_key_release_retains_enabled_hardware_without_replaying_a_goal():
    hardware = FakeHardware()
    jog, controller, clock = keyboard(hardware)
    controller.connect()
    controller.enable()
    jog.update(packet(jog, (0, 1, 0)))
    jog.tick(1 / 30)
    controller.tick(1 / 30)
    jog.update(packet(jog, (0, 0, 0), 1))
    assert controller.enabled
    assert hardware.armed
    assert controller.target_angles == pytest.approx(controller.measured_angles)
    count = len(hardware.sent)
    jog.tick(1 / 30)
    assert len(hardware.sent) == count


def test_unreachable_step_cancels_input_and_disables_hardware_movement():
    hardware = FakeHardware()
    jog, controller, clock = keyboard(hardware)
    controller.connect()
    controller.enable()
    jog.update(packet(jog, (0, 1, 0)))
    def refuse_target(*args, **kwargs):
        raise ValueError("Target is outside the workspace")
    controller.model.solve = refuse_target
    with pytest.raises(ValueError, match="workspace"):
        jog.tick(1 / 30)
    assert not jog.active
    assert not controller.enabled
    assert hardware.armed


@pytest.mark.parametrize("value", [
    [0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0],
    [0, float("nan"), 0, 0, 0], [0, 1, 0, 0.5, 0],
    [0, 1, 0, -1, 0], [0, True, 0, 0, 0], [0, "1", 0, 0, 0], {},
])
def test_invalid_keyboard_packets_are_rejected(value):
    jog, controller, clock = keyboard()
    with pytest.raises(ValueError):
        jog.update(value)
    assert not jog.active


def test_json_packet_uses_the_same_validation():
    jog, controller, clock = keyboard()
    assert jog.update(json.dumps(packet(jog, (0, 1, 0))))
    assert jog.active
