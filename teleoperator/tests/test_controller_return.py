"""Walking back to the rest pose before the port is freed."""

import pytest

from controller import RETURNING_STATUS, Controller
from fakes import REST_POSE, FakeHardware
from kinematics import JOINT_NAMES, Kinematics
from rest_return import RETURN_SPEED_DEGREES_PER_SECOND


def held_away_from_rest():
    hardware = FakeHardware()
    hardware.joint_limits = dict.fromkeys(JOINT_NAMES, (-106.0, 106.0))
    hardware.rest_positions = {**REST_POSE, "gripper": 0.4}
    hardware.positions.update(shoulder_lift=-60.0, elbow_flex=60.0, wrist_flex=60.0, wrist_roll=-50.0, gripper=30.0)
    controller = Controller(Kinematics(), hardware)
    controller.connect()
    controller.enable()
    return controller, hardware


def run_return(controller, max_ticks=3000):
    for _ in range(max_ticks):
        if not controller.returning:
            return
        controller.tick_return(1 / 30)
        controller.tick(1 / 30)
    pytest.fail("The return did not finish")


def test_return_walks_to_rest_then_releases_torque_and_frees_the_port():
    controller, hardware = held_away_from_rest()
    controller.start_return()
    assert controller.returning
    assert controller.status == RETURNING_STATUS
    run_return(controller)
    assert hardware.sent[-1] == pytest.approx({**REST_POSE, "gripper": 0.4})
    assert not controller.connected
    assert not hardware.armed
    assert controller.status == "Disconnected: preview only"


def test_return_moves_at_a_bounded_speed():
    controller, hardware = held_away_from_rest()
    controller.start_return()
    hardware.sent.clear()
    run_return(controller)
    max_step = RETURN_SPEED_DEGREES_PER_SECOND / 30
    for previous, current in zip(hardware.sent, hardware.sent[1:], strict=False):
        assert max(abs(current[name] - previous[name]) for name in JOINT_NAMES) <= max_step + 1e-6


def test_return_stops_and_holds_when_a_joint_does_not_follow():
    controller, hardware = held_away_from_rest()
    controller.start_return()
    hardware.follow = False
    for _ in range(600):
        controller.tick_return(1 / 30)
        controller.tick(1 / 30)
    assert not controller.returning
    assert controller.return_stopped
    assert controller.connected
    assert hardware.armed


def test_enable_takes_over_from_a_running_return():
    controller, hardware = held_away_from_rest()
    controller.start_return()
    controller.tick_return(1 / 30)
    controller.enable()
    assert not controller.returning
    assert controller.enabled


def test_return_needs_held_motors_and_a_known_rest_pose():
    controller, hardware = held_away_from_rest()
    assert controller.can_return_to_rest
    controller.release()
    assert not controller.can_return_to_rest
    controller.enable()
    hardware.rest_positions = None
    assert not controller.can_return_to_rest
