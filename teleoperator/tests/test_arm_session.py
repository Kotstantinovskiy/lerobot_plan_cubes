"""A whole operator session through the real input, jog and controller code, with stand-in motors."""

import json

import pytest

from controller import Controller
from fakes import REST_POSE, Clock, FakeHardware, point
from interaction import Interaction
from kinematics import JOINT_NAMES, Kinematics

OPERATOR = 1
STEP_SECONDS = 1 / 30
REST_GRIPPER = 0.4
LIFT_SECONDS = 5.0
MAX_RETURN_SECONDS = 120.0


class ArmSession:
    """Run the control loop one step at a time, with the operator's page sending its heartbeat."""

    def __init__(self):
        self.hardware = FakeHardware()
        self.hardware.joint_limits = dict.fromkeys(JOINT_NAMES, (-106.0, 106.0))
        self.hardware.rest_positions = {**REST_POSE, "gripper": REST_GRIPPER}
        self.hardware.positions.update(REST_POSE, gripper=REST_GRIPPER)
        self.controller = Controller(Kinematics(), self.hardware)
        self.clock = Clock()
        self.interaction = Interaction(self.controller, self.clock)
        self.sequence = 0

    def next_sequence(self):
        self.sequence += 1
        return self.sequence

    def step(self):
        self.interaction.submit(OPERATOR, "heartbeat", json.dumps([self.next_sequence(), True]))
        self.interaction.process({OPERATOR}, STEP_SECONDS)
        self.controller.tick(STEP_SECONDS)
        self.clock.advance(STEP_SECONDS)

    def press(self, kind):
        self.interaction.submit(OPERATOR, kind)
        self.step()

    def hold_stick(self, rates, seconds):
        for _ in range(int(seconds / STEP_SECONDS)):
            message = [*rates, self.next_sequence(), self.interaction.remote_epoch]
            self.interaction.submit(OPERATOR, "remote", message)
            self.step()

    def wait_until_disconnected(self):
        for _ in range(int(MAX_RETURN_SECONDS / STEP_SECONDS)):
            if not self.controller.connected:
                return
            self.step()
        pytest.fail("The arm did not get back to rest")


def enabled_session():
    session = ArmSession()
    session.press("connect")
    session.press("enable")
    return session


def test_operator_can_enable_the_arm_at_its_rest_pose():
    session = enabled_session()
    assert session.controller.enabled
    assert session.controller.must_rise_first
    assert session.interaction.owner == OPERATOR


def test_stick_lifts_the_arm_out_of_the_rest_pose_without_a_pause():
    session = enabled_session()
    start_height = point(session.controller)[2]
    session.hold_stick((0, 0, 1, 0, 0), LIFT_SECONDS)
    assert session.controller.enabled
    assert point(session.controller)[2] > start_height + 0.05
    assert not session.controller.must_rise_first


def test_disconnecting_walks_the_lifted_arm_home_and_frees_the_port():
    session = enabled_session()
    session.hold_stick((0, 0, 1, 0, 0), LIFT_SECONDS)
    session.press("disconnect")
    assert session.controller.returning
    session.wait_until_disconnected()
    assert session.hardware.sent[-1] == pytest.approx({**REST_POSE, "gripper": REST_GRIPPER})
    assert not session.hardware.armed
    assert not session.controller.returning


def test_stop_during_the_way_home_holds_the_arm_and_keeps_the_port():
    session = enabled_session()
    session.hold_stick((0, 0, 1, 0, 0), LIFT_SECONDS)
    session.press("disconnect")
    session.step()
    session.press("stop")
    assert not session.controller.returning
    assert session.controller.connected
    assert session.hardware.armed
    assert session.controller.status == "Stopped; motors hold the current pose"
