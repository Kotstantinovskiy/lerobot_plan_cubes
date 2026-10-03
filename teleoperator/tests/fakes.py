"""Stand-ins for the motors, the clock and the controller, shared by the test modules."""

import numpy as np

from controller import Controller
from interaction import Interaction
from kinematics import JOINT_NAMES, Kinematics

# Measured on the real follower folded at rest: past the model limits, tool 33 mm below the base plane.
REST_POSE = {
    "shoulder_pan": -2.2,
    "shoulder_lift": -104.4,
    "elbow_flex": 103.1,
    "wrist_flex": 79.6,
    "wrist_roll": -56.6,
}


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
        self.rest_positions = None

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
                name: self.positions[name]
                + float(np.clip(value - self.positions[name], -self.relative_limit, self.relative_limit))
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


class FakeModel:
    def forward(self, angles):
        pose = np.eye(4)
        pose[:3, 3] = (0.3, 0.0, 0.2)
        return pose


class FakeController:
    def __init__(self):
        self.model = FakeModel()
        self.angles = np.zeros(5)
        self.measured_angles = self.angles.copy()
        self.target_angles = self.angles.copy()
        self.connected = False
        self.enabled = False
        self.speed = 20.0
        self.status = "Preview"
        self.calls = []
        self.can_return_to_rest = False
        self.returning = False
        self.return_stopped = False

    def connect(self):
        self.connected = True
        self.calls.append(("connect",))

    def enable(self):
        self.enabled = True
        self.target_angles = self.angles.copy()
        self.calls.append(("enable", self.speed))

    def pause(self, reason):
        self.enabled = False
        self.target_angles = self.angles.copy()
        self.status = reason
        self.calls.append(("pause", reason))

    def release(self):
        self.enabled = False
        self.calls.append(("release",))

    def disconnect(self):
        self.connected = False
        self.enabled = False
        self.calls.append(("disconnect",))

    def set_target(self, position):
        self.calls.append(("target", tuple(position)))

    def set_joints(self, angles):
        self.target_angles = angles.copy()
        self.calls.append(("joints", tuple(np.rad2deg(angles))))

    def set_gripper(self, value):
        self.calls.append(("gripper", value))

    def start_return(self):
        self.returning = True
        self.calls.append(("start_return",))

    def cancel_return(self):
        self.returning = False

    def tick_return(self, dt):
        return False


def interaction_with_owner():
    controller = FakeController()
    interaction = Interaction(controller)
    interaction.submit(1, "speed", 20)
    interaction.process({1, 2})
    assert interaction.owner == 1
    controller.calls.clear()
    return interaction, controller


def drag_event(phase, pointer=(0.3, 0.0, 0.2), node="/nose"):
    return node, phase, np.array((0.3, 0.0, 0.2)), np.array(pointer)


def keyboard_interaction():
    clock = Clock()
    controller = Controller(Kinematics())
    interaction = Interaction(controller, clock)
    return interaction, controller, clock


def keyboard_message(interaction, direction=(0, 1, 0), sequence=0, epoch=None):
    return [*direction, sequence, interaction.keyboard_epoch if epoch is None else epoch]


def remote_message(interaction, rates=(0, 1, 0, 0, 0), sequence=0, epoch=None):
    return [*rates, sequence, interaction.remote_epoch if epoch is None else epoch]


def foreground_interaction():
    clock = Clock()
    controller = FakeController()
    interaction = Interaction(controller, clock)
    interaction.submit(1, "heartbeat", "[1,true]")
    interaction.submit(1, "speed", 20)
    interaction.process({1, 2})
    controller.enabled = True
    controller.calls.clear()
    return interaction, controller, clock


class Clock:
    def __init__(self):
        self.value = 1.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def point(controller):
    return controller.model.forward(controller.angles)[:3, 3].copy()
