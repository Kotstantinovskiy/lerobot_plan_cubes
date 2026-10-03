"""Bounded five-axis thumb control with an expiring input lease."""

import numpy as np

from jog_lease import MAX_DIRECTION_LENGTH, JogLease
from kinematics import MAX_GRIPPER_OPENING, WRIST_ROLL_INDEX

ROLL_RATE_INDEX = 3
GRIPPER_RATE_INDEX = 4


class RemoteJog(JogLease):
    NAME = "Remote"
    SHAPE_DESCRIPTION = "five rates"
    RATE_COUNT = 5
    ROLL_SPEED = 20.0  # degrees per second at full rate
    GRIPPER_SPEED = 25.0  # percent per second at full rate

    def _check_rates(self, rates):
        is_valid = (
            np.isfinite(rates).all()
            and np.linalg.norm(rates[:3]) <= MAX_DIRECTION_LENGTH
            and not np.any(np.abs(rates[3:]) > 1)
        )
        if not is_valid:
            raise ValueError("Remote rates must be finite and bounded by one")

    def _hold(self):
        # Retain the commanded grasp, which can differ from a loaded jaw's
        # measured opening, and discard any unfinished rate-control step.
        self.controller.target_gripper = self.controller.gripper
        self.controller.hold_motion()

    def _hides_a_gap(self, received_at):
        # A newly arrived packet must not conceal a gap while the control
        # loop was stalled; a fresh gesture is required after expiration.
        return self.active and received_at - self._updated_at > self.TIMEOUT

    def _move(self, dt):
        controller = self.controller
        roll_step = np.deg2rad(min(self.ROLL_SPEED, controller.speed)) * self._rates[ROLL_RATE_INDEX] * dt
        angles = controller.angles.copy()
        angles[WRIST_ROLL_INDEX] = np.clip(
            angles[WRIST_ROLL_INDEX] + roll_step,
            controller.model.lower[WRIST_ROLL_INDEX],
            controller.model.upper[WRIST_ROLL_INDEX],
        )
        controller.target_angles = angles
        if np.any(self._rates[:3]):
            controller.set_target(self._next_position(dt))
        else:
            controller.set_joints(angles)
        opening_step = self._rates[GRIPPER_RATE_INDEX] * self.GRIPPER_SPEED * dt
        controller.set_gripper(float(np.clip(controller.gripper + opening_step, 0.0, MAX_GRIPPER_OPENING)))
