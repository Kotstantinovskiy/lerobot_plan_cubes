"""Bounded five-axis thumb control with an expiring input lease."""

import json
import math
import time

import numpy as np


class RemoteJog:
    SPEED = 0.02
    ROLL_SPEED = 20.0
    GRIPPER_SPEED = 25.0
    TIMEOUT = 0.25

    def __init__(self, controller, clock=time.monotonic):
        self.controller = controller
        self.clock = clock
        self.epoch = 0
        self.active = False
        self._rates = np.zeros(5)
        self._updated_at = None
        self._sequence = -1
        self._height = 0.0

    def _hold(self):
        # Retain the commanded grasp, which can differ from a loaded jaw's
        # measured opening, and discard any unfinished rate-control step.
        self.controller.target_gripper = self.controller.gripper
        self.controller.hold_motion()

    def cancel(self, hold=False, invalidate=True):
        active = self.active
        self.active = False
        self._rates = np.zeros(5)
        self._updated_at = None
        if invalidate:
            self.epoch += 1
            self._sequence = -1
        if active and hold:
            self._hold()

    def update(self, payload, received_at=None):
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, (list, tuple)) or len(payload) != 7:
            raise ValueError("Remote input must contain five rates, sequence, and epoch")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in payload):
            raise ValueError("Remote input must contain finite numbers")
        rates = np.asarray(payload[:5], dtype=float)
        sequence, epoch = payload[5:]
        if not np.isfinite(rates).all() or np.linalg.norm(rates[:3]) > 1.00001 or np.any(np.abs(rates[3:]) > 1):
            raise ValueError("Remote rates must be finite and bounded by one")
        if any(not isinstance(value, int) or not 0 <= value < 2**53 for value in (sequence, epoch)):
            raise ValueError("Remote sequence and epoch must be nonnegative integers")
        if epoch != self.epoch or sequence <= self._sequence:
            return False
        self._sequence = sequence
        if not np.any(rates):
            self.cancel(hold=True, invalidate=False)
            return True
        now = self.clock()
        received_at = now if received_at is None else received_at
        if not math.isfinite(received_at) or not 0 <= now - received_at <= self.TIMEOUT:
            self.cancel(hold=True)
            return False
        if self.active and received_at - self._updated_at > self.TIMEOUT:
            # A newly arrived packet must not conceal a gap while the control
            # loop was stalled; a fresh gesture is required after expiration.
            self.cancel(hold=True)
            return False
        if not self.controller.motion_allowed:
            return False
        if not self.active:
            self._hold()
            self._height = self.controller.model.forward(self.controller.angles)[2, 3]
        rates[:3] /= max(1.0, np.linalg.norm(rates[:3]))
        self._rates = rates
        self._updated_at = received_at
        self.active = True
        return True

    def tick(self, dt):
        if not self.active:
            return False
        if not self.controller.motion_allowed or self.clock() - self._updated_at > self.TIMEOUT:
            self.cancel(hold=not self.controller.faulted)
            return True
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Remote time step must be finite and nonnegative")
        dt = min(float(dt), 0.05)
        if dt == 0:
            return False
        controller = self.controller
        angles = controller.angles.copy()
        angles[4] = np.clip(
            angles[4] + np.deg2rad(min(self.ROLL_SPEED, controller.speed)) * self._rates[3] * dt,
            controller.model.lower[4], controller.model.upper[4],
        )
        controller.target_angles = angles
        try:
            if np.any(self._rates[:3]):
                position = controller.model.forward(controller.angles)[:3, 3].copy()
                position += self._rates[:3] * self.SPEED * dt
                if self._rates[2] != 0:
                    self._height = position[2]
                position[2] = self._height
                controller.set_target(position)
            else:
                controller.set_joints(angles)
            opening = np.clip(controller.gripper + self._rates[4] * self.GRIPPER_SPEED * dt, 0.0, 100.0)
            controller.set_gripper(float(opening))
        except Exception:
            self.cancel()
            raise
        return True
