"""Small Cartesian keyboard steps with an expiring input lease."""

import json
import math
import time

import numpy as np


class KeyboardJog:
    SPEED = 0.02
    TIMEOUT = 0.25

    def __init__(self, controller, clock=time.monotonic):
        self.controller = controller
        self.clock = clock
        self.epoch = 0
        self.active = False
        self._direction = np.zeros(3)
        self._updated_at = None
        self._sequence = -1
        self._height = 0.0

    def cancel(self, hold=False, invalidate=True):
        """Cancel before a hold attempt, so a failed hold cannot leave input live."""
        active = self.active
        self.active = False
        self._direction = np.zeros(3)
        self._updated_at = None
        if invalidate:
            self.epoch += 1
            self._sequence = -1
        if active and hold:
            self.controller.hold_motion()

    def update(self, payload, received_at=None):
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, (list, tuple)) or len(payload) != 5:
            raise ValueError("Keyboard input must contain direction, sequence, and epoch")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in payload):
            raise ValueError("Keyboard input must contain finite numbers")
        direction = np.asarray(payload[:3], dtype=float)
        sequence, epoch = payload[3:]
        if not np.isfinite(direction).all() or np.linalg.norm(direction) > 1.00001:
            raise ValueError("Keyboard direction must be finite and no longer than one")
        if any(not isinstance(value, int) or not 0 <= value < 2**53 for value in (sequence, epoch)):
            raise ValueError("Keyboard sequence and epoch must be nonnegative integers")
        if epoch != self.epoch or sequence <= self._sequence:
            return False
        self._sequence = sequence
        if not np.any(direction):
            self.cancel(hold=True, invalidate=False)
            return True
        now = self.clock()
        received_at = now if received_at is None else received_at
        if not math.isfinite(received_at) or not 0 <= now - received_at <= self.TIMEOUT:
            self.cancel(hold=True)
            return False
        if not self.controller.motion_allowed:
            return False
        if not self.active:
            self.controller.hold_motion()
            self._height = self.controller.model.forward(self.controller.angles)[2, 3]
        self._direction = direction / max(1.0, np.linalg.norm(direction))
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
            raise ValueError("Keyboard time step must be finite and nonnegative")
        dt = min(float(dt), 0.05)
        if dt == 0:
            return False
        controller = self.controller
        position = controller.model.forward(controller.angles)[:3, 3].copy()
        position += self._direction * self.SPEED * dt
        if self._direction[2] != 0:
            self._height = position[2]
        position[2] = self._height
        # Seed from the last command, never from a distant mouse target.
        controller.target_angles = controller.angles.copy()
        try:
            controller.set_target(position)
        except Exception:
            self.cancel()
            raise
        return True
