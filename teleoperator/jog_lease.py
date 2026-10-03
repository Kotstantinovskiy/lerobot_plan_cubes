"""Numbered jog input with an expiring lease, shared by the keyboard and the thumb sticks."""

import json
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from controller import Controller

MAX_MESSAGE_NUMBER = 2**53
# Browsers normalize direction vectors in floating point, so a length of exactly one can read slightly above it.
MAX_DIRECTION_LENGTH = 1.00001
MAX_JOG_STEP_SECONDS = 0.05


class JogLease:
    """Turn messages of [*rates, sequence, epoch] into motion while they keep arriving.

    A message from an old epoch or with a sequence already seen is ignored. Silence longer than
    TIMEOUT ends the lease. A new epoch requires a fresh gesture from the browser.
    Subclasses name the input, check their rates and define what one step of motion does.
    """

    NAME = ""
    SHAPE_DESCRIPTION = ""
    RATE_COUNT = 0
    SPEED = 0.02  # metres per second at full rate
    TIMEOUT = 0.25

    def __init__(self, controller: Controller, clock: Callable[[], float] = time.monotonic) -> None:
        self.controller = controller
        self.clock = clock
        self.epoch = 0
        self.active = False
        self._rates = np.zeros(self.RATE_COUNT)
        self._updated_at = None
        self._sequence = -1
        self._height = 0.0

    def _check_rates(self, rates):
        raise NotImplementedError

    def _move(self, dt):
        raise NotImplementedError

    def _hold(self):
        self.controller.hold_motion()

    def _hides_a_gap(self, received_at):
        """Return whether a new message arrived after the lease should already have expired."""
        return False

    def _end_lease(self):
        """End the lease before any hold attempt, so a failed hold cannot leave input live."""
        was_active = self.active
        self.active = False
        self._rates = np.zeros(self.RATE_COUNT)
        self._updated_at = None
        return was_active

    def _start_new_epoch(self):
        self.epoch += 1
        self._sequence = -1

    def cancel(self) -> None:
        """Stop taking input until a fresh gesture, without commanding the arm."""
        self._end_lease()
        self._start_new_epoch()

    def cancel_and_hold(self) -> None:
        """Stop taking input until a fresh gesture and hold the arm if it was moving."""
        was_active = self._end_lease()
        self._start_new_epoch()
        if was_active:
            self._hold()

    def _release_and_hold(self):
        """End a gesture the operator let go of; the same input stream may start another."""
        if self._end_lease():
            self._hold()

    def _parse(self, payload):
        """Return (rates, sequence, epoch) from a message, or raise ValueError if it is malformed."""
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, (list, tuple)) or len(payload) != self.RATE_COUNT + 2:
            raise ValueError(f"{self.NAME} input must contain {self.SHAPE_DESCRIPTION}, sequence, and epoch")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in payload):
            raise ValueError(f"{self.NAME} input must contain finite numbers")
        rates = np.asarray(payload[: self.RATE_COUNT], dtype=float)
        sequence, epoch = payload[self.RATE_COUNT :]
        self._check_rates(rates)
        if any(not isinstance(value, int) or not 0 <= value < MAX_MESSAGE_NUMBER for value in (sequence, epoch)):
            raise ValueError(f"{self.NAME} sequence and epoch must be nonnegative integers")
        return rates, sequence, epoch

    def update(self, payload: Any, received_at: float | None = None) -> bool:
        """Apply one message; return whether it was accepted."""
        rates, sequence, epoch = self._parse(payload)
        if epoch != self.epoch or sequence <= self._sequence:
            return False
        self._sequence = sequence
        if not np.any(rates):
            self._release_and_hold()
            return True
        now = self.clock()
        received_at = now if received_at is None else received_at
        is_fresh = math.isfinite(received_at) and 0 <= now - received_at <= self.TIMEOUT
        if not is_fresh or self._hides_a_gap(received_at):
            self.cancel_and_hold()
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

    def _next_position(self, dt):
        """Return the tool target one step ahead, keeping the height unless the input changes it."""
        position = self.controller.model.forward(self.controller.angles)[:3, 3].copy()
        position += self._rates[:3] * self.SPEED * dt
        if self._rates[2] != 0:
            self._height = position[2]
        position[2] = self._height
        return position

    def tick(self, dt: float) -> bool:
        """Advance one control step; return whether anything changed."""
        if not self.active:
            return False
        if not self.controller.motion_allowed or self.clock() - self._updated_at > self.TIMEOUT:
            if self.controller.faulted:
                self.cancel()
            else:
                self.cancel_and_hold()
            return True
        if not math.isfinite(dt) or dt < 0:
            raise ValueError(f"{self.NAME} time step must be finite and nonnegative")
        dt = min(float(dt), MAX_JOG_STEP_SECONDS)
        if dt == 0:
            return False
        try:
            self._move(dt)
        except Exception:
            self.cancel()
            raise
        return True
