"""Small Cartesian keyboard steps with an expiring input lease."""

import numpy as np

from jog_lease import MAX_DIRECTION_LENGTH, JogLease


class KeyboardJog(JogLease):
    NAME = "Keyboard"
    SHAPE_DESCRIPTION = "direction"
    RATE_COUNT = 3

    def _check_rates(self, rates):
        if not np.isfinite(rates).all() or np.linalg.norm(rates) > MAX_DIRECTION_LENGTH:
            raise ValueError("Keyboard direction must be finite and no longer than one")

    def _move(self, dt):
        position = self._next_position(dt)
        # Seed from the last command, never from a distant mouse target.
        self.controller.target_angles = self.controller.angles.copy()
        self.controller.set_target(position)
