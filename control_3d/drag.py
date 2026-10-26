"""Convert a grabbed point on a moving mesh into a stable tool target."""

import numpy as np


class DragTarget:
    def __init__(self):
        self._offsets = {}

    def update(self, client_id, node_name, phase, grab_position, pointer_position, tool_position):
        """Preserve the initial grab offset for this client and scene node."""
        if phase not in {"start", "update", "end"}:
            raise ValueError("Drag phase must be start, update, or end")
        grab = self._position(grab_position)
        pointer = self._position(pointer_position)
        tool = self._position(tool_position)
        key = (client_id, node_name)
        if phase == "start":
            self._offsets[key] = tool - grab
            return tool.copy()
        offset = self._offsets.get(key)
        if offset is None:
            return None
        if phase == "end":
            del self._offsets[key]
        return pointer + offset

    def clear(self):
        """Discard active grabs after a mode change or client disconnect."""
        self._offsets.clear()

    @staticmethod
    def _position(value):
        try:
            position = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Drag coordinates must be finite 3D positions") from error
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Drag coordinates must be finite 3D positions")
        return position
