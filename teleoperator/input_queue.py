"""Browser input waiting for the control loop; safe to fill from the viewer's thread."""

import threading
from collections import deque
from typing import Any, NamedTuple

TRANSITIONS = {"connect", "enable", "stop", "release", "disconnect", "sync", "mode", "yield", "layout"}
MOTION = {"target", "gripper", "roll", "joint", "drag", "keyboard", "remote"}
# Only the newest queued value of these matters.
CONTINUOUS_INPUTS = {"target", "gripper", "roll", "speed", "joint", "keyboard", "remote"}
# A browser that goes away can still stop the arm and give up control.
ALWAYS_KEPT_KINDS = {"stop", "yield"}


class InputEvent(NamedTuple):
    client_id: Any
    kind: str
    value: Any


class HeartbeatArrival(NamedTuple):
    client_id: Any
    payload: Any
    received_at: float


def _is_drag_boundary(event):
    return event.kind == "drag" and event.value[1] != "update"


def _continuous_key(event):
    """Return what identifies a stream of input where only the newest value matters, or None."""
    if event.kind == "drag":
        return (event.client_id, event.kind, event.value[0]) if event.value[1] == "update" else None
    if event.kind in CONTINUOUS_INPUTS:
        return (event.client_id, event.kind, event.value[0] if event.kind == "joint" else None)
    return None


class InputQueue:
    def __init__(self):
        self._events = deque()
        self._heartbeats = deque()
        self._lock = threading.Lock()

    def put_stop(self, event: InputEvent) -> None:
        """Put a stop first; it also discards a queued enable or an old drag start."""
        with self._lock:
            self._events.clear()
            self._events.appendleft(event)

    def put(self, event: InputEvent, *, is_authorized: bool) -> None:
        """Queue an event; an authorized one also replaces older input it makes pointless."""
        with self._lock:
            if is_authorized and event.kind in TRANSITIONS:
                self._events = deque(
                    queued
                    for queued in self._events
                    if queued.client_id != event.client_id or queued.kind not in MOTION
                )
            key = _continuous_key(event)
            if is_authorized and key is not None:
                self._drop_older_input(key)
            self._events.append(event)

    def _drop_older_input(self, key):
        """Remove the older value of this input, so retained inputs keep their arrival order."""
        for index in range(len(self._events) - 1, -1, -1):
            queued = self._events[index]
            if queued.kind in TRANSITIONS or _is_drag_boundary(queued):
                return
            if _continuous_key(queued) == key:
                del self._events[index]
                return

    def pop(self) -> InputEvent | None:
        """Return the oldest event, or None when nothing is waiting."""
        with self._lock:
            return self._events.popleft() if self._events else None

    def drop_commands_of(self, client_id: Any) -> None:
        """Forget what this client has queued, except for the kinds that always stay valid."""
        with self._lock:
            self._events = deque(
                event for event in self._events if event.client_id != client_id or event.kind in ALWAYS_KEPT_KINDS
            )

    def put_heartbeat(self, arrival: HeartbeatArrival) -> None:
        with self._lock:
            self._heartbeats.append(arrival)

    def take_heartbeats(self) -> list[HeartbeatArrival]:
        """Return every heartbeat that arrived since the last call, oldest first."""
        with self._lock:
            arrivals = list(self._heartbeats)
            self._heartbeats.clear()
        return arrivals
