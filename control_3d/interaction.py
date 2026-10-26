"""Ordered browser input and ownership, independent of the viewer."""

from collections import deque
import json
import threading
import time

import numpy as np

from drag import DragTarget
from keyboard_jog import KeyboardJog
from remote_jog import RemoteJog

MODES = ("Тянуть захват", "Оси XYZ")
TRANSITIONS = {"connect", "enable", "stop", "release", "disconnect", "sync", "mode", "yield", "layout"}
MOTION = {"target", "gripper", "roll", "joint", "drag", "keyboard", "remote"}


class Interaction:
    OPERATOR_TIMEOUT = 0.5

    def __init__(self, controller, clock=time.monotonic):
        self.controller = controller
        self.owner = None
        self.direct_mode = True
        self.drag = DragTarget()
        self.keyboard = KeyboardJog(controller, clock)
        self.remote = RemoteJog(controller, clock)
        self._events = deque()
        self._heartbeat_events = deque()
        self._heartbeats = {}
        self._owner_since = None
        self._heartbeat_stopped = False
        self._lock = threading.Lock()

    @property
    def mode(self):
        return MODES[0 if self.direct_mode else 1]

    @property
    def keyboard_epoch(self):
        return self.keyboard.epoch

    @property
    def keyboard_active(self):
        return self.keyboard.active

    @property
    def remote_epoch(self):
        return self.remote.epoch

    @property
    def remote_active(self):
        return self.remote.active

    def submit(self, client_id, kind, value=None):
        """Collect input; only process() may touch the controller or motors."""
        if client_id is None:
            return
        if kind == "heartbeat":
            with self._lock:
                self._heartbeat_events.append((client_id, value, self.keyboard.clock()))
            return
        if kind in {"keyboard", "remote"}:
            value = (value, self.keyboard.clock())
        with self._lock:
            authorized = self.owner is None or self.owner == client_id
            if kind == "stop":
                # A stop must also discard a queued enable or an old drag start.
                self._events.clear()
                self._events.appendleft((client_id, kind, value))
                return
            if authorized and kind in TRANSITIONS:
                self._events = deque(
                    event for event in self._events
                    if event[0] != client_id or event[1] not in MOTION
                )
            key = self._continuous_key(client_id, kind, value)
            if authorized and key is not None:
                # Remove and append, so retained inputs keep their arrival order.
                for index in range(len(self._events) - 1, -1, -1):
                    previous = self._events[index]
                    if previous[1] in TRANSITIONS or self._drag_boundary(previous):
                        break
                    if self._continuous_key(*previous) == key:
                        del self._events[index]
                        break
            self._events.append((client_id, kind, value))

    @staticmethod
    def _drag_boundary(event):
        return event[1] == "drag" and event[2][1] != "update"

    @staticmethod
    def _continuous_key(client_id, kind, value):
        if kind == "drag":
            return (client_id, kind, value[0]) if value[1] == "update" else None
        if kind in {"target", "gripper", "roll", "speed", "joint", "keyboard", "remote"}:
            return (client_id, kind, value[0] if kind == "joint" else None)
        return None

    def pause(self, reason):
        """Forget active gestures even when a hardware hold fails."""
        self.drag.clear()
        self.keyboard.cancel()
        self.remote.cancel()
        try:
            self.controller.pause(reason)
        except Exception as error:
            self.controller.enabled = False
            self.controller.status = f"{reason}; hold failed: {error}"

    def _read_heartbeats(self, clients):
        """Read foreground leases before commands, retaining inactive transitions."""
        with self._lock:
            events = list(self._heartbeat_events)
            self._heartbeat_events.clear()
        now = self.keyboard.clock()
        self._heartbeats = {client: state for client, state in self._heartbeats.items() if client in clients}
        stop_owner = False
        for client, payload, received_at in events:
            if client not in clients:
                continue
            try:
                sequence, active = json.loads(payload) if isinstance(payload, str) else payload
            except (TypeError, ValueError):
                continue
            if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence < 2**53:
                continue
            if not isinstance(active, bool):
                continue
            previous = self._heartbeats.get(client)
            if previous is not None and sequence <= previous[0]:
                continue
            if not 0 <= now - received_at <= self.OPERATOR_TIMEOUT:
                continue
            if client == self.owner:
                baseline = previous[2] if previous is not None else self._owner_since
                if not active or (
                    not self._heartbeat_stopped and baseline is not None
                    and received_at - baseline > self.OPERATOR_TIMEOUT
                ):
                    stop_owner = True
            self._heartbeats[client] = (sequence, active, received_at)
        return stop_owner, now

    def _foreground(self, client, now):
        state = self._heartbeats.get(client)
        if state is None:
            return self._owner_since is None or now - self._owner_since <= self.OPERATOR_TIMEOUT
        return state[1] and now - state[2] <= self.OPERATOR_TIMEOUT

    def _discard_owner_commands(self):
        with self._lock:
            self._events = deque(
                event for event in self._events
                if event[0] != self.owner or event[1] in {"stop", "yield"}
            )

    def process(self, clients, dt=0.0):
        """Handle a bounded amount of input on the motor-control thread."""
        changed = False
        stop_owner, now = self._read_heartbeats(clients)
        if self.owner is not None and self.owner not in clients:
            self.pause("Operator disconnected; holding position")
            self.owner = None
            self._owner_since = None
            self._heartbeat_stopped = False
            changed = True
        if self.owner is not None:
            if stop_owner or not self._foreground(self.owner, now):
                if not self._heartbeat_stopped:
                    self.pause("Operator heartbeat lost; holding position")
                    changed = True
                self._heartbeat_stopped = True
                self._discard_owner_commands()
            elif self._heartbeat_stopped:
                # Do not replay a command queued while the operator was absent.
                self._discard_owner_commands()
                self._heartbeat_stopped = False
                changed = True
        for _ in range(64):
            with self._lock:
                if not self._events:
                    break
                client_id, kind, value = self._events.popleft()
            changed = True
            if kind == "stop":
                # The request remains valid if its browser closes immediately.
                self.pause("Stopped; motors hold the current pose")
                continue
            if client_id not in clients:
                continue
            if kind in {"yield", "layout"} and client_id != self.owner:
                continue
            if self.owner is not None and self.owner != client_id and kind != "stop":
                continue
            if self.owner is None and kind != "stop":
                if not self._foreground(client_id, now):
                    continue
                if kind in {"keyboard", "remote"}:
                    payload, received_at = value
                    try:
                        payload = json.loads(payload) if isinstance(payload, str) else payload
                        rate_count = 3 if kind == "keyboard" else 5
                        if not isinstance(payload, (list, tuple)) or len(payload) != rate_count + 2 or not any(payload[:rate_count]):
                            continue
                        if not self._update_jog(kind, payload, received_at):
                            continue
                    except (TypeError, ValueError):
                        continue
                    except Exception as error:
                        self.pause(f"Control error: {error}")
                        continue
                    self.owner = client_id
                    self._owner_since = now
                    continue
                self.owner = client_id
                self._owner_since = now
            try:
                self._apply(client_id, kind, value)
            except Exception as error:
                self.pause(f"Control error: {error}")
        try:
            changed = self.keyboard.tick(dt) or changed
            changed = self.remote.tick(dt) or changed
        except Exception as error:
            self.pause(f"Control error: {error}")
            changed = True
        return changed

    def _update_jog(self, kind, payload, received_at):
        jog = self.keyboard if kind == "keyboard" else self.remote
        other = self.remote if kind == "keyboard" else self.keyboard
        was_active = jog.active
        accepted = jog.update(payload, received_at)
        if accepted and jog.active:
            self.drag.clear()
            if not was_active:
                other.cancel(hold=True)
        return accepted

    def _apply(self, client_id, kind, value):
        controller = self.controller
        if kind in {"keyboard", "remote"}:
            payload, received_at = value
            self._update_jog(kind, payload, received_at)
            return
        if kind in TRANSITIONS:
            self.drag.clear()
            if kind not in {"stop", "sync", "mode", "yield", "layout"}:
                self.keyboard.cancel()
                self.remote.cancel()
        elif kind in MOTION:
            self.keyboard.cancel(hold=True)
            self.remote.cancel(hold=True)
        if kind == "connect":
            controller.connect()
        elif kind == "enable":
            controller.enable()
        elif kind == "stop":
            self.pause("Stopped; motors hold the current pose")
        elif kind == "release":
            controller.release()
        elif kind == "disconnect":
            controller.disconnect()
        elif kind == "yield":
            self.pause("Control released; movement paused")
            self._discard_owner_commands()
            self.owner = None
            self._owner_since = None
            self._heartbeat_stopped = False
        elif kind == "layout":
            self.pause("Control layout changed; movement paused")
            self._discard_owner_commands()
        elif kind == "sync":
            self.pause("Target synchronized; movement paused")
        elif kind == "mode":
            if value not in MODES:
                raise ValueError("Unknown control mode")
            self.pause("Control mode changed; movement paused")
            self.direct_mode = value == MODES[0]
        elif kind == "target" and not self.direct_mode:
            controller.set_target(value)
        elif kind == "drag" and self.direct_mode:
            name, phase, grab, pointer = value
            angles = controller.measured_angles if controller.connected else controller.angles
            position = self.drag.update(
                client_id, name, phase, grab, pointer, controller.model.forward(angles)[:3, 3],
            )
            if position is not None:
                controller.set_target(position)
        elif kind == "gripper":
            controller.set_gripper(value)
        elif kind in {"roll", "joint"}:
            angles = controller.target_angles.copy()
            index, degrees = (4, value) if kind == "roll" else value
            angles[index] = np.deg2rad(degrees)
            controller.set_joints(angles)
        elif kind == "speed":
            controller.speed = float(value)
