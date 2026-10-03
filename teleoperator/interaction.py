"""Ordered browser input and ownership, independent of the viewer."""

import json
import time
from collections.abc import Callable, Collection
from typing import Any, NamedTuple

import numpy as np

from controller import Controller
from drag import DragTarget
from input_queue import MOTION, TRANSITIONS, HeartbeatArrival, InputEvent, InputQueue
from jog_lease import MAX_MESSAGE_NUMBER
from keyboard_jog import KeyboardJog
from kinematics import WRIST_ROLL_INDEX
from remote_jog import RemoteJog

MODES = ("Тянуть захват", "Оси XYZ")
JOG_INPUTS = {"keyboard", "remote"}
# These pause, and a pause cancels the jogs by itself.
PAUSING_TRANSITIONS = {"sync", "mode", "yield", "layout"}
MAX_EVENTS_PER_PROCESS = 64


class Heartbeat(NamedTuple):
    sequence: int
    is_active: bool
    received_at: float


def _parse_heartbeat(payload):
    """Return (sequence, is_active) from a heartbeat message, or None if it is malformed."""
    try:
        sequence, is_active = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, ValueError):
        return None
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence < MAX_MESSAGE_NUMBER:
        return None
    if not isinstance(is_active, bool):
        return None
    return sequence, is_active


class Interaction:
    OPERATOR_TIMEOUT = 0.5
    # With nobody connected this long the arm returns to rest; long enough to survive a page reload.
    NOBODY_CONNECTED_SECONDS = 5.0

    def __init__(self, controller: Controller, clock: Callable[[], float] = time.monotonic) -> None:
        self.controller = controller
        self.owner = None
        self.direct_mode = True
        self.drag = DragTarget()
        self.keyboard = KeyboardJog(controller, clock)
        self.remote = RemoteJog(controller, clock)
        self._clock = clock
        self._queue = InputQueue()
        self._heartbeats = {}
        self._owner_since = None
        self._heartbeat_stopped = False
        self._nobody_since = None

    @property
    def mode(self) -> str:
        return MODES[0 if self.direct_mode else 1]

    @property
    def keyboard_epoch(self) -> int:
        return self.keyboard.epoch

    @property
    def keyboard_active(self) -> bool:
        return self.keyboard.active

    @property
    def remote_epoch(self) -> int:
        return self.remote.epoch

    @property
    def remote_active(self) -> bool:
        return self.remote.active

    def submit(self, client_id: int | None, kind: str, value: Any = None) -> None:
        """Collect input; only process() may touch the controller or motors."""
        if client_id is None:
            return
        if kind == "heartbeat":
            self._submit_heartbeat(client_id, value)
            return
        if kind in JOG_INPUTS:
            value = (value, self._clock())
        event = InputEvent(client_id, kind, value)
        if kind == "stop":
            self._queue.put_stop(event)
        else:
            self._queue.put(event, is_authorized=self.owner is None or self.owner == client_id)

    def _submit_heartbeat(self, client_id, payload):
        if client_id == self.owner and self._heartbeat_stopped:
            # Commands queued so far were issued while the operator was away. Anything
            # after this heartbeat is fresh, like the click that brought the page back.
            self._queue.drop_commands_of(self.owner)
        self._queue.put_heartbeat(HeartbeatArrival(client_id, payload, self._clock()))

    def pause(self, reason: str) -> None:
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
        """Read foreground leases before commands; return (did the owner stop, current time)."""
        arrivals = self._queue.take_heartbeats()
        now = self._clock()
        self._heartbeats = {client: state for client, state in self._heartbeats.items() if client in clients}
        stop_owner = False
        for client, payload, received_at in arrivals:
            parsed = _parse_heartbeat(payload) if client in clients else None
            if parsed is None:
                continue
            sequence, is_active = parsed
            previous = self._heartbeats.get(client)
            if previous is not None and sequence <= previous.sequence:
                continue
            if not 0 <= now - received_at <= self.OPERATOR_TIMEOUT:
                continue
            if client == self.owner and self._owner_went_away(previous, is_active, received_at):
                stop_owner = True
            self._heartbeats[client] = Heartbeat(sequence, is_active, received_at)
        return stop_owner, now

    def _owner_went_away(self, previous, is_active, received_at):
        """Return whether a heartbeat says the owner left or reveals a gap since the last one."""
        if not is_active:
            return True
        baseline = previous.received_at if previous is not None else self._owner_since
        has_gap = baseline is not None and received_at - baseline > self.OPERATOR_TIMEOUT
        return has_gap and not self._heartbeat_stopped

    def _foreground(self, client, now):
        heartbeat = self._heartbeats.get(client)
        if heartbeat is None:
            return self._owner_since is None or now - self._owner_since <= self.OPERATOR_TIMEOUT
        return heartbeat.is_active and now - heartbeat.received_at <= self.OPERATOR_TIMEOUT

    def _forget_owner(self):
        self.owner = None
        self._owner_since = None
        self._heartbeat_stopped = False

    def _take_ownership(self, client_id, now):
        self.owner = client_id
        self._owner_since = now

    def process(self, clients: Collection[int], dt: float = 0.0) -> bool:
        """Handle a bounded amount of input on the motor-control thread; return whether anything changed."""
        stop_owner, now = self._read_heartbeats(clients)
        changed = self._check_owner_presence(clients, stop_owner, now)
        for _ in range(MAX_EVENTS_PER_PROCESS):
            event = self._queue.pop()
            if event is None:
                break
            changed = True
            self._handle_event(event, clients, now)
        return self._advance_motion(clients, now, dt) or changed

    def _check_owner_presence(self, clients, stop_owner, now):
        """Pause when the owner left or went to the background; return whether anything changed."""
        if self.owner is None:
            return False
        if self.owner not in clients:
            self.pause("Operator disconnected; holding position")
            self._forget_owner()
            return True
        was_stopped = self._heartbeat_stopped
        if stop_owner or not self._foreground(self.owner, now):
            if not was_stopped:
                self.pause("Operator heartbeat lost; holding position")
            self._heartbeat_stopped = True
            self._queue.drop_commands_of(self.owner)
            return not was_stopped
        # submit() already dropped commands queued while the operator was absent.
        self._heartbeat_stopped = False
        return was_stopped

    def _handle_event(self, event, clients, now):
        client_id, kind, value = event
        if kind == "stop":
            # The request remains valid if its browser closes immediately.
            self.controller.cancel_return()
            self.pause("Stopped; motors hold the current pose")
            return
        if client_id not in clients:
            return
        if kind in {"yield", "layout"} and client_id != self.owner:
            return
        if self.owner is not None and self.owner != client_id:
            return
        if self.owner is None:
            if not self._foreground(client_id, now):
                return
            if kind in JOG_INPUTS:
                self._claim_with_jog(event, now)
                return
            self._take_ownership(client_id, now)
        try:
            self._apply(client_id, kind, value)
        except Exception as error:
            self.pause(f"Control error: {error}")

    def _claim_with_jog(self, event, now):
        """Give the arm to the sender of a jog message only if that message really starts motion."""
        client_id, kind, (payload, received_at) = event
        rate_count = self._jog(kind).RATE_COUNT
        try:
            payload = json.loads(payload) if isinstance(payload, str) else payload
            has_rates = isinstance(payload, (list, tuple)) and len(payload) == rate_count + 2
            if not has_rates or not any(payload[:rate_count]) or not self._update_jog(kind, payload, received_at):
                return
        except (TypeError, ValueError):
            return
        except Exception as error:
            self.pause(f"Control error: {error}")
            return
        self._take_ownership(client_id, now)

    def _advance_motion(self, clients, now, dt):
        """Advance ongoing motion by one control step; return whether anything changed."""
        try:
            keyboard_changed = self.keyboard.tick(dt)
            remote_changed = self.remote.tick(dt)
            nobody_changed = self._leave_if_nobody_connected(clients, now)
            return_changed = self.controller.tick_return(dt)
        except Exception as error:
            self.pause(f"Control error: {error}")
            return True
        return keyboard_changed or remote_changed or nobody_changed or return_changed

    def _return_or_disconnect(self):
        """Bring a held arm back to rest before freeing the port; an arm without torque needs no move."""
        if self.controller.returning:
            return
        if self.controller.can_return_to_rest:
            self.controller.start_return()
        else:
            self.controller.disconnect()

    def _leave_if_nobody_connected(self, clients, now):
        """Bring the arm back and free the port once nobody has been connected for a while."""
        controller = self.controller
        if clients or not controller.connected or controller.returning or controller.return_stopped:
            self._nobody_since = None
            return False
        if self._nobody_since is None:
            self._nobody_since = now
        if now - self._nobody_since < self.NOBODY_CONNECTED_SECONDS:
            return False
        self._nobody_since = None
        self._return_or_disconnect()
        return True

    def _jog(self, kind):
        return self.keyboard if kind == "keyboard" else self.remote

    def _update_jog(self, kind, payload, received_at):
        jog = self._jog(kind)
        other = self.remote if kind == "keyboard" else self.keyboard
        was_active = jog.active
        accepted = jog.update(payload, received_at)
        if accepted and jog.active:
            self.drag.clear()
            if not was_active:
                other.cancel_and_hold()
        return accepted

    def _apply(self, client_id, kind, value):
        if kind in JOG_INPUTS:
            payload, received_at = value
            self._update_jog(kind, payload, received_at)
        elif kind in TRANSITIONS:
            self._apply_transition(kind, value)
        else:
            self._apply_command(client_id, kind, value)

    def _apply_transition(self, kind, value):
        self.drag.clear()
        if kind not in PAUSING_TRANSITIONS:
            self.keyboard.cancel()
            self.remote.cancel()
        if kind == "connect":
            self.controller.connect()
        elif kind == "enable":
            self.controller.enable()
        elif kind == "release":
            self.controller.release()
        elif kind == "disconnect":
            self._return_or_disconnect()
        elif kind == "yield":
            self.pause("Control released; movement paused")
            self._queue.drop_commands_of(self.owner)
            self._forget_owner()
        elif kind == "layout":
            self.pause("Control layout changed; movement paused")
            self._queue.drop_commands_of(self.owner)
        elif kind == "sync":
            self.pause("Target synchronized; movement paused")
        elif kind == "mode":
            self._change_mode(value)

    def _change_mode(self, mode):
        if mode not in MODES:
            raise ValueError("Unknown control mode")
        self.pause("Control mode changed; movement paused")
        self.direct_mode = mode == MODES[0]

    def _apply_command(self, client_id, kind, value):
        controller = self.controller
        if kind in MOTION:
            self.keyboard.cancel_and_hold()
            self.remote.cancel_and_hold()
        if kind == "target" and not self.direct_mode:
            controller.set_target(value)
        elif kind == "drag" and self.direct_mode:
            self._apply_drag(client_id, value)
        elif kind == "gripper":
            controller.set_gripper(value)
        elif kind in {"roll", "joint"}:
            angles = controller.target_angles.copy()
            index, degrees = (WRIST_ROLL_INDEX, value) if kind == "roll" else value
            angles[index] = np.deg2rad(degrees)
            controller.set_joints(angles)
        elif kind == "speed":
            controller.speed = float(value)

    def _apply_drag(self, client_id, value):
        controller = self.controller
        name, phase, grab, pointer = value
        angles = controller.measured_angles if controller.connected else controller.angles
        tool_position = controller.model.forward(angles)[:3, 3]
        position = self.drag.update(client_id, name, phase, grab, pointer, tool_position)
        if position is not None:
            controller.set_target(position)
