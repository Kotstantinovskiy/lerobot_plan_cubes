"""Browser event ordering and operator ownership without a viewer or motors."""

import numpy as np
import pytest

from interaction import Interaction, MODES
from controller import Controller
from kinematics import Kinematics
from test_keyboard_jog import Clock
from test_controller import FakeHardware


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


def test_latest_coalesced_events_keep_their_actual_arrival_order():
    interaction, controller = interaction_with_owner()
    interaction.direct_mode = False
    interaction.submit(1, "target", (0.2, 0.0, 0.2))
    interaction.submit(1, "joint", (0, 15))
    interaction.submit(1, "target", (0.35, 0.0, 0.2))
    interaction.process({1})
    assert [call[0] for call in controller.calls] == ["joints", "target"]
    assert controller.calls[-1][1] == (0.35, 0.0, 0.2)


def test_stop_discards_drag_backlog_and_queued_enable_before_any_motion():
    interaction, controller = interaction_with_owner()
    controller.enabled = True
    for index in range(300):
        interaction.submit(1, "drag", drag_event("start", node=f"/nose{index}"))
    interaction.submit(1, "enable")
    interaction.submit(2, "stop")
    interaction.process({1, 2})
    assert [call[0] for call in controller.calls] == ["pause"]
    assert not controller.enabled
    assert interaction.owner == 1


def test_observer_transition_cannot_delete_owner_input_or_change_mode():
    interaction, controller = interaction_with_owner()
    interaction.submit(1, "gripper", 75)
    interaction.submit(2, "mode", MODES[1])
    assert interaction.process({1, 2})
    assert controller.calls == [("gripper", 75)]
    assert interaction.mode == MODES[0]
    assert interaction.owner == 1


def test_observer_disconnect_preserves_pending_owner_input():
    interaction, controller = interaction_with_owner()
    interaction.submit(1, "gripper", 75)
    # Membership changes alone must not invalidate another client's input.
    interaction.process({1})
    assert controller.calls == [("gripper", 75)]
    assert interaction.owner == 1


def test_enable_keeps_preceding_speed_setting_but_discards_motion_target():
    interaction, controller = interaction_with_owner()
    interaction.submit(1, "speed", 5)
    interaction.submit(1, "gripper", 90)
    interaction.submit(1, "enable")
    interaction.process({1})
    assert controller.calls == [("enable", 5.0)]
    assert controller.enabled


@pytest.mark.parametrize("kind,value", [("stop", None), ("enable", None), ("mode", MODES[0])])
def test_transition_cancels_active_drag_and_ignores_its_late_tail(kind, value):
    interaction, controller = interaction_with_owner()
    interaction.submit(1, "drag", drag_event("start"))
    interaction.process({1})
    controller.calls.clear()
    interaction.submit(1, kind, value)
    interaction.submit(1, "drag", drag_event("update", (0.4, 0.0, 0.2)))
    interaction.submit(1, "drag", drag_event("end", (0.4, 0.0, 0.2)))
    interaction.process({1})
    assert not any(call[0] == "target" for call in controller.calls)
    controller.calls.clear()
    # Only a new gesture can generate another motion target.
    interaction.submit(1, "drag", drag_event("start"))
    interaction.submit(1, "drag", drag_event("update", (0.31, 0.0, 0.2)))
    interaction.submit(1, "drag", drag_event("end", (0.31, 0.0, 0.2)))
    interaction.process({1})
    assert controller.calls[-1] == ("target", (0.31, 0.0, 0.2))


def test_drag_update_coalescing_preserves_both_lifecycle_boundaries():
    interaction, controller = interaction_with_owner()
    interaction.submit(1, "drag", drag_event("start"))
    for index in range(100):
        interaction.submit(1, "drag", drag_event("update", (0.3 + index / 1000, 0.0, 0.2)))
    interaction.submit(1, "drag", drag_event("end", (0.4, 0.0, 0.2)))
    interaction.process({1})
    assert [call[1] for call in controller.calls] == [
        (0.3, 0.0, 0.2), (0.399, 0.0, 0.2), (0.4, 0.0, 0.2),
    ]
    controller.calls.clear()
    interaction.submit(1, "drag", drag_event("update", (0.45, 0.0, 0.2)))
    interaction.process({1})
    assert not controller.calls


def test_stop_from_a_departing_observer_is_still_honored_without_claiming_control():
    interaction, controller = interaction_with_owner()
    controller.enabled = True
    interaction.submit(2, "stop")
    interaction.process({1})
    assert not controller.enabled
    assert [call[0] for call in controller.calls] == ["pause"]
    assert interaction.owner == 1


def test_owner_disconnect_holds_and_cancels_the_old_gesture():
    interaction, controller = interaction_with_owner()
    controller.enabled = True
    interaction.submit(1, "drag", drag_event("start"))
    interaction.process({1, 2})
    controller.calls.clear()
    interaction.submit(1, "drag", drag_event("update", (0.4, 0.0, 0.2)))
    interaction.process({2})
    assert not controller.enabled
    assert interaction.owner is None
    assert [call[0] for call in controller.calls] == ["pause"]
    controller.calls.clear()
    # Reusing the old client ID must not revive its cancelled drag.
    interaction.submit(1, "drag", drag_event("end", (0.4, 0.0, 0.2)))
    interaction.process({1, 2})
    assert not controller.calls


def test_owner_disconnect_keeps_server_state_stopped_if_hold_fails():
    interaction, controller = interaction_with_owner()
    controller.enabled = True

    def failed_hold(reason):
        raise OSError("USB read failed")

    controller.pause = failed_hold
    assert interaction.process({2})
    assert not controller.enabled
    assert interaction.owner is None
    assert "hold failed" in controller.status


def test_processing_is_bounded_and_stop_can_cancel_remaining_gestures():
    interaction, controller = interaction_with_owner()
    for index in range(300):
        interaction.submit(1, "drag", drag_event("start", node=f"/nose{index}"))
    interaction.process({1})
    assert len(controller.calls) == 64
    controller.calls.clear()
    interaction.submit(1, "stop")
    interaction.process({1})
    assert [call[0] for call in controller.calls] == ["pause"]


def test_programmatic_ui_refresh_does_not_claim_control_or_issue_commands():
    controller = FakeController()
    interaction = Interaction(controller)
    interaction.submit(None, "joint", (0, 20))
    assert not interaction.process({1})
    assert interaction.owner is None
    assert not controller.calls


def keyboard_interaction():
    clock = Clock()
    controller = Controller(Kinematics())
    interaction = Interaction(controller, clock)
    return interaction, controller, clock


def keyboard_message(interaction, direction=(0, 1, 0), sequence=0, epoch=None):
    return [*direction, sequence, interaction.keyboard_epoch if epoch is None else epoch]


def remote_message(interaction, rates=(0, 1, 0, 0, 0), sequence=0, epoch=None):
    return [*rates, sequence, interaction.remote_epoch if epoch is None else epoch]


def test_remote_input_claims_control_only_for_valid_fresh_motion():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "remote", remote_message(interaction))
    interaction.process({1, 2}, dt=1 / 30)
    assert interaction.owner == 1
    assert interaction.remote_active
    target = controller.target_angles.copy()
    interaction.submit(2, "remote", remote_message(interaction, (0, -1, 0, 0, 0), 1))
    interaction.process({1, 2})
    assert controller.target_angles == pytest.approx(target)


@pytest.mark.parametrize("payload", [
    "bad json", "[0,0,0,0,0,0,0]", "[0,2,0,0,0,0,0]", "[0,1,0,0,0,-1,0]",
    "[0,1,0,0,0,0,-1]", "[0,1,0,0,0,0.5,0]",
])
def test_invalid_or_neutral_remote_input_cannot_claim_control(payload):
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "remote", payload)
    interaction.process({1, 2}, dt=1 / 30)
    assert interaction.owner is None
    assert not interaction.remote_active


def test_queued_remote_arrival_does_not_receive_a_new_lease_when_processed():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "remote", remote_message(interaction))
    clock.advance(0.4)
    interaction.process({1, 2}, dt=1 / 30)
    assert interaction.owner is None
    assert not interaction.remote_active


@pytest.mark.parametrize("kind,value", [
    ("stop", None), ("mode", MODES[1]), ("sync", None),
    ("joint", (0, 10)), ("gripper", 75), ("layout", None),
])
def test_other_motion_and_transitions_invalidate_held_remote_messages(kind, value):
    interaction, controller, clock = keyboard_interaction()
    epoch = interaction.remote_epoch
    interaction.submit(1, "remote", remote_message(interaction))
    interaction.process({1}, dt=1 / 30)
    interaction.submit(1, kind, value)
    interaction.process({1})
    assert not interaction.remote_active
    assert interaction.remote_epoch > epoch
    expected = controller.target_angles.copy()
    interaction.submit(1, "remote", remote_message(interaction, sequence=1, epoch=epoch))
    interaction.process({1}, dt=1 / 30)
    assert not interaction.remote_active
    assert controller.target_angles == pytest.approx(expected)


def test_new_remote_gesture_cancels_keyboard_without_continuous_epoch_churn():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "keyboard", keyboard_message(interaction))
    interaction.process({1}, dt=1 / 30)
    keyboard_epoch = interaction.keyboard_epoch
    interaction.submit(1, "remote", remote_message(interaction))
    interaction.process({1}, dt=1 / 30)
    assert interaction.remote_active
    assert not interaction.keyboard_active
    assert interaction.keyboard_epoch == keyboard_epoch + 1
    interaction.submit(1, "remote", remote_message(interaction, sequence=1))
    interaction.process({1}, dt=1 / 30)
    assert interaction.keyboard_epoch == keyboard_epoch + 1
    interaction.submit(1, "keyboard", keyboard_message(interaction, sequence=1, epoch=keyboard_epoch))
    interaction.process({1}, dt=1 / 30)
    assert interaction.remote_active
    assert not interaction.keyboard_active


def test_keyboard_can_replace_remote_but_a_neutral_keyboard_tail_cannot():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "remote", remote_message(interaction))
    interaction.process({1}, dt=1 / 30)
    interaction.submit(1, "keyboard", keyboard_message(interaction, (0, 0, 0)))
    interaction.process({1}, dt=1 / 30)
    assert interaction.remote_active
    epoch = interaction.remote_epoch
    interaction.submit(1, "keyboard", keyboard_message(interaction, sequence=1))
    interaction.process({1}, dt=1 / 30)
    assert interaction.keyboard_active
    assert not interaction.remote_active
    assert interaction.remote_epoch == epoch + 1


def test_lost_operator_cancels_remote_and_handoff_discards_its_neutral_tail():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "heartbeat", "[1,true]")
    interaction.submit(1, "remote", remote_message(interaction))
    interaction.process({1, 2}, dt=1 / 30)
    epoch = interaction.remote_epoch
    interaction.submit(1, "heartbeat", "[2,false]")
    interaction.process({1, 2})
    assert not interaction.remote_active
    assert interaction.remote_epoch > epoch
    interaction.submit(1, "yield")
    interaction.process({1, 2})
    assert interaction.owner is None
    interaction.submit(1, "remote", remote_message(interaction, (0, 0, 0, 0, 0), 1))
    interaction.process({1, 2})
    assert interaction.owner is None


@pytest.mark.parametrize("owned", [False, True])
def test_observer_layout_change_never_claims_or_interrupts_control(owned):
    interaction, controller, clock = keyboard_interaction()
    if owned:
        interaction.submit(1, "remote", remote_message(interaction))
        interaction.process({1, 2}, dt=1 / 30)
    epoch = interaction.remote_epoch
    status = controller.status
    interaction.submit(2, "layout")
    interaction.process({1, 2})
    assert interaction.owner == (1 if owned else None)
    assert interaction.remote_epoch == epoch
    assert controller.status == status
    assert interaction.remote_active is owned


def test_owner_layout_change_pauses_hardware_and_discards_queued_input():
    hardware = FakeHardware()
    controller = Controller(Kinematics(), hardware)
    interaction = Interaction(controller, Clock())
    interaction.submit(1, "connect")
    interaction.process({1, 2})
    interaction.submit(1, "enable")
    interaction.process({1, 2})
    interaction.submit(1, "remote", remote_message(interaction))
    interaction.process({1, 2}, dt=1 / 30)
    epoch = interaction.remote_epoch
    interaction.submit(1, "layout")
    interaction.submit(1, "enable")
    interaction.submit(1, "remote", remote_message(interaction, sequence=1))
    interaction.process({1, 2}, dt=1 / 30)
    assert interaction.owner == 1
    assert not controller.enabled
    assert hardware.armed
    assert not interaction.remote_active
    assert interaction.remote_epoch > epoch
    assert controller.status == "Control layout changed; movement paused"


def test_keyboard_moves_in_preview_and_respects_operator_ownership():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "keyboard", keyboard_message(interaction))
    interaction.process({1, 2}, dt=1 / 30)
    assert interaction.keyboard_active
    assert interaction.owner == 1
    target = controller.target_angles.copy()
    interaction.submit(2, "keyboard", keyboard_message(interaction, (0, -1, 0), 1))
    interaction.process({1, 2})
    assert controller.target_angles == pytest.approx(target)


@pytest.mark.parametrize("kind,value", [
    ("stop", None), ("mode", MODES[1]), ("sync", None),
    ("joint", (0, 10)), ("gripper", 75),
])
def test_keyboard_old_held_messages_cannot_resume_after_another_command(kind, value):
    interaction, controller, clock = keyboard_interaction()
    epoch = interaction.keyboard_epoch
    interaction.submit(1, "keyboard", keyboard_message(interaction))
    interaction.process({1}, dt=1 / 30)
    interaction.submit(1, kind, value)
    interaction.process({1})
    assert not interaction.keyboard_active
    assert interaction.keyboard_epoch > epoch
    expected = controller.target_angles.copy()
    interaction.submit(1, "keyboard", keyboard_message(interaction, sequence=1, epoch=epoch))
    interaction.process({1}, dt=1 / 30)
    assert not interaction.keyboard_active
    assert controller.target_angles == pytest.approx(expected)
    interaction.submit(1, "keyboard", keyboard_message(interaction, sequence=2))
    interaction.process({1}, dt=1 / 30)
    assert interaction.keyboard_active


def test_keyboard_timeout_stops_once_and_old_packets_do_not_refresh_the_lease():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "keyboard", keyboard_message(interaction))
    interaction.process({1}, dt=1 / 30)
    controller.tick(1 / 30)
    clock.advance(0.3)
    assert interaction.process({1}, dt=1 / 30)
    assert not interaction.keyboard_active
    frozen = controller.angles.copy()
    assert not interaction.process({1}, dt=1 / 30)
    controller.tick(1 / 30)
    assert controller.angles == pytest.approx(frozen)


def test_keyboard_arrival_time_is_not_reset_by_a_delayed_control_loop():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "keyboard", keyboard_message(interaction))
    clock.advance(0.4)
    interaction.process({1}, dt=1 / 30)
    assert not interaction.keyboard_active


def test_owner_disconnect_cancels_keyboard_lease():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "keyboard", keyboard_message(interaction))
    interaction.process({1}, dt=1 / 30)
    epoch = interaction.keyboard_epoch
    interaction.process(set(), dt=1 / 30)
    assert not interaction.keyboard_active
    assert interaction.keyboard_epoch > epoch
    assert interaction.owner is None


def test_invalid_keyboard_payload_pauses_and_invalidates_held_input():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "keyboard", keyboard_message(interaction))
    interaction.process({1}, dt=1 / 30)
    interaction.submit(1, "keyboard", "[0,2,0,1,0]")
    interaction.process({1}, dt=1 / 30)
    assert not interaction.keyboard_active
    assert "direction" in controller.status


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


def test_session_heartbeats_do_not_claim_control_or_move_the_arm():
    controller = FakeController()
    interaction = Interaction(controller, Clock())
    interaction.submit(2, "heartbeat", "[1,true]")
    assert not interaction.process({1, 2})
    assert interaction.owner is None
    assert not controller.calls


def test_operator_heartbeat_expiry_holds_once_and_discards_pending_enable():
    interaction, controller, clock = foreground_interaction()
    clock.advance(0.51)
    interaction.submit(1, "enable")
    interaction.submit(1, "gripper", 75)
    assert interaction.process({1, 2})
    assert not controller.enabled
    assert interaction.owner == 1
    assert [call[0] for call in controller.calls] == ["pause"]
    assert "heartbeat lost" in controller.status
    controller.calls.clear()
    assert not interaction.process({1, 2})
    assert not controller.calls


def test_operator_has_a_short_initial_grace_but_commands_cannot_extend_it():
    clock = Clock()
    controller = FakeController()
    interaction = Interaction(controller, clock)
    interaction.submit(1, "speed", 20)
    interaction.process({1})
    clock.advance(0.3)
    interaction.submit(1, "speed", 10)
    interaction.process({1})
    assert not any(call[0] == "pause" for call in controller.calls)
    clock.advance(0.21)
    interaction.submit(1, "enable")
    interaction.process({1})
    assert not controller.enabled
    assert controller.calls[-1][0] == "pause"


def test_foreground_recovery_never_replays_enable_and_requires_a_new_command():
    interaction, controller, clock = foreground_interaction()
    clock.advance(0.51)
    interaction.process({1})
    interaction.submit(1, "enable")
    interaction.submit(1, "heartbeat", "[2,true]")
    interaction.process({1})
    assert not controller.enabled
    assert not any(call[0] == "enable" for call in controller.calls)
    interaction.submit(1, "enable")
    interaction.process({1})
    assert controller.enabled
    assert controller.calls[-1][0] == "enable"


def test_disconnect_button_brings_a_held_arm_back_first():
    interaction, controller = interaction_with_owner()
    controller.connected = True
    controller.can_return_to_rest = True
    interaction.submit(1, "disconnect")
    interaction.process({1, 2})
    assert ("start_return",) in controller.calls
    assert ("disconnect",) not in controller.calls


def test_disconnect_button_frees_an_arm_without_torque_at_once():
    interaction, controller = interaction_with_owner()
    controller.connected = True
    interaction.submit(1, "disconnect")
    interaction.process({1, 2})
    assert ("disconnect",) in controller.calls
    assert ("start_return",) not in controller.calls


def nobody_connected_interaction():
    clock = Clock()
    controller = FakeController()
    controller.connected = True
    controller.can_return_to_rest = True
    return Interaction(controller, clock), controller, clock


def test_arm_returns_after_nobody_has_been_connected_for_a_while():
    interaction, controller, clock = nobody_connected_interaction()
    interaction.process(set())
    clock.advance(4.9)
    interaction.process(set())
    assert ("start_return",) not in controller.calls
    clock.advance(0.2)
    interaction.process(set())
    assert controller.calls == [("start_return",)]


def test_connecting_again_in_time_keeps_the_arm_where_it_is():
    interaction, controller, clock = nobody_connected_interaction()
    interaction.process(set())
    clock.advance(4.0)
    interaction.process({2})
    clock.advance(4.0)
    interaction.process(set())
    clock.advance(4.0)
    interaction.process(set())
    assert controller.calls == []


def test_nobody_connected_frees_the_port_of_an_arm_without_torque():
    interaction, controller, clock = nobody_connected_interaction()
    controller.can_return_to_rest = False
    interaction.process(set())
    clock.advance(5.1)
    interaction.process(set())
    assert controller.calls == [("disconnect",)]


def test_stopped_return_is_not_retried_while_nobody_is_there():
    interaction, controller, clock = nobody_connected_interaction()
    controller.return_stopped = True
    interaction.process(set())
    clock.advance(60.0)
    interaction.process(set())
    assert controller.calls == []


def test_stop_cancels_a_running_return():
    interaction, controller = interaction_with_owner()
    controller.returning = True
    interaction.submit(2, "stop")
    interaction.process({1, 2})
    assert not controller.returning


def test_click_that_brings_the_page_back_is_not_lost():
    interaction, controller, clock = foreground_interaction()
    controller.enabled = False
    interaction.submit(1, "heartbeat", "[2,false]")
    interaction.process({1})
    interaction.submit(1, "heartbeat", "[3,true]")
    interaction.submit(1, "enable")
    interaction.process({1})
    assert controller.enabled
    assert controller.calls[-1][0] == "enable"


def test_background_transition_pauses_even_if_foreground_returns_in_the_same_batch():
    interaction, controller, clock = foreground_interaction()
    interaction.submit(1, "heartbeat", "[2,false]")
    interaction.submit(1, "enable")
    interaction.submit(1, "heartbeat", "[3,true]")
    interaction.process({1})
    assert not controller.enabled
    assert [call[0] for call in controller.calls] == ["pause"]


def test_returning_heartbeat_reveals_a_gap_hidden_by_a_stalled_control_loop():
    interaction, controller, clock = foreground_interaction()
    clock.advance(0.3)
    interaction.submit(1, "enable")
    clock.advance(0.3)
    interaction.submit(1, "heartbeat", "[2,true]")
    interaction.process({1})
    assert not controller.enabled
    assert [call[0] for call in controller.calls] == ["pause"]


def test_viewer_heartbeats_cannot_keep_the_operator_alive_or_steal_control():
    interaction, controller, clock = foreground_interaction()
    clock.advance(0.51)
    interaction.submit(2, "heartbeat", "[1,true]")
    interaction.process({1, 2})
    assert not controller.enabled
    assert interaction.owner == 1


def test_stale_arrival_and_duplicate_heartbeat_sequence_do_not_renew_the_lease():
    interaction, controller, clock = foreground_interaction()
    clock.advance(0.1)
    interaction.submit(1, "heartbeat", "[2,true]")
    clock.advance(0.51)
    interaction.process({1})
    assert not controller.enabled
    assert interaction._heartbeats[1][0] == 1
    interaction.submit(1, "heartbeat", "[1,true]")
    interaction.process({1})
    assert interaction._heartbeats[1][0] == 1


@pytest.mark.parametrize("payload", ["not json", "[1]", "[2,1]", "[true,true]", "[2.5,true]"])
def test_invalid_session_heartbeats_cannot_refresh_an_operator(payload):
    interaction, controller, clock = foreground_interaction()
    clock.advance(0.51)
    interaction.submit(1, "heartbeat", payload)
    interaction.process({1})
    assert not controller.enabled


def test_inactive_viewer_cannot_become_operator_until_it_returns_to_foreground():
    interaction, controller, clock = foreground_interaction()
    interaction.process({2})
    interaction.submit(2, "heartbeat", "[1,false]")
    interaction.submit(2, "enable")
    interaction.process({2})
    assert interaction.owner is None
    assert not controller.enabled
    interaction.submit(2, "heartbeat", "[2,true]")
    interaction.submit(2, "enable")
    interaction.process({2})
    assert interaction.owner == 2
    assert controller.enabled


def test_disconnected_clients_and_their_heartbeat_state_are_removed():
    interaction, controller, clock = foreground_interaction()
    interaction.submit(2, "heartbeat", "[1,true]")
    interaction.process({1, 2})
    interaction.process(set())
    assert interaction.owner is None
    assert interaction._owner_since is None
    assert not interaction._heartbeats


def test_owner_can_yield_while_the_connected_arm_keeps_holding_its_pose():
    hardware = FakeHardware()
    controller = Controller(Kinematics(), hardware)
    interaction = Interaction(controller, Clock())
    interaction.submit(1, "connect")
    interaction.process({1, 2})
    interaction.submit(1, "enable")
    interaction.process({1, 2})
    epoch = interaction.keyboard_epoch
    interaction.submit(1, "yield")
    interaction.process({1, 2})
    assert hardware.connected
    assert hardware.armed
    assert not controller.enabled
    assert interaction.owner is None
    assert interaction._owner_since is None
    assert not interaction.keyboard_active
    assert interaction.keyboard_epoch > epoch
    assert controller.status == "Control released; movement paused"
    interaction.submit(2, "heartbeat", "[1,true]")
    interaction.process({1, 2})
    assert interaction.owner is None
    assert not controller.enabled
    interaction.submit(2, "enable")
    interaction.process({1, 2})
    assert interaction.owner == 2
    assert controller.enabled
    assert hardware.armed


def test_viewing_client_cannot_force_a_handoff_or_discard_the_owner_input():
    interaction, controller, clock = foreground_interaction()
    interaction.submit(1, "gripper", 75)
    interaction.submit(2, "yield")
    interaction.process({1, 2})
    assert interaction.owner == 1
    assert controller.enabled
    assert controller.calls == [("gripper", 75)]


def test_yield_discards_queued_old_owner_motion_and_enable_requests():
    interaction, controller, clock = foreground_interaction()
    interaction.submit(1, "yield")
    interaction.submit(1, "enable")
    interaction.submit(1, "gripper", 75)
    interaction.submit(1, "drag", drag_event("start"))
    interaction.process({1, 2})
    assert interaction.owner is None
    assert not controller.enabled
    assert [call[0] for call in controller.calls] == ["pause"]


def test_yield_without_an_owner_does_not_claim_control():
    controller = FakeController()
    interaction = Interaction(controller, Clock())
    interaction.submit(1, "yield")
    interaction.process({1, 2})
    assert interaction.owner is None
    assert not controller.calls


def test_late_keyboard_neutral_after_handoff_does_not_reclaim_control():
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "keyboard", keyboard_message(interaction))
    interaction.process({1, 2}, dt=1 / 30)
    interaction.submit(1, "yield")
    interaction.process({1, 2})
    assert interaction.owner is None
    interaction.submit(1, "keyboard", keyboard_message(interaction, (0, 0, 0), sequence=1))
    interaction.process({1, 2}, dt=1 / 30)
    assert interaction.owner is None
    assert not interaction.keyboard_active
    interaction.submit(2, "speed", 10)
    interaction.process({1, 2})
    assert interaction.owner == 2


@pytest.mark.parametrize("payload", [
    "bad json", "[0,0,0,1,0]", "[0,2,0,1,0]", "[0,1,0,1,-1]", "[0,1,0,1.5,0]",
])
def test_invalid_or_neutral_keyboard_input_cannot_claim_an_unowned_arm(payload):
    interaction, controller, clock = keyboard_interaction()
    interaction.submit(1, "keyboard", payload)
    interaction.process({1, 2}, dt=1 / 30)
    assert interaction.owner is None
    assert not interaction.keyboard_active


def test_stale_epoch_or_stale_arrival_cannot_claim_after_handoff():
    interaction, controller, clock = keyboard_interaction()
    old_epoch = interaction.keyboard_epoch
    interaction.submit(1, "speed", 20)
    interaction.process({1, 2})
    interaction.submit(1, "yield")
    interaction.process({1, 2})
    interaction.submit(1, "keyboard", keyboard_message(interaction, sequence=1, epoch=old_epoch))
    interaction.process({1, 2}, dt=1 / 30)
    assert interaction.owner is None
    interaction.submit(1, "keyboard", keyboard_message(interaction, sequence=2))
    clock.advance(0.3)
    interaction.process({1, 2}, dt=1 / 30)
    assert interaction.owner is None
    assert not interaction.keyboard_active


@pytest.mark.parametrize("foreground_again", [False, True])
def test_background_or_recovering_owner_can_yield_without_replaying_motion(foreground_again):
    interaction, controller, clock = foreground_interaction()
    controller.connected = True
    interaction.submit(1, "heartbeat", "[2,false]")
    interaction.process({1, 2})
    assert not controller.enabled
    controller.calls.clear()
    interaction.submit(1, "enable")
    interaction.submit(1, "gripper", 75)
    if foreground_again:
        interaction.submit(1, "heartbeat", "[3,true]")
    interaction.submit(1, "yield")
    interaction.process({1, 2})
    assert interaction.owner is None
    assert controller.connected
    assert not controller.enabled
    assert [call[0] for call in controller.calls] == ["pause"]
    interaction.submit(2, "heartbeat", "[1,true]")
    interaction.submit(2, "enable")
    interaction.process({1, 2})
    assert interaction.owner == 2
    assert controller.enabled
