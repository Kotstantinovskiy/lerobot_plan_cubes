"""Foreground heartbeats decide whether the operator is still there."""

import pytest

from fakes import Clock, FakeController, foreground_interaction
from interaction import Interaction


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
