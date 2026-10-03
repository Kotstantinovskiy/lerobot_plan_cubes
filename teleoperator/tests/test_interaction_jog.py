"""Keyboard and thumb-stick input as seen by the operator's session."""

import pytest

from fakes import keyboard_interaction, keyboard_message, remote_message
from interaction import MODES


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


@pytest.mark.parametrize(
    "payload",
    [
        "bad json",
        "[0,0,0,0,0,0,0]",
        "[0,2,0,0,0,0,0]",
        "[0,1,0,0,0,-1,0]",
        "[0,1,0,0,0,0,-1]",
        "[0,1,0,0,0,0.5,0]",
    ],
)
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


@pytest.mark.parametrize(
    "kind,value",
    [
        ("stop", None),
        ("mode", MODES[1]),
        ("sync", None),
        ("joint", (0, 10)),
        ("gripper", 75),
        ("layout", None),
    ],
)
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


@pytest.mark.parametrize(
    "kind,value",
    [
        ("stop", None),
        ("mode", MODES[1]),
        ("sync", None),
        ("joint", (0, 10)),
        ("gripper", 75),
    ],
)
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
