"""Who controls the arm: layout changes, handing over and claiming it again."""

import pytest

from controller import Controller
from fakes import (
    Clock,
    FakeController,
    FakeHardware,
    drag_event,
    foreground_interaction,
    keyboard_interaction,
    keyboard_message,
    remote_message,
)
from interaction import Interaction
from kinematics import Kinematics


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


@pytest.mark.parametrize(
    "payload",
    [
        "bad json",
        "[0,0,0,1,0]",
        "[0,2,0,1,0]",
        "[0,1,0,1,-1]",
        "[0,1,0,1.5,0]",
    ],
)
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
