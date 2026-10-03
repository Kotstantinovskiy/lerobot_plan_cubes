"""Bringing the arm back when the operator disconnects or nobody is connected."""

from fakes import Clock, FakeController, interaction_with_owner
from interaction import Interaction


def nobody_connected_interaction():
    clock = Clock()
    controller = FakeController()
    controller.connected = True
    controller.can_return_to_rest = True
    return Interaction(controller, clock), controller, clock


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
