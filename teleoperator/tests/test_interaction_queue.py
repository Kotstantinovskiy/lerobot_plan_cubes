"""Browser event ordering without a viewer or motors."""

import pytest

from fakes import FakeController, drag_event, interaction_with_owner
from interaction import MODES, Interaction


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
        (0.3, 0.0, 0.2),
        (0.399, 0.0, 0.2),
        (0.4, 0.0, 0.2),
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
