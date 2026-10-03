"""The panel only queues input and shows state: a real Viser server, without a browser or motors."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import viser
import yourdfpy

from control_panel import ControlPanel, _parse_remote_message
from controller import Controller
from interaction import Interaction
from kinematics import MODEL_PATH, Kinematics

MIN_HEIGHT_MM = 15.0


class RecordingInteraction:
    def __init__(self, *, direct_mode=True):
        self.direct_mode = direct_mode
        self.submitted = []

    def submit(self, client_id, kind, value=None):
        self.submitted.append((client_id, kind, value))


@pytest.fixture
def server():
    viser_server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    yield viser_server
    viser_server.stop()


@pytest.fixture
def urdf():
    return yourdfpy.URDF.load(str(MODEL_PATH), load_collision_meshes=False)


def recording_panel(server, urdf, *, direct_mode=True):
    interaction = RecordingInteraction(direct_mode=direct_mode)
    controller = Controller(Kinematics())
    panel = ControlPanel(server, urdf, controller, interaction, min_height_mm=MIN_HEIGHT_MM)
    return panel, interaction


def value_event(value):
    return SimpleNamespace(client_id=1, target=SimpleNamespace(value=value))


def drag_event():
    return SimpleNamespace(
        client_id=1,
        target=SimpleNamespace(name="/nose"),
        phase="start",
        start_position=(0.0, 0.0, 0.0),
        end_position=(0.0, 0.0, 0.1),
    )


def test_remote_action_becomes_a_command_of_that_kind():
    assert _parse_remote_message('{"action": "stop", "sequence": 3}') == ("stop", None)


def test_remote_rates_go_on_for_the_jog_to_check():
    assert _parse_remote_message("[0, 1, 0, 0, 0, 7, 0]") == ("remote", [0, 1, 0, 0, 0, 7, 0])


@pytest.mark.parametrize(
    "message",
    [
        "not json",
        '{"action": "release", "sequence": 3}',
        '{"action": "stop"}',
        '{"action": "stop", "sequence": 3, "extra": 1}',
        '{"action": 1, "sequence": 3}',
        "{}",
    ],
)
def test_other_remote_messages_are_ignored(message):
    assert _parse_remote_message(message) is None


def test_panel_shows_preview_and_the_tool_position_after_a_control_step(server, urdf):
    controller = Controller(Kinematics())
    panel = ControlPanel(server, urdf, controller, Interaction(controller), min_height_mm=MIN_HEIGHT_MM)
    panel.update_controls()
    tool_position = panel.show_arm()
    panel.update_status(tool_position)
    np.testing.assert_allclose(tool_position, controller.model.forward(controller.angles)[:3, 3])
    assert "Предпросмотр" in panel._status.content
    assert "Минимальная высота 15 мм" in panel._telemetry.content


def test_remote_message_is_queued_with_its_parsed_value(server, urdf):
    panel, interaction = recording_panel(server, urdf)
    asyncio.run(panel._on_remote(value_event("[0, 1, 0, 0, 0, 7, 0]")))
    assert interaction.submitted == [(1, "remote", [0, 1, 0, 0, 0, 7, 0])]


def test_ignored_remote_message_queues_nothing(server, urdf):
    panel, interaction = recording_panel(server, urdf)
    asyncio.run(panel._on_remote(value_event("not json")))
    assert interaction.submitted == []


def test_round_control_queues_the_wrist_roll_before_the_gripper(server, urdf):
    panel, interaction = recording_panel(server, urdf)
    asyncio.run(panel._on_gripper_control(value_event((10.0, 40.0))))
    assert interaction.submitted == [(1, "roll", 10.0), (1, "gripper", 40.0)]


def test_dragging_the_tool_is_queued_only_while_it_is_the_active_mode(server, urdf):
    panel, interaction = recording_panel(server, urdf, direct_mode=False)
    asyncio.run(panel._on_tool_drag(drag_event()))
    assert interaction.submitted == []


def test_dragging_the_tool_queues_the_grab_and_the_pointer(server, urdf):
    panel, interaction = recording_panel(server, urdf)
    asyncio.run(panel._on_tool_drag(drag_event()))
    [(client_id, kind, (name, phase, grab, pointer))] = interaction.submitted
    assert (client_id, kind, name, phase) == (1, "drag", "/nose", "start")
    np.testing.assert_allclose(grab, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(pointer, (0.0, 0.0, 0.1))
