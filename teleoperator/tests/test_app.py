"""The command line and one step of the control loop, without a server or motors."""

import sys
from types import SimpleNamespace

import pytest

from app import _control_step, parse_args


class FailingController:
    enabled = True
    faulted = False

    def tick(self, dt):
        self.enabled = False
        raise RuntimeError("lost the bus")


class SteadyController:
    enabled = True
    faulted = False

    def tick(self, dt):
        pass


class RecordingInteraction:
    def __init__(self, *, has_changes=False):
        self.has_changes = has_changes
        self.pauses = []
        self.drag_clears = 0
        self.drag = SimpleNamespace(clear=self._clear_drag)

    def _clear_drag(self):
        self.drag_clears += 1

    def process(self, clients, dt):
        return self.has_changes

    def pause(self, reason):
        self.pauses.append(reason)


class RecordingPanel:
    def __init__(self):
        self.control_updates = 0

    def update_controls(self):
        self.control_updates += 1


def parse(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["app.py", *arguments])
    return parse_args()


def test_failed_motor_step_pauses_and_clears_the_drag_with_the_reason():
    interaction = RecordingInteraction()
    panel = RecordingPanel()
    _control_step(FailingController(), interaction, panel, set(), 0.03)
    assert interaction.pauses == ["Stopped: lost the bus"]
    assert interaction.drag_clears == 1
    assert panel.control_updates == 1


def test_processed_input_refreshes_the_controls_without_touching_the_drag():
    interaction = RecordingInteraction(has_changes=True)
    panel = RecordingPanel()
    _control_step(SteadyController(), interaction, panel, set(), 0.03)
    assert panel.control_updates == 1
    assert interaction.drag_clears == 0


def test_quiet_step_leaves_the_controls_alone():
    panel = RecordingPanel()
    _control_step(SteadyController(), RecordingInteraction(), panel, set(), 0.03)
    assert panel.control_updates == 0


def test_defaults_start_in_preview_on_the_usual_port(monkeypatch):
    args = parse(monkeypatch)
    assert (args.hardware, args.port, args.speed, args.min_height_mm) == (False, 8080, 20.0, 15.0)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--port", "0"),
        ("--port", "65536"),
        ("--address", "192.168.1.5"),
        ("--speed", "0.5"),
        ("--speed", "46"),
        ("--speed", "nan"),
        ("--min-height-mm", "-1"),
        ("--min-height-mm", "201"),
        ("--seconds", "0"),
        ("--seconds", "inf"),
    ],
)
def test_values_out_of_range_are_refused(monkeypatch, arguments):
    with pytest.raises(SystemExit):
        parse(monkeypatch, *arguments)
