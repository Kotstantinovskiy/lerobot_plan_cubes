"""Direct mesh dragging preserves the original tool-to-pointer offset."""

import numpy as np
import pytest

from drag import DragTarget


def test_off_center_grab_does_not_jump_to_the_mouse_position():
    drag = DragTarget()
    tool = [0.4, 0.0, 0.2]
    grab = [0.42, 0.01, 0.19]
    start = drag.update(1, "/nose", "start", grab, [0.45, 0.01, 0.19], tool)
    np.testing.assert_allclose(start, tool)
    target = drag.update(1, "/nose", "update", grab, grab, tool)
    np.testing.assert_allclose(target, tool)


def test_offset_stays_fixed_when_the_rendered_arm_moves_during_drag():
    drag = DragTarget()
    drag.update(1, "/nose", "start", [0.42, 0.01, 0.19], [0.42, 0.01, 0.19], [0.4, 0, 0.2])
    first = drag.update(1, "/nose", "update", [0.44, 0.02, 0.2], [0.47, 0.02, 0.22], [0.43, 0.01, 0.21])
    final = drag.update(1, "/nose", "end", [0.48, 0.04, 0.23], [0.47, 0.02, 0.22], [0.46, 0.03, 0.24])
    np.testing.assert_allclose(first, [0.45, 0.01, 0.23])
    np.testing.assert_allclose(final, first)
    assert drag.update(1, "/nose", "update", [0, 0, 0], [0, 0, 0], [0, 0, 0]) is None


def test_grabs_are_independent_between_clients_and_nodes():
    drag = DragTarget()
    drag.update(1, "/nose", "start", [0, 0, 0], [0, 0, 0], [0.1, 0, 0])
    drag.update(2, "/nose", "start", [0, 0, 0], [0, 0, 0], [0, 0.2, 0])
    drag.update(1, "/other", "start", [0, 0, 0], [0, 0, 0], [0, 0, 0.3])
    for client, node, expected in [
        (1, "/nose", [0.1, 0, 0]),
        (2, "/nose", [0, 0.2, 0]),
        (1, "/other", [0, 0, 0.3]),
    ]:
        result = drag.update(client, node, "update", [0, 0, 0], [0, 0, 0], [9, 9, 9])
        np.testing.assert_allclose(result, expected)
    drag.clear()
    assert drag.update(2, "/nose", "end", [0, 0, 0], [0, 0, 0], [0, 0, 0]) is None


@pytest.mark.parametrize("phase", ["update", "end"])
def test_events_without_a_start_are_ignored(phase):
    drag = DragTarget()
    assert drag.update(1, "/nose", phase, [0, 0, 0], [0, 0, 0], [0, 0, 0]) is None


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("bad_position", [[0, 0], [0, 0, float("nan")], [float("inf"), 0, 0], "invalid"])
def test_invalid_coordinates_are_rejected(index, bad_position):
    drag = DragTarget()
    positions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    positions[index] = bad_position
    with pytest.raises(ValueError, match="finite 3D"):
        drag.update(1, "/nose", "start", *positions)


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="phase"):
        DragTarget().update(1, "/nose", "move", [0, 0, 0], [0, 0, 0], [0, 0, 0])
