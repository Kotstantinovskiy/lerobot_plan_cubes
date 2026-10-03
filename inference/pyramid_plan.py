"""The cube pyramid as a plan: what the top camera shows, which steps are allowed, and the plain loop planner.

The pyramid is two cubes side by side and one resting on both. A slot is one of those three places; it is recognised on
the top camera by a box in the pixels of the 1280x720 frame, as in cube_boxes.py: (x0, y0, x1, y1), inclusive-exclusive.
The table can hold several cubes of one colour, so a step names one exact cube by its id.
Nothing here moves the arm or uses the network. llm_planner.py chooses steps with a language model instead of a loop.
"""

from dataclasses import dataclass
import math

import cube_boxes

PICK_AND_PLACE, FINISH = "pick_and_place", "finish"
NO_CHOICE = "none"  # the cube and the slot of a finish step
BASE_SLOTS = ("base_left", "base_right")  # left and right as seen on the top camera
TOP_SLOT = "top"
SLOTS = (*BASE_SLOTS, TOP_SLOT)


@dataclass(frozen=True)
class Cube:
    """A free cube. `id` names it for this one look at the table; `box` is where the arm's target box goes."""

    id: str
    color: str
    box: list[int]


@dataclass(frozen=True)
class TableState:
    """What the colour detector sees: free cubes, the colour in each slot (None if empty), colours it cannot find."""

    free_cubes: list[Cube]
    slots: dict[str, str | None]
    not_seen: list[str]


@dataclass(frozen=True)
class PlannedStep:
    """Take the free cube with the id `cube` and put it on `slot`, or stop (kind FINISH). `comment` is for people."""

    kind: str
    cube: str
    slot: str
    comment: str


@dataclass(frozen=True)
class HistoryEntry:
    """A step chosen earlier and what came of it."""

    step: PlannedStep
    outcome: str


def find_slot(box: list[int], slot_boxes: dict[str, list[int]]) -> str | None:
    """The slot whose box holds the centre of a cube's box, the nearest one if slot boxes overlap; None without one.

    A base cube partly hidden by the top one keeps its slot: the centre of its visible part is still on the cube.
    """
    center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    nearest_slot, nearest_distance = None, math.inf
    for slot, (x0, y0, x1, y1) in slot_boxes.items():
        is_inside = x0 <= center_x < x1 and y0 <= center_y < y1
        distance = math.hypot(center_x - (x0 + x1) / 2, center_y - (y0 + y1) / 2)
        if is_inside and distance < nearest_distance:
            nearest_slot, nearest_distance = slot, distance
    return nearest_slot


def describe_table(found_cubes: dict[str, list[list[int]]], slot_boxes: dict[str, list[int]]) -> TableState:
    """State from cube_boxes.detect_cubes's {colour: [box, ...]} and {slot: box} of the same camera.

    Free cubes get the ids 1, 2, ... from left to right on the camera. The numbering is made anew for every state.
    """
    free_boxes = []
    slots: dict[str, str | None] = {slot: None for slot in SLOTS}
    for color, boxes in found_cubes.items():
        for box in boxes:
            slot = find_slot(box, slot_boxes)
            if slot is None:
                free_boxes.append((color, box))
            else:
                slots[slot] = color
    free_boxes.sort(key=lambda color_and_box: color_and_box[1])
    free_cubes = [Cube(str(number), color, box) for number, (color, box) in enumerate(free_boxes, start=1)]
    not_seen = [color for color, boxes in found_cubes.items() if not boxes]
    return TableState(free_cubes, slots, not_seen)


def find_free_cube(cube_id: str, state: TableState) -> Cube | None:
    """The free cube with this id, or None if there is no such cube."""
    return next((cube for cube in state.free_cubes if cube.id == cube_id), None)


def find_rule_violation(step: PlannedStep, state: TableState) -> str | None:
    """Why the arm must not run this step, or None if it may. Goes by the detector, whoever chose the step."""
    if step.kind == FINISH:
        return None
    if find_free_cube(step.cube, state) is None:
        free_ids = [cube.id for cube in state.free_cubes]
        return f"Cube {step.cube!r} is not free on the table; ids of the free cubes: {free_ids}"
    if step.slot not in SLOTS:
        return f"Slot {step.slot!r} does not exist; slots: {list(SLOTS)}"
    if state.slots[step.slot] is not None:
        return f"Slot {step.slot} already holds the {state.slots[step.slot]} cube"
    is_base_ready = all(state.slots[slot] is not None for slot in BASE_SLOTS)
    if step.slot == TOP_SLOT and not is_base_ready:
        return "The top cube needs both base cubes in place first"
    return None


def finish_step(comment: str) -> PlannedStep:
    return PlannedStep(FINISH, NO_CHOICE, NO_CHOICE, comment)


def assign_colors(colors: list[str]) -> dict[str, str]:
    """{slot: colour} from three colours in the order of SLOTS: base_left, base_right, top. A colour may repeat."""
    unknown_colors = set(colors) - set(cube_boxes.COLORS)
    if unknown_colors:
        raise ValueError(f"Unknown colours {sorted(unknown_colors)}; choose from {list(cube_boxes.COLORS)}")
    if len(colors) != len(SLOTS):
        raise ValueError(f"Give {len(SLOTS)} colours for {', '.join(SLOTS)}, got {colors}")
    return dict(zip(SLOTS, colors, strict=True))


def next_loop_step(colors_by_slot: dict[str, str], state: TableState) -> PlannedStep:
    """The first slot from the bottom up without its cube, looked up afresh from the camera before every step.

    With several free cubes of the colour it takes the leftmost one. A cube that fell or was moved simply becomes the
    next step. A slot holding the wrong cube ends the plan: the arm can only take cubes lying free on the table.
    """
    # Checked first: under the finished top cube the lower ones no longer matter and may be hidden from the camera.
    if state.slots[TOP_SLOT] == colors_by_slot[TOP_SLOT]:
        return finish_step("The pyramid is finished")
    slot = next(slot for slot in SLOTS if state.slots[slot] != colors_by_slot[slot])
    color = colors_by_slot[slot]
    if state.slots[slot] is not None:
        return finish_step(f"Slot {slot} holds the {state.slots[slot]} cube instead of {color}; clear it by hand")
    cube = next((cube for cube in state.free_cubes if cube.color == color), None)
    if cube is None:
        return finish_step(f"No {color} cube is free on the table; bring one back by hand")
    return PlannedStep(PICK_AND_PLACE, cube.id, slot, f"Placing the {color} cube {cube.id} on {slot}")
