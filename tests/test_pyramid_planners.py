"""The pyramid as a plan: what the table state says, which steps the rules allow, what the loop planner chooses."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import cube_boxes  # noqa: E402
from inference_pyramid_helpers import (  # noqa: E402
    COLORS_BY_SLOT,
    FINISHED,
    FREE_ALL,
    PLACE_RED_LEFT,
    SLOT_BOXES,
    box_centered_at,
    make_frame,
    make_state,
)
from pyramid_plan import (  # noqa: E402
    FINISH,
    NO_CHOICE,
    PICK_AND_PLACE,
    Cube,
    PlannedStep,
    assign_colors,
    describe_table,
    find_free_cube,
    find_rule_violation,
    next_loop_step,
)


class TableStateTests(unittest.TestCase):
    def test_cube_inside_a_slot_box_is_in_that_slot(self):
        state = describe_table({"red": [box_centered_at(590, 330)]}, SLOT_BOXES)
        self.assertEqual(state.slots["base_left"], "red")

    def test_cube_outside_slot_boxes_is_free(self):
        box = box_centered_at(800, 500)
        state = describe_table({"red": [box]}, SLOT_BOXES)
        self.assertEqual(state.free_cubes, [Cube("1", "red", box)])

    def test_color_without_cubes_is_reported_as_not_seen(self):
        state = describe_table({"red": []}, SLOT_BOXES)
        self.assertEqual(state.not_seen, ["red"])

    def test_overlapping_slot_boxes_choose_the_nearest_center(self):
        state = describe_table({"yellow": [box_centered_at(620, 330)]}, SLOT_BOXES)
        self.assertEqual(state.slots["top"], "yellow")

    def test_two_free_cubes_of_a_color_get_their_own_ids(self):
        left_box, right_box = box_centered_at(450, 500), box_centered_at(750, 500)
        state = describe_table({"red": [right_box, left_box]}, SLOT_BOXES)
        self.assertEqual(state.free_cubes, [Cube("1", "red", left_box), Cube("2", "red", right_box)])

    def test_free_cubes_are_numbered_left_to_right_across_colors(self):
        state = describe_table(
            {"red": [box_centered_at(800, 500)], "blue": [box_centered_at(450, 500)]},
            SLOT_BOXES,
        )
        self.assertEqual([(cube.id, cube.color) for cube in state.free_cubes], [("1", "blue"), ("2", "red")])

    def test_cube_in_a_slot_leaves_its_twin_free(self):
        twin_box = box_centered_at(800, 500)
        state = describe_table({"red": [box_centered_at(590, 330), twin_box]}, SLOT_BOXES)
        self.assertEqual(state.slots["base_left"], "red")
        self.assertEqual(state.free_cubes, [Cube("1", "red", twin_box)])

    def test_cubes_of_one_color_can_fill_both_base_slots(self):
        state = describe_table({"red": [box_centered_at(590, 330), box_centered_at(650, 330)]}, SLOT_BOXES)
        self.assertEqual((state.slots["base_left"], state.slots["base_right"]), ("red", "red"))

    def test_detector_on_a_frame_gives_the_table_state(self):
        frame = make_frame({"red": (590, 330), "blue": (800, 500)})
        state = describe_table(cube_boxes.detect_cubes(frame), SLOT_BOXES)
        self.assertEqual(state.slots["base_left"], "red")
        self.assertEqual([cube.color for cube in state.free_cubes], ["blue"])
        self.assertCountEqual(state.not_seen, ["green", "yellow"])


class RuleTests(unittest.TestCase):
    def test_free_cube_on_an_empty_base_slot_is_allowed(self):
        self.assertIsNone(find_rule_violation(PLACE_RED_LEFT, make_state(FREE_ALL)))

    def test_top_is_refused_until_both_base_cubes_are_in_place(self):
        step = PlannedStep(PICK_AND_PLACE, "1", "top", "")
        state = make_state(["yellow", "blue"], base_left="red")
        self.assertIn("both base cubes", find_rule_violation(step, state))

    def test_top_is_allowed_on_a_full_base(self):
        step = PlannedStep(PICK_AND_PLACE, "1", "top", "")
        state = make_state(["yellow"], base_left="red", base_right="blue")
        self.assertIsNone(find_rule_violation(step, state))

    def test_occupied_slot_is_refused(self):
        state = make_state(["red"], base_left="blue")
        self.assertIn("already holds", find_rule_violation(PLACE_RED_LEFT, state))

    def test_id_without_a_free_cube_is_refused_with_the_valid_ids(self):
        step = PlannedStep(PICK_AND_PLACE, "9", "base_right", "")
        problem = find_rule_violation(step, make_state(["blue", "red"]))
        self.assertIn("not free", problem)
        self.assertIn("['1', '2']", problem)

    def test_pick_and_place_without_a_slot_is_refused(self):
        step = PlannedStep(PICK_AND_PLACE, "1", NO_CHOICE, "")
        self.assertIn("does not exist", find_rule_violation(step, make_state(FREE_ALL)))

    def test_finish_is_always_allowed(self):
        self.assertIsNone(find_rule_violation(FINISHED, make_state([])))


class LoopPlannerTests(unittest.TestCase):
    def test_first_step_fills_base_left(self):
        state = make_state(FREE_ALL)
        step = next_loop_step(COLORS_BY_SLOT, state)
        self.assertEqual((find_free_cube(step.cube, state).color, step.slot), ("red", "base_left"))

    def test_second_step_fills_base_right(self):
        state = make_state(["blue", "yellow"], base_left="red")
        step = next_loop_step(COLORS_BY_SLOT, state)
        self.assertEqual((find_free_cube(step.cube, state).color, step.slot), ("blue", "base_right"))

    def test_top_comes_last(self):
        state = make_state(["yellow"], base_left="red", base_right="blue")
        step = next_loop_step(COLORS_BY_SLOT, state)
        self.assertEqual((find_free_cube(step.cube, state).color, step.slot), ("yellow", "top"))

    def test_finishes_when_top_is_right_even_with_a_hidden_base(self):
        step = next_loop_step(COLORS_BY_SLOT, make_state([], top="yellow"))
        self.assertEqual(step.kind, FINISH)

    def test_cube_knocked_off_its_slot_becomes_the_next_step(self):
        state = make_state(["red", "yellow"], base_right="blue")
        step = next_loop_step(COLORS_BY_SLOT, state)
        self.assertEqual((find_free_cube(step.cube, state).color, step.slot), ("red", "base_left"))

    def test_wrong_cube_in_a_slot_ends_the_plan(self):
        step = next_loop_step(COLORS_BY_SLOT, make_state(["red"], base_left="green"))
        self.assertEqual(step.kind, FINISH)
        self.assertIn("clear it by hand", step.comment)

    def test_missing_cube_ends_the_plan(self):
        step = next_loop_step(COLORS_BY_SLOT, make_state(["blue"]))
        self.assertEqual(step.kind, FINISH)
        self.assertIn("red", step.comment)

    def test_first_of_two_free_cubes_of_a_color_is_taken(self):
        step = next_loop_step(COLORS_BY_SLOT, make_state(["blue", "red", "red"]))
        self.assertEqual(step.cube, "2")

    def test_repeated_color_fills_two_slots(self):
        self.assertEqual(
            assign_colors(["red", "red", "blue"]),
            {"base_left": "red", "base_right": "red", "top": "blue"},
        )

    def test_pair_of_one_color_takes_a_second_cube_of_it(self):
        colors_by_slot = assign_colors(["red", "red", "blue"])
        state = make_state(["blue", "red"], base_left="red")
        step = next_loop_step(colors_by_slot, state)
        self.assertEqual((step.cube, step.slot), ("2", "base_right"))

    def test_wrong_number_of_colors_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Give 3 colours"):
            assign_colors(["red", "blue"])

    def test_unknown_color_is_refused(self):
        with self.assertRaisesRegex(ValueError, "purple"):
            assign_colors(["red", "blue", "purple"])


if __name__ == "__main__":
    unittest.main()
