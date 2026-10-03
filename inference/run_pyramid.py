#!/usr/bin/env python3
"""Run a pyramid planner over top camera frames, without the arm: each frame is the table before the next step.

--planner loop goes through given colours in order, offline and free: --colors red,red,yellow fills base_left,
base_right and then top. --planner llm asks a language model through OpenRouter to reach a free-form --goal; it needs
OPENROUTER_API_KEY and sends the cropped frames out of this computer.

Frames are 1280x720 images of the top camera. Slot boxes come from a JSON file, {"base_left": [x0, y0, x1, y1], ...};
without it every slot counts as empty. The script prints the detector's boxes, which helps to write that file, and for
each step the id, colour and box of the exact cube to take.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2

import cube_boxes
from llm_planner import DEFAULT_MODEL, choose_step, read_api_key
from pyramid_plan import (
    FINISH,
    SLOTS,
    HistoryEntry,
    assign_colors,
    describe_table,
    find_free_cube,
    next_loop_step,
)

LOOP_PLANNER, LLM_PLANNER = "loop", "llm"
DEFAULT_COLORS = "red,blue,yellow"
TOP_FRAME_SHAPE = (720, 1280)  # the frame cube_boxes.py is tuned for


def load_frame(path: Path):
    """HxWx3 uint8 RGB frame from an image file."""
    frame_bgr = cv2.imread(str(path))
    if frame_bgr is None:
        raise ValueError(f"{path}: cannot read it as an image")
    if frame_bgr.shape[:2] != TOP_FRAME_SHAPE:
        height, width = frame_bgr.shape[:2]
        raise ValueError(f"{path}: expected a 1280x720 top camera frame, got {width}x{height}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def load_slot_boxes(path: Path | None) -> dict[str, list[int]]:
    """{slot: box} from a JSON file; empty without a file."""
    if path is None:
        print("No --slot-boxes: every slot counts as empty.", flush=True)
        return {}
    slot_boxes = json.loads(path.read_text())
    unknown_slots = set(slot_boxes) - set(SLOTS)
    if unknown_slots:
        raise ValueError(f"{path}: unknown slots {sorted(unknown_slots)}; slots are {list(SLOTS)}")
    return slot_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", choices=(LOOP_PLANNER, LLM_PLANNER), default=LOOP_PLANNER)
    parser.add_argument("--frame", type=Path, nargs="+", required=True, help="Top camera frames, in order")
    parser.add_argument("--slot-boxes", type=Path, help="JSON file with the box of each slot")
    parser.add_argument("--colors", default=DEFAULT_COLORS, help="loop: colours for base_left, base_right and top")
    parser.add_argument("--goal", help="llm: what to build, in free form")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="llm: OpenRouter model id")
    args = parser.parse_args()
    if args.planner == LLM_PLANNER and not args.goal:
        parser.error("--planner llm needs --goal")
    return args


def make_step_chooser(args: argparse.Namespace):
    """The chosen planner as one function: (top_rgb, state, history) gives the next step."""
    if args.planner == LOOP_PLANNER:
        colors_by_slot = assign_colors(args.colors.split(","))

        def choose_loop_step(_top_rgb, state, _history):
            return next_loop_step(colors_by_slot, state)

        return choose_loop_step

    api_key = read_api_key()

    def choose_llm_step(top_rgb, state, history):
        return choose_step(args.goal, top_rgb, state, history, api_key=api_key, model=args.model)

    return choose_llm_step


def run_planner(args: argparse.Namespace) -> None:
    """Print the step the chosen planner gives for each frame, until it finishes or the frames run out."""
    slot_boxes = load_slot_boxes(args.slot_boxes)
    choose_next_step = make_step_chooser(args)
    history = []
    for frame_path in args.frame:
        top_rgb = load_frame(frame_path)
        found_cubes = cube_boxes.detect_cubes(top_rgb)
        state = describe_table(found_cubes, slot_boxes)
        print(f"\n{frame_path}\n  boxes: {found_cubes}\n  table: {asdict(state)}", flush=True)

        rejected_before = len(history)
        step = choose_next_step(top_rgb, state, history)
        for entry in history[rejected_before:]:
            print(f"  {entry.outcome} (cube {entry.step.cube} to {entry.step.slot})", flush=True)

        if step.kind == FINISH:
            print(f"  step: finish\n  comment: {step.comment}", flush=True)
            return
        cube = find_free_cube(step.cube, state)
        print(f"  step: cube {cube.id} ({cube.color}) to {step.slot}", flush=True)
        print(f"  target box: {cube.box}\n  comment: {step.comment}", flush=True)
        history.append(HistoryEntry(step, "run"))
    print("\nThe frames ran out before the planner finished.", flush=True)


def main() -> int:
    args = parse_args()
    try:
        run_planner(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
