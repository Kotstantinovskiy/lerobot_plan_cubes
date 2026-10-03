"""Choose the next pyramid step with a language model through OpenRouter.

The model only chooses among the allowed steps: it sees neither joints nor coordinates, and nothing here moves the arm.
The colour detector says where the cubes are, the model turns a free-form goal into one step at a time, and
pyramid_plan.find_rule_violation checks that step; a rejected one goes back to the model with the reason.
Several cubes can share a colour, so every free cube gets its id drawn on the photo and the model answers with an id.
The cropped top camera frame leaves this computer, so keep OPENROUTER_API_KEY in the environment, not in config.json
(it is in Git).
"""

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict

import cv2
import numpy as np

import cube_boxes
import top_crop
from pyramid_plan import (
    FINISH,
    NO_CHOICE,
    PICK_AND_PLACE,
    SLOTS,
    Cube,
    HistoryEntry,
    PlannedStep,
    TableState,
    find_rule_violation,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-5.5"
API_KEY_VARIABLE = "OPENROUTER_API_KEY"
REQUEST_TIMEOUT_SECONDS = 60
MAX_REJECTED_ANSWERS = 3
JPEG_QUALITY = 85
ID_FONT = cv2.FONT_HERSHEY_SIMPLEX
ID_FONT_SCALE = 0.7
ID_THICKNESS = 2
ID_OUTLINE_THICKNESS = 5
WHITE, BLACK = (255, 255, 255), (0, 0, 0)

STEP_KINDS = (PICK_AND_PLACE, FINISH)
SLOT_CHOICES = (*SLOTS, NO_CHOICE)
STEP_SCHEMA = {
    "name": "next_step",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(STEP_KINDS)},
            "cube": {"type": "string"},
            "slot": {"type": "string", "enum": list(SLOT_CHOICES)},
            "comment": {"type": "string"},
        },
        "required": ["kind", "cube", "slot", "comment"],
        "additionalProperties": False,
    },
}

PLANNER_RULES = f"""\
You choose the next step for a small robot arm that builds a pyramid of toy cubes on a table.
The cubes come in four colours: {", ".join(cube_boxes.COLORS)}, and the table can hold several cubes of one colour. \
The pyramid has three slots: base_left and base_right stand side by side (left and right as seen on the photo), and \
top rests on both.

Every request gives the goal, the table state from a colour detector, the history of earlier steps and a photo from \
the camera above the table. The arm is folded away at the back of the photo. Each cube in table.free_cubes has its id \
drawn on it in the photo. Ids are numbered anew for every photo, so an id in the history may mean another cube now. \
The table state decides what is allowed; use the photo to understand the scene, to tell cubes of one colour apart and \
to talk about it.

Answer with exactly one step:
- {PICK_AND_PLACE}: the arm takes the free cube whose id is in "cube" and puts it on the empty slot in "slot".
- {FINISH}: stop, with cube and slot set to "{NO_CHOICE}". Use it when the goal is reached and also when it cannot be \
reached, and say why in the comment.

Rules:
- The top slot can be filled only when both base slots are filled.
- The arm can do nothing else: it cannot take a cube out of a slot, move other objects or build other shapes. If the \
goal needs that, finish and say so.
- When several free cubes suit the goal and it does not say which to take, take one standing clear of other cubes: the \
gripper needs room on both sides of the cube.
- A step in the history marked as rejected broke a rule; its outcome says which one. Choose differently.

comment: one short sentence for the people watching, in the language of the goal. Say which cube you take by its \
colour and its place on the table, not by its id.
"""


def read_api_key() -> str:
    """The OpenRouter key from the environment. Raises RuntimeError, saying what to do, if it is not set."""
    api_key = os.environ.get(API_KEY_VARIABLE)
    if not api_key:
        raise RuntimeError(f"{API_KEY_VARIABLE} is not set: create a key at https://openrouter.ai/keys and export it")
    return api_key


def draw_cube_ids(top_rgb: np.ndarray, cubes: list[Cube]) -> np.ndarray:
    """A copy of an HxWx3 uint8 RGB top camera frame with each cube's id in the middle of its box."""
    labelled = top_rgb.copy()
    for cube in cubes:
        x0, y0, x1, y1 = cube.box
        (text_width, text_height), _ = cv2.getTextSize(
            cube.id,
            ID_FONT,
            ID_FONT_SCALE,
            ID_THICKNESS,
        )
        origin = ((x0 + x1 - text_width) // 2, (y0 + y1 + text_height) // 2)
        cv2.putText(
            labelled,
            cube.id,
            origin,
            ID_FONT,
            ID_FONT_SCALE,
            BLACK,
            ID_OUTLINE_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.putText(
            labelled,
            cube.id,
            origin,
            ID_FONT,
            ID_FONT_SCALE,
            WHITE,
            ID_THICKNESS,
            cv2.LINE_AA,
        )
    return labelled


def encode_top_view(top_rgb: np.ndarray, cubes: list[Cube]) -> bytes:
    """JPEG bytes of an HxWx3 uint8 RGB top camera frame with the cubes' ids on it, cropped as for the policy."""
    view_bgr = cv2.cvtColor(top_crop.crop_hwc(draw_cube_ids(top_rgb, cubes)), cv2.COLOR_RGB2BGR)
    is_encoded, jpeg = cv2.imencode(".jpg", view_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not is_encoded:
        raise RuntimeError("Could not encode the top camera frame as JPEG")
    return jpeg.tobytes()


def describe_table_for_model(state: TableState) -> dict:
    """The table state without the boxes: their pixels are those of the full frame, not of the cropped photo."""
    free_cubes = [{"id": cube.id, "color": cube.color} for cube in state.free_cubes]
    return {"free_cubes": free_cubes, "slots": state.slots, "not_seen": state.not_seen}


def build_request_body(goal: str, state: TableState, history: list[HistoryEntry], top_jpeg: bytes, model: str) -> dict:
    """The chat completion request for one step; the text goes before the image, as OpenRouter recommends."""
    scene = {
        "goal": goal,
        "table": describe_table_for_model(state),
        "history": [asdict(entry) for entry in history],
    }
    image_url = "data:image/jpeg;base64," + base64.b64encode(top_jpeg).decode("ascii")
    user_content = [
        {"type": "text", "text": json.dumps(scene, ensure_ascii=False)},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    return {
        "model": model,
        "messages": [{"role": "system", "content": PLANNER_RULES}, {"role": "user", "content": user_content}],
        "response_format": {"type": "json_schema", "json_schema": STEP_SCHEMA},
        # Only endpoints that support the schema and do not collect the camera frame.
        "provider": {"require_parameters": True, "data_collection": "deny"},
    }


def post_chat_completion(body: dict, api_key: str) -> dict:
    """The decoded answer to a request. Raises RuntimeError if OpenRouter refuses or cannot be reached."""
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter answered {error.code}: {details}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Cannot reach OpenRouter: {error}") from error


def parse_step(answer: dict) -> PlannedStep:
    """The step from a decoded chat completion. Raises RuntimeError on a reported error or a malformed answer."""
    # OpenRouter can report a provider's failure in a normal 200 answer.
    if "error" in answer:
        raise RuntimeError(f"OpenRouter reported an error: {answer['error']}")
    content = answer["choices"][0]["message"]["content"]
    try:
        fields = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"The model's answer is not JSON: {content!r}") from error
    # Not every provider holds the model to the schema exactly.
    is_valid = (
        isinstance(fields, dict)
        and fields.get("kind") in STEP_KINDS
        and isinstance(fields.get("cube"), str)
        and fields.get("slot") in SLOT_CHOICES
        and isinstance(fields.get("comment"), str)
    )
    if not is_valid:
        raise RuntimeError(f"The model's answer is outside the schema: {fields!r}")
    return PlannedStep(
        kind=fields["kind"],
        cube=fields["cube"],
        slot=fields["slot"],
        comment=fields["comment"],
    )


def choose_step(
    goal: str,
    top_rgb: np.ndarray,
    state: TableState,
    history: list[HistoryEntry],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> PlannedStep:
    """The model's next allowed step for a top camera frame with the arm folded away and its table state.

    The step's cube is an id from state.free_cubes. Rejected answers are appended to `history` with their reasons.
    Raises RuntimeError if the request fails or the model breaks the rules MAX_REJECTED_ANSWERS times in a row.
    """
    top_jpeg = encode_top_view(top_rgb, state.free_cubes)
    for _ in range(MAX_REJECTED_ANSWERS):
        body = build_request_body(
            goal,
            state,
            history,
            top_jpeg,
            model,
        )
        step = parse_step(post_chat_completion(body, api_key))
        problem = find_rule_violation(step, state)
        if problem is None:
            return step
        history.append(HistoryEntry(step, f"rejected: {problem}"))
    raise RuntimeError(f"The model broke the rules {MAX_REJECTED_ANSWERS} times in a row; {history[-1].outcome}")
