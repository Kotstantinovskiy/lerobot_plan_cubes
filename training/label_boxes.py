#!/usr/bin/env python3
"""Find every cube on the top camera at the start of each episode and save the boxes to meta/cube_boxes.json.

The box of the commanded colour is the target that training draws on the top frames for the whole episode, the same
way run_policy.py draws the box it detects before the arm starts. With two cubes of that colour on the table the
target is the one that moved during the episode: the last frame shows which. Boxes of the other colours let
eval_offline.py point the policy at a wrong cube. A contact sheet of the table area with all boxes is saved for a
visual check: coloured boxes are detections, the white one is the target.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import cube_boxes  # noqa: E402
from run_policy import PROJECT, load_config  # noqa: E402

DRAW_BGR = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 200, 0), "yellow": (0, 220, 255)}
TWO_CUBES_SIDE = 70  # pixels; a cube's box is ~35-50 wide, or up to ~65 when it is turned by 45 degrees
CLEARLY_MOVED = 40  # pixels; a cube closer than this to a box of its colour at the end may just have been covered


def box_center(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def distance_to_nearest(box: list[int], others: list[list[int]]) -> float:
    """Pixels from the centre of `box` to the nearest centre in `others`; infinite if there are none."""
    center_x, center_y = box_center(box)
    distances = [math.hypot(center_x - box_center(other)[0], center_y - box_center(other)[1]) for other in others]
    return min(distances, default=math.inf)


def split_two_cubes(both: list[int], stayed: list[int]) -> list[int]:
    """The box of the cube that left, from the box around two touching cubes and the box of the one that stayed."""
    both_x, both_y = box_center(both)
    stayed_x, stayed_y = box_center(stayed)
    half_width, half_height = (stayed[2] - stayed[0]) / 2, (stayed[3] - stayed[1]) / 2
    left_x, left_y = 2 * both_x - stayed_x, 2 * both_y - stayed_y
    return [
        round(left_x - half_width),
        round(left_y - half_height),
        round(left_x + half_width),
        round(left_y + half_height),
    ]


def find_moved_box(start_boxes: list[list[int]], end_boxes: list[list[int]]) -> tuple[list[int], bool]:
    """Which box of a colour is the cube that moved between the first and last frames, and is that certain?

    Input: the colour's boxes in each frame, which must hold at least one at the start. Output: the box in the first
    frame and whether exactly one cube clearly moved. A single box at the start is the answer as it is, unless it
    surrounds two touching cubes, one of which stayed.
    """
    if len(start_boxes) == 1:
        only = start_boxes[0]
        is_two_cubes = max(only[2] - only[0], only[3] - only[1]) > TWO_CUBES_SIDE
        if not is_two_cubes:
            return only, True
        stayed = min(end_boxes, key=lambda box: distance_to_nearest(box, [only]), default=None)
        if stayed is None or distance_to_nearest(stayed, [only]) > TWO_CUBES_SIDE:
            return only, False
        return split_two_cubes(only, stayed), True
    moved_distances = [distance_to_nearest(box, end_boxes) for box in start_boxes]
    clearly_moved_count = sum(distance > CLEARLY_MOVED for distance in moved_distances)
    farthest = max(range(len(start_boxes)), key=moved_distances.__getitem__)
    return start_boxes[farthest], clearly_moved_count == 1


def frame_rgb(dataset, index: int, camera: str):
    return (dataset[index][camera].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def label_episode(dataset, ep: int, camera: str, color_of_task: dict[str, str], offsets: list[int]):
    """The episode's entry for cube_boxes.json, what to report about it, and the frame its boxes come from."""
    start = int(dataset.meta.episodes[ep]["dataset_from_index"])
    length = int(dataset.meta.episodes[ep]["length"])
    problems = []
    for offset in offsets:
        index = start + min(offset, length - 1)
        target = color_of_task[dataset[index]["task"]]
        image = frame_rgb(dataset, index, camera)
        found = cube_boxes.detect_all(image)
        if found[target]:
            break

    boxes = {color: (found[color][0] if found[color] else None) for color in found}
    if not found[target]:
        problems.append(f"эпизод {ep}: нет целевого {target}")
    else:
        end_boxes = cube_boxes.detect_all(frame_rgb(dataset, start + length - 1, camera))[target]
        boxes[target], is_certain = find_moved_box(found[target], end_boxes)
        if not is_certain:
            problems.append(f"эпизод {ep}: не ясно, какой {target} сдвинулся, взят {boxes[target]}: проверь глазами")
    missing = [color for color, box in boxes.items() if box is None]
    if boxes[target] is not None and missing:
        problems.append(f"эпизод {ep}: не найдены {', '.join(missing)} (цель {target} есть, кадр {offset})")
    return {"target": target, "frame": offset, "boxes": boxes}, problems, image


def make_tile(ep: int, entry: dict, image):
    """A piece of the contact sheet: the table area with the episode's boxes."""
    target, offset, boxes = entry["target"], entry["frame"], entry["boxes"]
    bgr = np.ascontiguousarray(image[:, :, ::-1])
    for color, box in boxes.items():
        if box is not None:
            cv2.rectangle(bgr, (box[0] - 3, box[1] - 3), (box[2] + 3, box[3] + 3), DRAW_BGR[color], 2)
    if boxes[target] is not None:
        bgr = np.ascontiguousarray(cube_boxes.draw_rgb(bgr[:, :, ::-1].copy(), boxes[target])[:, :, ::-1])
    x0, y0, x1, y1 = cube_boxes.TABLE
    tile = cv2.resize(bgr[y0:y1, x0:x1], ((x1 - x0) // 2, (y1 - y0) // 2))
    cv2.putText(tile, f"{ep} {target}" + (f" f{offset}" if offset else ""), (8, 26), 0, 0.8, (255, 255, 255), 2)
    return tile


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--dataset", type=Path, help="Папка датасета вместо dataset_root из конфига")
    parser.add_argument("--offsets", default="0,10,20,40", help="Кадры от начала эпизода, по очереди, пока цель не найдена")
    parser.add_argument("--sheet", type=Path, help="Куда сохранить лист с рамками (по умолчанию outputs/boxes/<датасет>.jpg)")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.dataset.expanduser().resolve() if args.dataset else config["dataset_root"])
    color_of_task = {task: color for color, task in config["tasks"].items()}
    dataset = LeRobotDataset(config["repo_id"], root=root, video_backend="pyav")
    offsets = [int(x) for x in args.offsets.split(",")]
    camera = cube_boxes.STYLE["camera"]

    episodes, tiles, problems = {}, [], []
    for ep in range(dataset.meta.total_episodes):
        entry, episode_problems, image = label_episode(dataset, ep, camera, color_of_task, offsets)
        episodes[str(ep)] = entry
        problems += episode_problems
        tiles.append(make_tile(ep, entry, image))

    out = root / "meta" / "cube_boxes.json"
    out.write_text(json.dumps({"camera": camera, "episodes": episodes}, indent=1))
    columns = 5
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % columns)
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    sheet_path = args.sheet or PROJECT / "outputs" / "boxes" / f"{root.name}.jpg"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    found = sum(e["boxes"][e["target"]] is not None for e in episodes.values())
    print(f"Цель найдена в {found} из {len(episodes)} эпизодов. Рамки: {out}\nЛист для проверки: {sheet_path}")
    for line in problems:
        print("  " + line)
    return 0 if found == len(episodes) else 1


if __name__ == "__main__":
    sys.exit(main())
