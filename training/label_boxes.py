#!/usr/bin/env python3
"""Find every cube on the top camera at the start of each episode and save the boxes to meta/cube_boxes.json.

The box of the commanded colour is the target that training draws on the top frames for the whole episode, the same
way run_policy.py draws the box it detects before the arm starts. Boxes of the other colours let eval_offline.py point
the policy at a wrong cube. A contact sheet of the table area with all boxes is saved for a visual check:
coloured boxes are detections, the white one is the target.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import cube_boxes  # noqa: E402
from run_policy import PROJECT, load_config  # noqa: E402

DRAW_BGR = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 200, 0), "yellow": (0, 220, 255)}


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
        start = int(dataset.meta.episodes[ep]["dataset_from_index"])
        length = int(dataset.meta.episodes[ep]["length"])
        for offset in offsets:
            item = dataset[start + min(offset, length - 1)]
            target = color_of_task[item["task"]]
            image = (item[camera].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            boxes = cube_boxes.detect(image)
            if boxes[target] is not None:
                break
        missing = [c for c, b in boxes.items() if b is None]
        if boxes[target] is None:
            problems.append(f"эпизод {ep}: нет целевого {target}")
        elif missing:
            problems.append(f"эпизод {ep}: не найдены {', '.join(missing)} (цель {target} есть, кадр {offset})")
        episodes[str(ep)] = {"target": target, "frame": offset, "boxes": boxes}

        bgr = np.ascontiguousarray(image[:, :, ::-1])
        for color, box in boxes.items():
            if box is not None:
                cv2.rectangle(bgr, (box[0] - 3, box[1] - 3), (box[2] + 3, box[3] + 3), DRAW_BGR[color], 2)
        if boxes[target] is not None:
            bgr = np.ascontiguousarray(cube_boxes.draw_rgb(bgr[:, :, ::-1].copy(), boxes[target])[:, :, ::-1])
        x0, y0, x1, y1 = cube_boxes.TABLE
        tile = cv2.resize(bgr[y0:y1, x0:x1], ((x1 - x0) // 2, (y1 - y0) // 2))
        cv2.putText(tile, f"{ep} {target}" + (f" f{offset}" if offset else ""), (8, 26), 0, 0.8, (255, 255, 255), 2)
        tiles.append(tile)

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
