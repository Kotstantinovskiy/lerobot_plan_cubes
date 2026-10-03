#!/usr/bin/env python3
"""Check a trained policy on recorded frames, without the robot.

Frames are sampled every --every-s seconds. For each one the policy predicts an action chunk under the command of
every colour, with the same noise, and the script reports the mean absolute difference from the recorded actions
over the first --horizon steps (normalized joint units, as in the dataset). The diagonal of the table is the error
with the right command. If the wrong colours give about the same error and the chunk barely moves when only the
colour word changes, the policy ignores the colour in the command.

For a policy trained with target boxes (target_box.json in its folder) a command of a colour also draws the box
around that colour's cube, taken from the dataset's meta/cube_boxes.json: a wrong colour points the box at another
cube, so the table shows whether the policy follows the box and the word. A policy trained on the cropped top view
(top_crop.json) gets the same crop, after the box.

Episodes the policy was trained on only show how well it fitted them; hold some out (train.py --episodes) to see
generalisation. The policy is loaded exactly as inference/run_policy.py loads it on the robot.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
import cube_boxes  # noqa: E402
import top_crop  # noqa: E402
from project_config import PROJECT, load_config, positive_int  # noqa: E402
from run_policy import load_policy, target_box_style  # noqa: E402

ERROR_EXIT_CODE = 2
OBSERVATION_PREFIX = "observation."


class EvalSetup(NamedTuple):
    """What stays the same for every batch of an evaluation."""

    policy: Any
    preprocess: Callable
    postprocess: Callable
    device: str
    horizon: int  # actions of a chunk that are compared with the recording
    observation_keys: list[str]
    tasks: dict[str, str]  # {colour: its command}
    color_of_task: dict[str, str]
    robot_type: str
    box_style: dict | None  # None for a policy trained without target boxes
    box_labels: dict | None  # {episode number as text: {"boxes": {colour: box or None}}}, None without target boxes
    crop: dict | None  # None for a policy trained on the full top view


def episode_list(value: str) -> list[int]:
    """Parse --episodes. argparse names this function when a number is invalid, so its name is part of the message."""
    return sorted({int(item) for item in value.split(",") if item.strip()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="Папка pretrained_model чекпойнта")
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--dataset", type=Path, help="Папка датасета вместо dataset_root из конфига")
    parser.add_argument("--episodes", type=episode_list, help="Номера эпизодов через запятую; по умолчанию все")
    parser.add_argument("--every-s", type=float, default=1.0, help="Шаг выборки кадров, секунды")
    parser.add_argument("--horizon", type=positive_int, default=50, help="Сколько шагов пачки сравнивать")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--output", type=Path, help="Сохранить результаты в JSON")
    return parser


def make_setup(
    args: argparse.Namespace,
    config: dict,
    root: str,
    dataset_meta: LeRobotDatasetMetadata,
    policy: Any,
    preprocess: Callable,
    postprocess: Callable,
) -> EvalSetup:
    """Collect what evaluate_batch needs; reads the policy's box style and crop and, for boxes, the dataset's labels."""
    box_style = target_box_style(args.policy)
    box_labels = None
    if box_style:
        box_labels = json.loads((Path(root) / "meta" / "cube_boxes.json").read_text())["episodes"]
    return EvalSetup(
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        device=args.device,
        horizon=args.horizon,
        observation_keys=[key for key in dataset_meta.features if key.startswith(OBSERVATION_PREFIX)],
        tasks=config["tasks"],
        color_of_task={task: color for color, task in config["tasks"].items()},
        robot_type=dataset_meta.robot_type,
        box_style=box_style,
        box_labels=box_labels,
        crop=top_crop.load(args.policy),
    )


def frame_indices(episode_table, episodes: list[int], stride: int) -> list[int]:
    """Dataset indices of every `stride`-th frame of `episodes`, in the given order.

    `episode_table[episode]` holds the episode's dataset_from_index and dataset_to_index (exclusive).
    """
    indices = []
    for episode in episodes:
        first_index = int(episode_table[episode]["dataset_from_index"])
        end_index = int(episode_table[episode]["dataset_to_index"])
        indices += range(first_index, end_index, stride)
    return indices


def with_target_boxes(observations: dict, items: list[dict], color: str, setup: EvalSetup) -> dict:
    """The observations with a copy of the top frames, a box drawn on `color`'s cube of each frame's episode."""
    camera = setup.box_style["camera"]
    frames = observations[camera].clone()
    for i, item in enumerate(items):
        box = setup.box_labels[str(int(item["episode_index"]))]["boxes"][color]
        if box is not None:  # a cube the detector missed is left unmarked
            cube_boxes.draw_chw(frames[i], box, setup.box_style)
    return {**observations, camera: frames}


def commanded_view(observations: dict, items: list[dict], color: str, setup: EvalSetup) -> dict:
    """The observations as the policy sees them under a command of `color`: with that colour's box, then the crop."""
    view = observations
    if setup.box_style:
        view = with_target_boxes(view, items, color, setup)
    if setup.crop:
        camera = setup.crop["camera"]
        view = {**view, camera: top_crop.crop_chw(view[camera], setup.crop).contiguous()}
    return view


def predict_chunks(observations: dict, items: list[dict], noise: torch.Tensor, setup: EvalSetup) -> dict:
    """{commanded colour: [batch, horizon, action size] array of actions in dataset units}, all from the same noise."""
    chunks = {}
    for color, task in setup.tasks.items():
        view = commanded_view(observations, items, color, setup)
        batch = setup.preprocess({**view, "task": [task] * len(items), "robot_type": setup.robot_type})
        chunk = setup.policy.predict_action_chunk(batch, noise=noise.to(setup.device))
        chunks[color] = setup.postprocess(chunk).float().cpu().numpy()[:, : setup.horizon]
    return chunks


def mean_differences(predicted: dict[str, np.ndarray], reference: np.ndarray) -> dict[str, float]:
    """{commanded colour: mean absolute difference of its actions from `reference`}."""
    return {color: float(np.abs(actions - reference).mean()) for color, actions in predicted.items()}


def make_rows(
    items: list[dict],
    true_colors: list[str],
    chunks: dict,
    recorded_actions: np.ndarray,
    valid_steps: np.ndarray,
) -> list[dict]:
    """One result per frame of a batch: episode, frame, true colour, error and shift per commanded colour.

    `chunks` comes from predict_chunks, `recorded_actions` is [batch, horizon, action size], `valid_steps` a boolean
    [batch, horizon]. "error" is the difference from the recording, "shift" from the chunk of the true colour, both
    over the valid steps.
    """
    rows = []
    for i, item in enumerate(items):
        valid = valid_steps[i]
        predicted = {color: chunk[i][valid] for color, chunk in chunks.items()}
        rows.append(
            {
                "episode": int(item["episode_index"]),
                "frame": int(item["frame_index"]),
                "color": true_colors[i],
                "error": mean_differences(predicted, recorded_actions[i][valid]),
                "shift": mean_differences(predicted, predicted[true_colors[i]]),
            }
        )
    return rows


def evaluate_batch(items: list[dict], noise: torch.Tensor, setup: EvalSetup) -> list[dict]:
    """The rows of a batch of dataset items; `noise` is [batch, chunk size, max action size]."""
    observations = {key: torch.stack([item[key] for item in items]) for key in setup.observation_keys}
    true_colors = [setup.color_of_task[item["task"]] for item in items]
    recorded_actions = torch.stack([item["action"] for item in items]).numpy()
    valid_steps = ~torch.stack([item["action_is_pad"] for item in items]).numpy()
    chunks = predict_chunks(observations, items, noise, setup)
    return make_rows(items, true_colors, chunks, recorded_actions, valid_steps)


def evaluate_frames(dataset, indices: list[int], batch_size: int, seed: int, setup: EvalSetup) -> list[dict]:
    """The rows of the dataset frames at `indices`, in order; prints the progress after every batch."""
    noise_shape = (setup.policy.config.chunk_size, setup.policy.config.max_action_dim)
    generator = torch.Generator().manual_seed(seed)
    rows = []  # one per frame: episode, true colour, error per commanded colour, shift per commanded colour
    for start in range(0, len(indices), batch_size):
        items = [dataset[i] for i in indices[start : start + batch_size]]
        noise = torch.randn((len(items), *noise_shape), generator=generator)
        rows += evaluate_batch(items, noise, setup)
        print(f"{min(start + batch_size, len(indices))}/{len(indices)} кадров", flush=True)
    return rows


def mean_over_other_colors(rows: list[dict], field: str, colors: list[str]) -> float:
    """Mean of a row field ("error" or "shift") over all rows, in each row over the colours other than its own."""
    values = []
    for row in rows:
        values += [row[field][color] for color in colors if color != row["color"]]
    return float(np.mean(values))


def summarize_by_color(rows: list[dict], colors: list[str]) -> dict:
    """{colour in the episode: frame count, mean error per commanded colour, mean shift to the other colours}.

    Colours without frames are left out.
    """
    summary = {}
    for color in colors:
        own_rows = [row for row in rows if row["color"] == color]
        if not own_rows:
            continue
        error = {commanded: float(np.mean([row["error"][commanded] for row in own_rows])) for commanded in colors}
        shift = mean_over_other_colors(own_rows, "shift", colors)
        summary[color] = {"frames": len(own_rows), "error": error, "shift_to_other_colors": shift}
    return summary


def print_report(results: dict, colors: list[str], *, uses_boxes: bool) -> None:
    """Print the header, the table of `summary` and the overall scores of a results dict, as saved by save_results."""
    print(
        f"\nПолитика: {results['policy']}\n"
        f"Эпизоды: {results['episodes']}, кадр каждые {results['every_s']:.2f} с, горизонт {results['horizon']}\n"
        f"Средняя |предсказание - запись| по суставам; строка — цвет в эпизоде, столбец — цвет в команде"
        + (" (слово и рамка на кубике этого цвета)." if uses_boxes else ".")
    )
    print(f"{'':>8} {'кадров':>6} " + " ".join(f"{color:>8}" for color in colors) + f" {'сдвиг':>8}")
    for color, color_summary in results["summary"].items():
        errors = " ".join(f"{color_summary['error'][commanded]:>8.2f}" for commanded in colors)
        print(f"{color:>8} {color_summary['frames']:>6} {errors} {color_summary['shift_to_other_colors']:>8.2f}")
    rows = results["frames"]
    right = np.mean([row["error"][row["color"]] for row in rows])
    wrong = mean_over_other_colors(rows, "error", colors)
    shift = mean_over_other_colors(rows, "shift", colors)
    print(
        f"\nОшибка с правильной командой {right:.2f}, с чужим цветом {wrong:.2f}; "
        f"сдвиг пачки от смены цвета {shift:.2f}."
    )


def save_results(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"Результаты: {output_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    root = str(args.dataset.expanduser().resolve()) if args.dataset else config["dataset_root"]
    colors = list(config["tasks"])
    dataset_meta = LeRobotDatasetMetadata(config["repo_id"], root=root)
    fps = dataset_meta.fps
    dataset = LeRobotDataset(
        config["repo_id"],
        root=root,
        video_backend=args.video_backend,
        delta_timestamps={"action": [i / fps for i in range(args.horizon)]},
    )
    policy, preprocess, postprocess = load_policy(args.policy, args.device, args.horizon, dataset_meta)
    if args.horizon > policy.config.chunk_size:
        parser.exit(ERROR_EXIT_CODE, f"Ошибка: --horizon больше chunk_size политики ({policy.config.chunk_size})\n")
    setup = make_setup(args, config, root, dataset_meta, policy, preprocess, postprocess)

    episodes = args.episodes if args.episodes is not None else list(range(dataset_meta.total_episodes))
    stride = max(1, round(args.every_s * fps))
    indices = frame_indices(dataset_meta.episodes, episodes, stride)
    rows = evaluate_frames(dataset, indices, args.batch_size, args.seed, setup)

    results = {
        "policy": str(args.policy),
        "episodes": episodes,
        "horizon": args.horizon,
        "every_s": stride / fps,
        "summary": summarize_by_color(rows, colors),
        "frames": rows,
    }
    print_report(results, colors, uses_boxes=bool(setup.box_style))
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
