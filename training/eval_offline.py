#!/usr/bin/env python3
"""Check a trained policy on recorded frames, without the robot.

Frames are sampled every --every-s seconds. For each one the policy predicts an action chunk under the command of
every colour, with the same noise, and the script reports the mean absolute difference from the recorded actions
over the first --horizon steps (normalized joint units, as in the dataset). The diagonal of the table is the error
with the right command. If the wrong colours give about the same error and the chunk barely moves when only the
colour word changes, the policy ignores the colour in the command.

Episodes the policy was trained on only show how well it fitted them; hold some out (train.py --episodes) to see
generalisation. The policy is loaded exactly as inference/run_policy.py loads it on the robot.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from run_policy import PROJECT, load_config, load_policy, positive_int  # noqa: E402


def episode_list(value):
    return sorted({int(item) for item in value.split(",") if item.strip()})


def main():
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
    args = parser.parse_args()

    config = load_config(args.config)
    root = str(args.dataset.expanduser().resolve()) if args.dataset else config["dataset_root"]
    colors = list(config["tasks"])
    color_of_task = {task: color for color, task in config["tasks"].items()}
    ds_meta = LeRobotDatasetMetadata(config["repo_id"], root=root)
    fps = ds_meta.fps
    dataset = LeRobotDataset(config["repo_id"], root=root, video_backend=args.video_backend,
                             delta_timestamps={"action": [i / fps for i in range(args.horizon)]})
    policy, pre, post = load_policy(args.policy, args.device, args.horizon, ds_meta)
    if args.horizon > policy.config.chunk_size:
        parser.exit(2, f"Ошибка: --horizon больше chunk_size политики ({policy.config.chunk_size})\n")
    noise_shape = (policy.config.chunk_size, policy.config.max_action_dim)
    generator = torch.Generator().manual_seed(args.seed)

    episodes = args.episodes if args.episodes is not None else list(range(ds_meta.total_episodes))
    stride = max(1, round(args.every_s * fps))
    indices = [i for e in episodes for i in range(int(ds_meta.episodes[e]["dataset_from_index"]),
                                                  int(ds_meta.episodes[e]["dataset_to_index"]), stride)]
    obs_keys = [k for k in ds_meta.features if k.startswith("observation.")]
    rows = []  # one per frame: episode, true colour, error per commanded colour, shift per commanded colour
    for start in range(0, len(indices), args.batch_size):
        items = [dataset[i] for i in indices[start:start + args.batch_size]]
        obs = {k: torch.stack([item[k] for item in items]) for k in obs_keys}
        truth = [color_of_task[item["task"]] for item in items]
        gt = torch.stack([item["action"] for item in items]).numpy()
        valid = ~torch.stack([item["action_is_pad"] for item in items]).numpy()
        noise = torch.randn((len(items), *noise_shape), generator=generator)
        chunks = {}
        for color in colors:
            batch = pre({**obs, "task": [config["tasks"][color]] * len(items), "robot_type": ds_meta.robot_type})
            chunk = policy.predict_action_chunk(batch, noise=noise.to(args.device))
            chunks[color] = post(chunk).float().cpu().numpy()[:, :args.horizon]
        for j, item in enumerate(items):
            mask = valid[j]
            right = chunks[truth[j]][j][mask]
            rows.append({
                "episode": int(item["episode_index"]),
                "frame": int(item["frame_index"]),
                "color": truth[j],
                "error": {c: float(np.abs(chunks[c][j][mask] - gt[j][mask]).mean()) for c in colors},
                "shift": {c: float(np.abs(chunks[c][j][mask] - right).mean()) for c in colors},
            })
        print(f"{min(start + args.batch_size, len(indices))}/{len(indices)} кадров", flush=True)

    print(f"\nПолитика: {args.policy}\nЭпизоды: {episodes}, кадр каждые {stride / fps:.2f} с, горизонт {args.horizon}\n"
          f"Средняя |предсказание - запись| по суставам; строка — цвет в эпизоде, столбец — цвет в команде.")
    print(f"{'':>8} {'кадров':>6} " + " ".join(f"{c:>8}" for c in colors) + f" {'сдвиг':>8}")
    summary = {}
    for color in colors:
        mine = [r for r in rows if r["color"] == color]
        if not mine:
            continue
        error = {c: float(np.mean([r["error"][c] for r in mine])) for c in colors}
        shift = float(np.mean([r["shift"][c] for r in mine for c in colors if c != color]))
        summary[color] = {"frames": len(mine), "error": error, "shift_to_other_colors": shift}
        print(f"{color:>8} {len(mine):>6} " + " ".join(f"{error[c]:>8.2f}" for c in colors) + f" {shift:>8.2f}")
    right = np.mean([r["error"][r["color"]] for r in rows])
    wrong = np.mean([r["error"][c] for r in rows for c in colors if c != r["color"]])
    shift = np.mean([r["shift"][c] for r in rows for c in colors if c != r["color"]])
    print(f"\nОшибка с правильной командой {right:.2f}, с чужим цветом {wrong:.2f}; "
          f"сдвиг пачки от смены цвета {shift:.2f}.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"policy": str(args.policy), "episodes": episodes, "horizon": args.horizon,
                                           "every_s": stride / fps, "summary": summary, "frames": rows},
                                          ensure_ascii=False, indent=1))
        print(f"Результаты: {args.output}")


if __name__ == "__main__":
    main()
