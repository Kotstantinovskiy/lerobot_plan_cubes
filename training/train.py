#!/usr/bin/env python3
"""Fine-tune SmolVLA on our dataset with the installed LeRobot trainer (lerobot_train).

Defaults follow the LeRobot SmolVLA fine-tuning recipe (batch 64, 20k steps) with project-specific fixes:
camera keys are renamed to the ones smolvla_base was pretrained with, image augmentation is on but without
hue jitter (the task is defined by cube colour) and sharpness jitter (too slow on full-resolution frames),
and the LR schedule decays over exactly --steps.
Runs are written to outputs/train/<run-name>; --resume continues the last checkpoint of that run.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
from record import PROJECT, load_config, positive_int  # noqa: E402

# Our camera keys -> the keys smolvla_base was pretrained with (camera1 = top, camera2 = wrist).
RENAME = {"observation.images.top": "observation.images.camera1",
          "observation.images.wrist": "observation.images.camera2"}

# LeRobot default augmentation minus two transforms. Hue: a ±0.05 shift moves red towards orange/yellow.
# Sharpness: 0.3 s per sample on our 1080p+720p frames (8x the rest together), which starved the GPU.
IMAGE_TRANSFORMS = {
    "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
    "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
    "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
    "affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}},
}


def episode_list(value):
    try:
        episodes = sorted({int(item) for item in value.split(",") if item.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError("Ожидается список номеров через запятую, например 0,1,2") from None
    if not episodes or episodes[0] < 0:
        raise argparse.ArgumentTypeError("Нужен хотя бы один неотрицательный номер эпизода")
    return episodes


def encode(value):
    return json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)


def build_command(args, config):
    output_dir = PROJECT / "outputs" / "train" / args.run_name
    if args.resume:
        train_config = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        options = {"config_path": train_config, "resume": True}
    else:
        options = {
            "policy.path": args.policy,
            "policy.device": args.device,
            "policy.push_to_hub": False,
            # smolvla_base already holds every VLM weight; true would download SmolVLM2 weights on top.
            "policy.load_vlm_weights": False,
            "policy.scheduler_decay_steps": args.steps,
            "dataset.repo_id": config["repo_id"],
            "dataset.root": config["dataset_root"],
            "dataset.video_backend": args.video_backend,
            "dataset.image_transforms.enable": not args.no_augment,
            "dataset.image_transforms.tfs": IMAGE_TRANSFORMS,
            "rename_map": RENAME,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "save_freq": args.save_freq,
            "log_freq": args.log_freq,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "output_dir": output_dir,
            "job_name": args.run_name,
            "wandb.enable": args.wandb,
        }
        if args.episodes:
            options["dataset.episodes"] = args.episodes
        if args.full_finetune:
            options["policy.train_expert_only"] = False
            options["policy.freeze_vision_encoder"] = False
    # lerobot_train with faster data loading: frames shrunk to SmolVLA's input size inside the workers (same model
    # inputs) and no worker restarts between epochs.
    command = [sys.executable, str(Path(__file__).with_name("lerobot_train_fast.py"))]
    command += [f"--{key}={encode(value)}" for key, value in options.items()]
    return command + args.extra, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Аргументы после -- передаются lerobot_train без изменений.")
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json",
                        help="Берутся repo_id и dataset_root")
    parser.add_argument("--dataset", type=Path, help="Папка датасета вместо dataset_root из конфига")
    parser.add_argument("--policy", default="lerobot/smolvla_base", help="Hub id или локальная pretrained_model")
    parser.add_argument("--run-name", default=f"smolvla-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--steps", type=positive_int, default=20_000)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--save-freq", type=positive_int, default=2_500)
    parser.add_argument("--log-freq", type=positive_int, default=100)
    # Workers decode and augment full-resolution frames, ~0.1-0.15 s per sample: on devbox01 (H200) 32 of them could
    # not keep up with a 0.32 s bf16 step, 64 leave the GPU waiting 0.02 s per step.
    parser.add_argument("--num-workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=episode_list, help="Обучать только на этих эпизодах, например 0,1,2")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Обучать и VLM с энкодером изображений, а не только action expert")
    parser.add_argument("--no-augment", action="store_true", help="Без аугментации изображений")
    parser.add_argument("--video-backend", default="pyav", help="pyav или torchcodec (нужен системный FFmpeg)")
    parser.add_argument("--device", default="cuda")
    # lerobot_train creates its Accelerator without mixed_precision, so accelerate takes it from the environment.
    # bf16 autocast keeps fp32 master weights and optimizer state; matmuls run in bf16 (TF32 is on either way).
    # On the pilot with the same seed and data, loss and grad norm matched fp32 to 3 decimals over 900 steps,
    # and the GPU part of a step went from 0.47 to 0.32 s.
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                        help="bf16 — autocast при fp32-весах; fp32 — без autocast")
    parser.add_argument("--gpu", help="Номер GPU (CUDA_VISIBLE_DEVICES), например 0")
    parser.add_argument("--wandb", action="store_true", help="Логировать в Weights & Biases")
    parser.add_argument("--resume", action="store_true", help="Продолжить --run-name с последнего чекпойнта")
    parser.add_argument("--dry-run", action="store_true", help="Показать команду без запуска")
    parser.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.dataset:
            config["dataset_root"] = str(args.dataset.expanduser().resolve())
        command, output_dir = build_command(args, config)
        print(f"Датасет: {config['dataset_root']}\nЗапуск: {output_dir}", flush=True)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        if importlib.util.find_spec("lerobot") is None:
            raise ValueError("LeRobot не установлен в этом Python. См. training/README.md")
        if args.resume:
            if not (output_dir / "checkpoints" / "last").exists():
                raise ValueError(f"Нет чекпойнта для продолжения: {output_dir / 'checkpoints' / 'last'}")
        else:
            if not (Path(config["dataset_root"]) / "meta" / "info.json").is_file():
                raise ValueError(f"Нет датасета: {config['dataset_root']}. См. training/fetch_dataset.sh")
            if output_dir.exists():
                raise ValueError(f"Папка запуска уже существует: {output_dir}. Укажи другой --run-name или --resume")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["ACCELERATE_MIXED_PRECISION"] = "bf16" if args.precision == "bf16" else "no"
        # Relative paths inside lerobot (outputs/, caches) resolve against the project root.
        return subprocess.call(command, env=env, cwd=PROJECT)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
