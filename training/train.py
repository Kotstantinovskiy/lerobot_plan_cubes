#!/usr/bin/env python3
"""Fine-tune SmolVLA or pi0.5 on our dataset with the installed LeRobot trainer (lerobot_train).

SmolVLA follows the LeRobot fine-tuning recipe (batch 64, 20k steps); pi0.5 trains all its weights as in openpi, in
fp32 with bf16 autocast like SmolVLA, on batches of 16 for 40k steps (see DEFAULTS). Project-specific fixes for both:
camera keys are renamed to the ones the base policy was pretrained with, image augmentation is on but without hue
jitter (the task is defined by cube colour) and sharpness jitter (too slow on full-resolution frames), and the LR
schedule decays over exactly --steps.
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

# Our camera keys -> the keys each base policy was pretrained with.
RENAME = {
    # camera1 = top, camera2 = wrist
    "smolvla": {"observation.images.top": "observation.images.camera1",
                "observation.images.wrist": "observation.images.camera2"},
    # As openpi feeds single-arm DROID to pi0.5: base_0_rgb is the outside view, left_wrist_0_rgb the wrist camera.
    "pi05": {"observation.images.top": "observation.images.base_0_rgb",
             "observation.images.wrist": "observation.images.left_wrist_0_rgb"},
}
# Batch size, steps and checkpoint interval when not given on the command line. pi0.5 in fp32 (weights, gradients and
# AdamW state: 54 GB) with batches of 16 peaks at 109 GB on an H200 without gradient checkpointing, 0.58 s per step;
# openpi's batch of 32 would need checkpointing, which costs 60% more time per sample. LeRobot's suggested bf16 weights
# fit 32, but there AdamW's steps of ~lr are below the rounding step of most weights and are lost. Every pi0.5
# checkpoint takes 43 GB with the optimizer state.
DEFAULTS = {
    "smolvla": {"batch_size": 64, "steps": 20_000, "save_freq": 2_500},
    "pi05": {"batch_size": 16, "steps": 40_000, "save_freq": 5_000},
}
# pi05_base's processors later got steps (relative actions) that our LeRobot 0.4.4 fork does not know; this revision
# has the same weights and config, with processors it can load.
REVISIONS = {"lerobot/pi05_base": "a538eb273274eb30f126a118f39dbc0ee212c883"}

# LeRobot default augmentation minus two transforms. Hue: a ±0.05 shift moves red towards orange/yellow.
# Sharpness: 0.3 s per sample on our 1080p+720p frames (8x the rest together), which starved the GPU.
# Brightness and contrast are wider than LeRobot's ±0.2 and uneven light is added (LightingGradient, defined in
# lerobot_train_fast.py): all demonstrations were recorded in the evening with a lamp on the left, while the arm also
# runs in daylight. As in LeRobot, each frame gets a random 3 of these; the wrist camera, fixed to the gripper, never
# gets "affine" (lerobot_train_fast.py), so it draws 3 of the other 4.
IMAGE_TRANSFORMS = {
    "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.5, 1.5]}},
    "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.6, 1.4]}},
    "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
    "lighting": {"weight": 1.0, "type": "LightingGradient", "kwargs": {"gain": [0.6, 1.4]}},
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


def local_policy(policy):
    """A pretrained_model folder: the given one, or a Hub id downloaded at its revision from REVISIONS."""
    if Path(policy).is_dir():
        return Path(policy)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(policy, revision=REVISIONS.get(policy)))


def policy_type(folder):
    """"smolvla" or "pi05", from the config.json of a pretrained_model folder."""
    kind = json.loads((folder / "config.json").read_text())["type"]
    if kind not in RENAME:
        raise ValueError(f"{folder}: policy type {kind}, train.py knows {', '.join(RENAME)}")
    return kind


def build_command(args, config, kind):
    output_dir = PROJECT / "outputs" / "train" / args.run_name
    if args.resume:
        train_config = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        options = {"config_path": train_config, "resume": True}
    else:
        options = {
            "policy.path": args.policy,
            "policy.device": args.device,
            "policy.push_to_hub": False,
            "policy.scheduler_decay_steps": args.steps,
            "dataset.repo_id": config["repo_id"],
            "dataset.root": config["dataset_root"],
            "dataset.video_backend": args.video_backend,
            "dataset.image_transforms.enable": not args.no_augment,
            "dataset.image_transforms.tfs": IMAGE_TRANSFORMS,
            "rename_map": RENAME[kind],
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
        if kind == "smolvla":
            # smolvla_base already holds every VLM weight; true would download SmolVLM2 weights on top.
            options["policy.load_vlm_weights"] = False
        if args.episodes:
            options["dataset.episodes"] = args.episodes
        if args.full_finetune:
            options["policy.train_expert_only"] = False
            options["policy.freeze_vision_encoder"] = False
    # lerobot_train with faster data loading: frames shrunk to the policy's input size inside the workers (same model
    # inputs) and no worker restarts between epochs.
    command = [sys.executable, str(Path(__file__).with_name("lerobot_train_fast.py"))]
    gpu_count = len(args.gpu.split(",")) if args.gpu else 1
    if gpu_count > 1:
        # One process per GPU, each with --batch-size samples: the effective batch is gpu_count * --batch-size.
        command[1:1] = ["-m", "accelerate.commands.launch", "--multi_gpu", f"--num_processes={gpu_count}",
                        "--num_machines=1", "--dynamo_backend=no"]
    command += [f"--{key}={encode(value)}" for key, value in options.items()]
    return command + args.extra, output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Аргументы после -- передаются lerobot_train без изменений.")
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json",
                        help="Берутся repo_id и dataset_root")
    parser.add_argument("--dataset", type=Path, help="Папка датасета вместо dataset_root из конфига")
    parser.add_argument("--policy", default="lerobot/smolvla_base",
                        help="Hub id или локальная pretrained_model: lerobot/smolvla_base или lerobot/pi05_base")
    parser.add_argument("--run-name", default=f"run-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--steps", type=positive_int, help="По умолчанию SmolVLA 20000, pi0.5 40000")
    parser.add_argument("--batch-size", type=positive_int, help="На одну GPU; по умолчанию SmolVLA 64, pi0.5 16")
    parser.add_argument("--save-freq", type=positive_int, help="Чекпойнт каждые N шагов: SmolVLA 2500, pi0.5 5000")
    parser.add_argument("--log-freq", type=positive_int, default=100)
    # Workers decode and augment full-resolution frames, ~0.1-0.15 s per sample: on devbox01 (H200) 32 of them could
    # not keep up with a 0.32 s bf16 step, 64 leave the GPU waiting 0.02 s per step.
    parser.add_argument("--num-workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=episode_list, help="Обучать только на этих эпизодах, например 0,1,2")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Обучать и VLM с энкодером изображений, а не только action expert")
    parser.add_argument("--no-augment", action="store_true", help="Без аугментации изображений")
    parser.add_argument("--target-box", action="store_true",
                        help="Рисовать рамку целевого кубика на верхней камере (нужен meta/cube_boxes.json, "
                             "см. training/label_boxes.py)")
    parser.add_argument("--top-crop", action="store_true",
                        help="Обрезать верхнюю камеру до левой части стола без ведущей руки и руки оператора "
                             "(inference/top_crop.py)")
    parser.add_argument("--video-backend", default="pyav", help="pyav или torchcodec (нужен системный FFmpeg)")
    parser.add_argument("--device", default="cuda")
    # lerobot_train creates its Accelerator without mixed_precision, so accelerate takes it from the environment.
    # bf16 autocast keeps fp32 master weights and optimizer state; matmuls run in bf16 (TF32 is on either way).
    # On the pilot with the same seed and data, loss and grad norm matched fp32 to 3 decimals over 900 steps,
    # and the GPU part of a step went from 0.47 to 0.32 s.
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                        help="bf16 — autocast при fp32-весах; fp32 — без autocast")
    parser.add_argument("--gpu", help="Номер GPU (CUDA_VISIBLE_DEVICES), например 0, или несколько: 0,3")
    parser.add_argument("--wandb", action="store_true", help="Логировать в Weights & Biases")
    parser.add_argument("--resume", action="store_true", help="Продолжить --run-name с последнего чекпойнта")
    parser.add_argument("--dry-run", action="store_true", help="Показать команду без запуска")
    parser.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.dataset:
            config["dataset_root"] = str(args.dataset.expanduser().resolve())
        last = PROJECT / "outputs" / "train" / args.run_name / "checkpoints" / "last" / "pretrained_model"
        if args.resume and not last.is_dir():
            raise ValueError(f"Нет чекпойнта для продолжения: {last.parent}")
        if not args.resume:
            args.policy = local_policy(args.policy)
        kind = policy_type(last if args.resume else args.policy)
        for name, value in DEFAULTS[kind].items():
            if getattr(args, name) is None:
                setattr(args, name, value)
        command, output_dir = build_command(args, config, kind)
        print(f"Датасет: {config['dataset_root']}\nЗапуск: {output_dir}", flush=True)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        if importlib.util.find_spec("lerobot") is None:
            raise ValueError("LeRobot не установлен в этом Python. См. training/README.md")
        if args.resume:
            # A run trained with boxes or the crop keeps them; the checkpoint says so.
            args.target_box |= (last / "target_box.json").is_file()
            args.top_crop |= (last / "top_crop.json").is_file()
        else:
            if not (Path(config["dataset_root"]) / "meta" / "info.json").is_file():
                raise ValueError(f"Нет датасета: {config['dataset_root']}. См. training/fetch_dataset.sh")
            if output_dir.exists():
                raise ValueError(f"Папка запуска уже существует: {output_dir}. Укажи другой --run-name или --resume")
            if args.target_box:
                meta = Path(config["dataset_root"]) / "meta"
                if not (meta / "cube_boxes.json").is_file():
                    raise ValueError("Нет meta/cube_boxes.json: сначала python training/label_boxes.py")
                # Labels made before episodes were added miss them, and training would stop at the first one.
                labelled = {int(ep) for ep in json.loads((meta / "cube_boxes.json").read_text())["episodes"]}
                total = json.loads((meta / "info.json").read_text())["total_episodes"]
                missing = sorted(set(args.episodes or range(total)) - labelled)
                if missing:
                    raise ValueError(f"В meta/cube_boxes.json нет эпизодов {', '.join(map(str, missing))}: "
                                     "заново python training/label_boxes.py")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["ACCELERATE_MIXED_PRECISION"] = "bf16" if args.precision == "bf16" else "no"
        env["TARGET_BOX"] = "1" if args.target_box else "0"
        env["TOP_CROP"] = "1" if args.top_crop else "0"
        env["POLICY_TYPE"] = kind
        # Relative paths inside lerobot (outputs/, caches) resolve against the project root.
        return subprocess.call(command, env=env, cwd=PROJECT)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
