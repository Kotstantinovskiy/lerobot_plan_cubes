#!/usr/bin/env python3
"""Fine-tune SmolVLA or pi0.5 on our dataset with the installed LeRobot trainer (lerobot_train).

SmolVLA follows the LeRobot fine-tuning recipe (batch 64, 20k steps); pi0.5 trains all its weights as in openpi, in
fp32 with bf16 autocast like SmolVLA, on batches of 16 for 40k steps (see DEFAULTS). Project-specific fixes for both:
camera keys are renamed to the ones the base policy was pretrained with, image augmentation is on but without hue
jitter (the task is defined by cube colour) and sharpness jitter (too slow on full-resolution frames), and the LR
schedule decays over exactly --steps.
Runs are written to outputs/train/<run-name>; --resume continues the last checkpoint of that run.
"""

# Annotations stay unevaluated: this launcher needs only the standard library, so a plain python3 can run --dry-run.
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
from project_config import PROJECT, load_config, positive_int  # noqa: E402

# Our camera keys, each with the key the base policy was pretrained with.
RENAME = {
    # camera1 = top, camera2 = wrist
    "smolvla": {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    },
    # As openpi feeds single-arm DROID to pi0.5: base_0_rgb is the outside view, left_wrist_0_rgb the wrist camera.
    "pi05": {
        "observation.images.top": "observation.images.base_0_rgb",
        "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    },
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

TRAINER_SCRIPT = Path(__file__).with_name("lerobot_train_fast.py")
# Workers decode and augment full-resolution frames, ~0.1-0.15 s per sample: on devbox01 (H200) 32 of them could
# not keep up with a 0.32 s bf16 step, 64 leave the GPU waiting 0.02 s per step.
MAX_NUM_WORKERS = 64
# lerobot_train_fast.py leaves these in a checkpoint trained with boxes or the crop.
TARGET_BOX_FILE = "target_box.json"
TOP_CROP_FILE = "top_crop.json"
# And this in every checkpoint: the --precision it was trained with.
PRECISION_FILE = "train_precision.txt"
DEFAULT_PRECISION = "bf16"
ERROR_EXIT_CODE = 2


def episode_list(value: str) -> list[int]:
    """Parse --episodes: the numbers of a comma-separated list, sorted, without repeats."""
    try:
        episodes = sorted({int(item) for item in value.split(",") if item.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError("Ожидается список номеров через запятую, например 0,1,2") from None
    if not episodes or episodes[0] < 0:
        raise argparse.ArgumentTypeError("Нужен хотя бы один неотрицательный номер эпизода")
    return episodes


def encode_option(value) -> str:
    """An option value as lerobot_train reads it from the command line: JSON for dicts, lists and booleans."""
    return json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)


def find_policy_folder(policy: str) -> Path:
    """A pretrained_model folder: the given one, or a Hub id downloaded at its revision from REVISIONS."""
    if Path(policy).is_dir():
        # Absolute: the trainer runs from the project root, which may not be where this was called from.
        return Path(policy).resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(policy, revision=REVISIONS.get(policy)))


def read_policy_type(folder: Path) -> str:
    """The policy type, "smolvla" or "pi05", from the config.json of a pretrained_model folder."""
    policy_type = json.loads((folder / "config.json").read_text())["type"]
    if policy_type not in RENAME:
        raise ValueError(f"{folder}: policy type {policy_type}, train.py knows {', '.join(RENAME)}")
    return policy_type


def run_folder(run_name: str) -> Path:
    """The folder of a run's checkpoints and logs."""
    return PROJECT / "outputs" / "train" / run_name


def last_checkpoint_model(run_name: str) -> Path:
    """The pretrained_model folder of a run's last checkpoint."""
    return run_folder(run_name) / "checkpoints" / "last" / "pretrained_model"


def new_run_options(args: argparse.Namespace, config: dict, policy_type: str) -> dict:
    """{lerobot_train option: value, not yet encoded} of a new run, in command line order."""
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
        "rename_map": RENAME[policy_type],
        "batch_size": args.batch_size,
        "steps": args.steps,
        "save_freq": args.save_freq,
        "log_freq": args.log_freq,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "output_dir": run_folder(args.run_name),
        "job_name": args.run_name,
        "wandb.enable": args.wandb,
    }
    if policy_type == "smolvla":
        # smolvla_base already holds every VLM weight; true would download SmolVLM2 weights on top.
        options["policy.load_vlm_weights"] = False
    if args.episodes:
        options["dataset.episodes"] = args.episodes
    if args.full_finetune:
        options["policy.train_expert_only"] = False
        options["policy.freeze_vision_encoder"] = False
    return options


def autocast_mode(precision: str) -> str:
    """accelerate's mixed_precision setting for a --precision value."""
    return "bf16" if precision == "bf16" else "no"


def multi_gpu_launcher(gpu: str | None, precision: str) -> list[str]:
    """What goes between the Python interpreter and the script for --gpu; nothing for a single GPU."""
    gpu_count = len(gpu.split(",")) if gpu else 1
    if gpu_count == 1:
        return []
    # One process per GPU, each with --batch-size samples: the effective batch is gpu_count * --batch-size.
    return [
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        f"--num_processes={gpu_count}",
        "--num_machines=1",
        "--dynamo_backend=no",
        # accelerate launch overwrites ACCELERATE_MIXED_PRECISION in the processes it starts unless it is told this.
        f"--mixed_precision={autocast_mode(precision)}",
    ]


def build_command(args: argparse.Namespace, config: dict, policy_type: str) -> tuple[list[str], Path]:
    """The training command line, as a list of arguments, and the run's output folder.

    `args` is the parsed command line with the policy folder, the policy's DEFAULTS and the precision filled in,
    `config` the project config (repo_id and dataset_root are used), `policy_type` a key of RENAME.
    """
    if args.resume:
        options = {"config_path": last_checkpoint_model(args.run_name) / "train_config.json", "resume": True}
    else:
        options = new_run_options(args, config, policy_type)
    # lerobot_train with faster data loading: frames shrunk to the policy's input size inside the workers (same model
    # inputs) and no worker restarts between epochs.
    command = [sys.executable, *multi_gpu_launcher(args.gpu, args.precision), str(TRAINER_SCRIPT)]
    command += [f"--{key}={encode_option(value)}" for key, value in options.items()]
    return command + args.extra, run_folder(args.run_name)


def training_environment(args: argparse.Namespace, policy_type: str) -> dict[str, str]:
    """This process's environment plus what the trainer reads from it: GPU choice, autocast, box, crop, policy type."""
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["ACCELERATE_MIXED_PRECISION"] = autocast_mode(args.precision)
    env["TARGET_BOX"] = "1" if args.target_box else "0"
    env["TOP_CROP"] = "1" if args.top_crop else "0"
    env["POLICY_TYPE"] = policy_type
    return env


def fill_policy_defaults(args: argparse.Namespace, policy_type: str) -> None:
    """Set the values of DEFAULTS that the command line did not give, in place."""
    for name, value in DEFAULTS[policy_type].items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def saved_precision(saved_model: Path) -> str:
    """The --precision a checkpoint was trained with; older checkpoints do not say, so they get the default."""
    precision_path = saved_model / PRECISION_FILE
    if precision_path.is_file():
        return precision_path.read_text().strip()
    print(
        f"Чекпойнт не хранит точность обучения: продолжаю с {DEFAULT_PRECISION}, другую можно задать через --precision",
        flush=True,
    )
    return DEFAULT_PRECISION


def check_box_labels(meta: Path, episodes: list[int] | None) -> None:
    """Raise ValueError unless the dataset's meta/cube_boxes.json labels `episodes` (None means all of them)."""
    labels_path = meta / "cube_boxes.json"
    if not labels_path.is_file():
        raise ValueError("Нет meta/cube_boxes.json: сначала python training/label_boxes.py")
    # Labels made before episodes were added miss them, and training would stop at the first one.
    labelled = {int(episode) for episode in json.loads(labels_path.read_text())["episodes"]}
    total = json.loads((meta / "info.json").read_text())["total_episodes"]
    missing = sorted(set(episodes or range(total)) - labelled)
    if missing:
        raise ValueError(
            f"В meta/cube_boxes.json нет эпизодов {', '.join(map(str, missing))}: заново python training/label_boxes.py"
        )


def check_new_run(args: argparse.Namespace, dataset_root: str, output_dir: Path) -> None:
    """Raise ValueError if a new run cannot start: no dataset, the run folder is taken, or box labels are missing."""
    meta = Path(dataset_root) / "meta"
    if not (meta / "info.json").is_file():
        raise ValueError(f"Нет датасета: {dataset_root}. См. training/fetch_dataset.sh")
    if output_dir.exists():
        raise ValueError(f"Папка запуска уже существует: {output_dir}. Укажи другой --run-name или --resume")
    if args.target_box:
        check_box_labels(meta, args.episodes)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json", help="Берутся repo_id и dataset_root")
    parser.add_argument("--dataset", type=Path, help="Папка датасета вместо dataset_root из конфига")
    parser.add_argument(
        "--policy",
        default="lerobot/smolvla_base",
        help="Hub id или локальная pretrained_model: lerobot/smolvla_base или lerobot/pi05_base",
    )
    parser.add_argument("--run-name", default=f"run-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--steps", type=positive_int, help="По умолчанию SmolVLA 20000, pi0.5 40000")
    parser.add_argument("--batch-size", type=positive_int, help="На одну GPU; по умолчанию SmolVLA 64, pi0.5 16")
    parser.add_argument("--save-freq", type=positive_int, help="Чекпойнт каждые N шагов: SmolVLA 2500, pi0.5 5000")
    parser.add_argument("--log-freq", type=positive_int, default=100)
    parser.add_argument("--num-workers", type=int, default=min(MAX_NUM_WORKERS, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1000)


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=episode_list, help="Обучать только на этих эпизодах, например 0,1,2")
    parser.add_argument(
        "--full-finetune", action="store_true", help="Обучать и VLM с энкодером изображений, а не только action expert"
    )
    parser.add_argument("--no-augment", action="store_true", help="Без аугментации изображений")
    parser.add_argument(
        "--target-box",
        action="store_true",
        help="Рисовать рамку целевого кубика на верхней камере (нужен meta/cube_boxes.json, "
        "см. training/label_boxes.py)",
    )
    parser.add_argument(
        "--top-crop",
        action="store_true",
        help="Обрезать верхнюю камеру до левой части стола без ведущей руки и руки оператора (inference/top_crop.py)",
    )
    parser.add_argument("--video-backend", default="pyav", help="pyav или torchcodec (нужен системный FFmpeg)")


def add_launch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    # lerobot_train creates its Accelerator without mixed_precision, so accelerate takes it from the environment.
    # bf16 autocast keeps fp32 master weights and optimizer state; matmuls run in bf16 (TF32 is on either way).
    # On the pilot with the same seed and data, loss and grad norm matched fp32 to 3 decimals over 900 steps,
    # and the GPU part of a step went from 0.47 to 0.32 s.
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        help=f"bf16 — autocast при fp32-весах; fp32 — без autocast. По умолчанию {DEFAULT_PRECISION}, "
        "а при --resume та же, что у чекпойнта",
    )
    parser.add_argument("--gpu", help="Номер GPU (CUDA_VISIBLE_DEVICES), например 0, или несколько: 0,3")
    parser.add_argument("--wandb", action="store_true", help="Логировать в Weights & Biases")
    parser.add_argument("--resume", action="store_true", help="Продолжить --run-name с последнего чекпойнта")
    parser.add_argument("--dry-run", action="store_true", help="Показать команду без запуска")
    parser.add_argument("extra", nargs="*", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Аргументы после -- передаются lerobot_train без изменений.",
    )
    add_run_arguments(parser)
    add_input_arguments(parser)
    add_launch_arguments(parser)
    return parser


def run_training(args: argparse.Namespace) -> int:
    """Check the setup and run the trainer, or only show its command with --dry-run; gives the exit code to report.

    Fills `args` in place: the policy folder, the policy's defaults, and the box, crop and precision of a checkpoint.
    """
    config = load_config(args.config)
    if args.dataset:
        config["dataset_root"] = str(args.dataset.expanduser().resolve())
    saved_model = last_checkpoint_model(args.run_name)
    if args.resume and not saved_model.is_dir():
        raise ValueError(f"Нет чекпойнта для продолжения: {saved_model.parent}")
    if not args.resume:
        args.policy = find_policy_folder(args.policy)
    policy_type = read_policy_type(saved_model if args.resume else args.policy)
    fill_policy_defaults(args, policy_type)
    if args.precision is None:
        args.precision = saved_precision(saved_model) if args.resume else DEFAULT_PRECISION
    command, output_dir = build_command(args, config, policy_type)
    print(f"Датасет: {config['dataset_root']}\nЗапуск: {output_dir}", flush=True)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    if importlib.util.find_spec("lerobot") is None:
        raise ValueError("LeRobot не установлен в этом Python. См. training/README.md")
    if args.resume:
        # A run trained with boxes or the crop keeps them; the checkpoint says so.
        args.target_box |= (saved_model / TARGET_BOX_FILE).is_file()
        args.top_crop |= (saved_model / TOP_CROP_FILE).is_file()
    else:
        check_new_run(args, config["dataset_root"], output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    env = training_environment(args, policy_type)
    # Relative paths inside lerobot (outputs/, caches) resolve against the project root.
    return subprocess.call(command, env=env, cwd=PROJECT)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_training(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(ERROR_EXIT_CODE, f"Ошибка: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
