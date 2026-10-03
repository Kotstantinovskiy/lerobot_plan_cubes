#!/usr/bin/env python3
"""Record cube colors using the installed LeRobot recorder, with the videos encoded live (live_video.py).

One color per session, or several taking turns episode by episode (--color blue,yellow,red,green).
"""

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from project_config import CONFIG_PATH, PROJECT, load_config, positive_int

TASK_CYCLE = "--task_cycle="  # live_video.py's argument: the colors and their tasks in turn, as JSON [[color, task]]
LIVE_VIDEO_SCRIPT = PROJECT / "recording" / "live_video.py"
ARM_SECTIONS = ("robot", "teleop")
REQUIRED_ARM_SETTINGS = ("type", "port", "id")
UNSET_MARK = "SET_"  # the values of config.json still to be replaced start like this
VIDEO_CODEC = "h264"
SETUP_ERRORS = (OSError, ValueError, KeyError, TypeError)  # reported as one line instead of a traceback
SETUP_ERROR_EXIT_CODE = 2


def arm_options(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten the config's robot and teleop sections into LeRobot options such as {"robot.port": "/dev/tty..."}."""
    options = {}
    for section in ARM_SECTIONS:
        for key, value in config[section].items():
            options[f"{section}.{key}"] = value
    return options


def option_arguments(options: dict[str, Any]) -> list[str]:
    """Turn options into `--key=value` arguments: dicts, lists and booleans as JSON, anything else as text."""
    arguments = []
    for key, value in options.items():
        encoded = json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)
        arguments.append(f"--{key}={encoded}")
    return arguments


def build_command(config: dict[str, Any], colors: list[str], episodes: int, *, resume: bool) -> tuple[list[str], Path]:
    """Build the recorder's command for `episodes` episodes of `colors`, taking turns when there are several.

    `config` is what load_config returns. Returns the command and the dataset folder; raises ValueError when a
    color's task is not a non-empty text.
    """
    tasks = [config["tasks"][color] for color in colors]
    if not all(isinstance(task, str) and task.strip() for task in tasks):
        raise ValueError("Для цвета нужна непустая текстовая команда")
    root = Path(config["dataset_root"]).resolve()
    options = arm_options(config)
    options.update(
        {
            "dataset.repo_id": config["repo_id"],
            "dataset.root": str(root),
            "dataset.single_task": tasks[0],
            "dataset.num_episodes": episodes,
            "dataset.fps": config["fps"],
            "dataset.episode_time_s": config["episode_time_s"],
            "dataset.reset_time_s": config["reset_time_s"],
            "dataset.video": True,
            "dataset.vcodec": VIDEO_CODEC,
            "dataset.push_to_hub": False,
            "display_data": config.get("display_data", True),
            "play_sounds": config.get("play_sounds", False),
            "resume": resume,
        }
    )
    command = [sys.executable, str(LIVE_VIDEO_SCRIPT), *option_arguments(options)]
    if len(colors) > 1:
        command.append(TASK_CYCLE + json.dumps(list(zip(colors, tasks, strict=False))))
    return command, root


def validate_arms(config: dict[str, Any]) -> None:
    """Raise ValueError unless both arms have their settings filled in and different ports."""
    for section in ARM_SECTIONS:
        for key in REQUIRED_ARM_SETTINGS:
            value = config[section].get(key)
            is_filled_in = isinstance(value, str) and value and UNSET_MARK not in value
            if not is_filled_in:
                raise ValueError(f"Заполни {section}.{key} в config.json")
    if config["robot"]["port"] == config["teleop"]["port"]:
        raise ValueError("У ведущей и ведомой руки должны быть разные порты")


def validate_cameras(config: dict[str, Any]) -> None:
    """Raise ValueError unless the robot has at least one camera and every camera has an index or path."""
    cameras = config["robot"].get("cameras", {})
    if not cameras:
        raise ValueError("Для VLA нужна хотя бы одна камера")
    for name, camera in cameras.items():
        if camera.get("index_or_path") is None:
            raise ValueError(f"Укажи индекс камеры {name} или удали её из настроек")


def require_lerobot() -> None:
    """Raise ValueError when LeRobot is not installed in this Python."""
    if importlib.util.find_spec("lerobot") is None:
        raise ValueError("LeRobot не установлен в этом Python. См. раздел установки в README.md")


def check_dataset_folder(root: Path, *, resume: bool) -> None:
    """Raise ValueError unless --resume finds a dataset to add to, or a new recording finds its folder free."""
    if resume and not (root / "meta" / "info.json").is_file():
        raise ValueError("Нет существующего датасета: первый запуск сделай без --resume")
    if not resume and root.exists():
        raise ValueError("Папка датасета уже существует: используй --resume или другой dataset_root")


def print_session(config: dict[str, Any], colors: list[str], root: Path) -> None:
    if len(colors) > 1:
        print(f"Цвета по очереди, по одному на эпизод: {', '.join(colors)}", flush=True)
    for color in dict.fromkeys(colors):
        print(f"Задание: {config['tasks'][color]}", flush=True)
    print(f"Датасет: {root}", flush=True)


def run_recorder(command: list[str]) -> int:
    """Run the recorder and return its exit code."""
    # Use the same interpreter/environment; never execute configuration as shell code.
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.call(command, env=env)


def record_session(args: argparse.Namespace) -> int:
    """Check the setup and record, or just print the command with --dry-run. Returns the exit code."""
    config = load_config(args.config)
    colors = args.color.split(",")
    if not set(colors) <= set(config["tasks"]):
        raise ValueError(f"Доступные цвета: {', '.join(config['tasks'])}")
    command, root = build_command(
        config,
        colors,
        args.episodes,
        resume=args.resume,
    )
    print_session(config, colors, root)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    validate_arms(config)
    validate_cameras(config)
    require_lerobot()
    check_dataset_folder(root, resume=args.resume)
    root.parent.mkdir(parents=True, exist_ok=True)
    return run_recorder(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--color",
        required=True,
        help="Ключ цвета из tasks в config.json; несколько через запятую идут по очереди, по эпизоду",
    )
    parser.add_argument("--episodes", type=positive_int, default=5)
    parser.add_argument("--resume", action="store_true", help="Добавить эпизоды в существующий датасет")
    parser.add_argument("--dry-run", action="store_true", help="Показать команду без подключения устройств")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        return record_session(args)
    except SETUP_ERRORS as exc:
        parser.exit(SETUP_ERROR_EXIT_CODE, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
