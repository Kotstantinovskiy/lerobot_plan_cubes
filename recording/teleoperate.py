#!/usr/bin/env python3
"""Управление ведомой рукой через ведущую, без записи датасета."""

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from project_config import CONFIG_PATH, load_config, positive_int
from record import (
    SETUP_ERROR_EXIT_CODE,
    SETUP_ERRORS,
    arm_options,
    option_arguments,
    require_lerobot,
    validate_arms,
    validate_cameras,
)

TELEOPERATE_MODULE = "lerobot.scripts.lerobot_teleoperate"
DEFAULT_FPS = 30


def _teleoperate_command(options: dict[str, Any], seconds: int | None) -> list[str]:
    if seconds is not None:
        options["teleop_time_s"] = seconds
    return [sys.executable, "-m", TELEOPERATE_MODULE, *option_arguments(options)]


def build_command_without_cameras(config: dict[str, Any], seconds: int | None = None) -> list[str]:
    """Build LeRobot's teleoperation command with the robot's cameras left out and nothing displayed."""
    options = arm_options(config)
    options.pop("robot.cameras", None)
    options["robot.cameras"] = {}
    options["fps"] = config.get("fps", DEFAULT_FPS)
    options["display_data"] = False
    return _teleoperate_command(options, seconds)


def build_command_with_cameras(config: dict[str, Any], seconds: int | None = None) -> list[str]:
    """Build LeRobot's teleoperation command with the configured cameras connected and displayed."""
    options = arm_options(config)
    options["fps"] = config.get("fps", DEFAULT_FPS)
    options["display_data"] = True
    return _teleoperate_command(options, seconds)


def teleoperate(args: argparse.Namespace) -> int:
    """Check the setup and hand over to LeRobot, or just print the command with --dry-run and return 0."""
    config = load_config(args.config)
    if args.cameras:
        command = build_command_with_cameras(config, args.seconds)
    else:
        command = build_command_without_cameras(config, args.seconds)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    validate_arms(config)
    if args.cameras:
        validate_cameras(config)
    require_lerobot()
    print("Телеуправление без записи. Остановка: Ctrl+C.", flush=True)
    # Replace the launcher so Ctrl+C reaches LeRobot directly.
    os.execv(sys.executable, command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cameras", action="store_true", help="Подключить камеры и показать изображение")
    parser.add_argument("--seconds", type=positive_int, help="Завершить управление через указанное число секунд")
    parser.add_argument("--dry-run", action="store_true", help="Показать команду без подключения оборудования")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        return teleoperate(args)
    except SETUP_ERRORS as exc:
        parser.exit(SETUP_ERROR_EXIT_CODE, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
