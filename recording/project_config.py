"""Where the project lives and how to read its config.json; scripts in every folder import this."""

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "config.json"


def _resolve_from(base: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the project config, with its dataset and calibration paths made absolute."""
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    for section in ("robot", "teleop"):
        calibration_dir = config.get(section, {}).get("calibration_dir")
        if calibration_dir:
            config[section]["calibration_dir"] = _resolve_from(path.parent, calibration_dir)
    config["dataset_root"] = _resolve_from(path.parent, config["dataset_root"])
    return config


def positive_int(value: str) -> int:
    """Parse a command line integer that must be at least 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Значение должно быть больше нуля")
    return number
