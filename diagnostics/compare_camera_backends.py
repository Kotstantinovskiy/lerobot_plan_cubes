#!/usr/bin/env python3
"""Sequential wrist-camera A/B capture test; never connects to robot motors."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from camera_backend_trial import BACKENDS, run_worker, run_worker_with_top

PROJECT = Path(__file__).resolve().parents[1]

SUMMARY_KEYS = ("trial", "frames", "fps", "max_gap_ms", "timeouts", "error", "ffmpeg_mode_warnings", "top")
WORKER_EXTRA_TIMEOUT_S = 40  # on top of the seconds measured: startup and cleanup
PAUSE_BETWEEN_TRIALS_S = 2


def worker_command(backend: str, trial_dir: Path, seconds: int, *, with_top: bool) -> list[str]:
    command = [sys.executable, __file__, "--worker", backend, "--out", str(trial_dir), "--seconds", str(seconds)]
    if with_top:
        command.append("--with-top")
    return command


def run_trial(backend: str, trial_dir: Path, seconds: int, *, with_top: bool) -> dict[str, Any]:
    """Run one worker process and return its report; failing to run it gives a report with the error."""
    command = worker_command(
        backend,
        trial_dir,
        seconds,
        with_top=with_top,
    )
    with (trial_dir / "worker.log").open("w") as log:
        try:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=log,
                timeout=seconds + WORKER_EXTRA_TIMEOUT_S,
            )
            report = json.loads((trial_dir / "report.json").read_text())
            report["exit_code"] = completed.returncode
        except Exception as exc:
            report = {"backend": backend, "error": repr(exc)}
    report["trial"] = trial_dir.name
    return report


def trial_summary(report: dict[str, Any]) -> dict[str, Any]:
    """The parts of a trial's report worth printing as it finishes."""
    return {key: report[key] for key in SUMMARY_KEYS if key in report}


def save_usb_tree(path: Path) -> None:
    path.write_text(subprocess.check_output(["ioreg", "-p", "IOUSB", "-w", "0"], text=True))


def run_rounds(seconds: int, rounds: int, *, with_top: bool) -> None:
    """Run the backends in turn, one process per trial, into a new folder of camera_check/."""
    out = PROJECT / "camera_check" / f"backend-comparison-{time.time_ns()}"
    out.mkdir(parents=True)
    save_usb_tree(out / "usb-before.txt")
    print("OUTPUT", out, flush=True)
    reports = []
    for round_id in range(1, rounds + 1):
        for backend in BACKENDS:
            trial_dir = out / f"{round_id}-{backend}"
            trial_dir.mkdir()
            print("START", trial_dir.name, flush=True)
            report = run_trial(
                backend,
                trial_dir,
                seconds,
                with_top=with_top,
            )
            reports.append(report)
            (out / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps(trial_summary(report)), flush=True)
            time.sleep(PAUSE_BETWEEN_TRIALS_S)
    save_usb_tree(out / "usb-after.txt")
    print("DONE", out / "report.json", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=BACKENDS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--with-top", action="store_true", help="Keep top camera streaming through our code during each wrist test"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.worker:
        run_rounds(args.seconds, args.rounds, with_top=args.with_top)
    elif args.with_top:
        run_worker_with_top(args.worker, args.out, args.seconds)
    else:
        run_worker(args.worker, args.out, args.seconds)


if __name__ == "__main__":
    main()
