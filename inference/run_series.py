#!/usr/bin/env python3
"""A series of run_policy.py trials over colours and cube layouts, each one's outcome typed in after it.

Every layout serves each colour once, in an order that rotates from one layout to the next. While a trial's policy
loads, set up the cubes: a new layout when the series says so, otherwise put back what the arm moved; the arm starts
when Enter is pressed. Ctrl+C during a trial stops it as in run_policy.py (the arm folds back) and the series goes on.
After each trial type its outcome:
  p  placed on the platform     g  grasped but not placed
  m  not grasped                w  went for another cube
  r  run the trial again (not counted)     q  stop the series
The outcomes, with each trial's summary from run_policy.py, go to camera_check/series-<time>/results.csv, and a
tally is printed after every trial.

Arguments other than those below go to run_policy.py, e.g.
  python inference/run_series.py --policy DIR --layouts 3 --seconds 30 --max-step 10 --replan 10 --rtc
"""

import argparse
import csv
import json
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple

INFERENCE = Path(__file__).resolve().parent
PROJECT = INFERENCE.parent
PLACED, GRASPED, MISSED, WRONG_CUBE = "placed", "grasped", "missed", "wrong cube"
OUTCOMES = {"p": PLACED, "g": GRASPED, "m": MISSED, "w": WRONG_CUBE}
RUN_AGAIN, STOP = "r", "q"
RESULTS_FILE = "results.csv"
OUTPUT_FOLDER = PROJECT / "camera_check"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
PROMPT = "Outcome: p placed, g grasped, m missed, w wrong cube; r run again, q stop: "


class ScheduledTrial(NamedTuple):
    """One planned trial: the cube layout it belongs to (from 1) and the colour commanded."""

    layout: int
    color: str


@dataclass(frozen=True)
class TrialResult:
    """One line of results.csv; the fields are its columns, in order."""

    trial: int
    layout: int
    color: str
    outcome: str
    target_box: list[int] | None
    seam_median: float | None
    seam_p90: float | None
    steps: int
    video: str


def count_outcomes(results: list[TrialResult]) -> str:
    """Placed, grasped (placed or not) and wrong-cube counts of these trials, as a text."""
    trial_count = len(results)
    placed_count = sum(result.outcome == PLACED for result in results)
    grasped_count = sum(result.outcome in (PLACED, GRASPED) for result in results)
    wrong_cube_count = sum(result.outcome == WRONG_CUBE for result in results)
    return (
        f"placed {placed_count}/{trial_count}, grasped {grasped_count}/{trial_count}, "
        f"wrong cube {wrong_cube_count}/{trial_count}"
    )


def tally(results: list[TrialResult]) -> str:
    """Placed, grasped (placed or not) and wrong-cube counts over all trials, then per colour."""
    lines = [f"All: {count_outcomes(results)}"]
    for color in dict.fromkeys(result.color for result in results):
        results_of_color = [result for result in results if result.color == color]
        lines.append(f"  {color}: {count_outcomes(results_of_color)}")
    return "\n".join(lines)


def make_schedule(colors: list[str], layout_count: int) -> list[ScheduledTrial]:
    """Every layout serves each colour once, the order rotating by one colour from one layout to the next."""
    schedule = []
    for layout in range(1, layout_count + 1):
        for i in range(len(colors)):
            schedule.append(ScheduledTrial(layout, colors[(layout - 1 + i) % len(colors)]))
    return schedule


def run_trial(policy: str, color: str, summary_path: Path, run_args: list[str]) -> dict | None:
    """Runs run_policy.py; its summary, or None if no trial ran (cancelled, or failed before the arm moved)."""
    summary_path.unlink(missing_ok=True)
    child = subprocess.Popen(
        [
            sys.executable,
            str(INFERENCE / "run_policy.py"),
            "--policy",
            policy,
            "--color",
            color,
            "--wait",
            "--summary",
            str(summary_path),
            *run_args,
        ],
        cwd=PROJECT,
    )
    # Ctrl+C reaches the trial too, which folds the arm back; here it is ignored. Only once the trial has started:
    # it would inherit an ignored Ctrl+C.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        child.wait()
    finally:
        signal.signal(signal.SIGINT, previous)
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    return summary if summary["stats"]["steps"] else None


def ask_outcome() -> str:
    """The letter typed after a trial, asked for until it is a valid one."""
    while True:
        code = input(PROMPT).strip().lower()
        if code in (*OUTCOMES, RUN_AGAIN, STOP):
            return code


def make_result(trial: int, scheduled: ScheduledTrial, outcome: str, summary: dict) -> TrialResult:
    """The results.csv line of a trial, from its typed outcome and the summary run_policy.py wrote."""
    seams = summary["seams"]
    return TrialResult(
        trial=trial,
        layout=scheduled.layout,
        color=scheduled.color,
        outcome=outcome,
        target_box=summary["target_box"],
        seam_median=round(seams["median"], 1) if seams else None,
        seam_p90=round(seams["p90"], 1) if seams else None,
        steps=summary["stats"]["steps"],
        video=summary["video"],
    )


def write_results(path: Path, results: list[TrialResult]) -> None:
    """Rewrite results.csv with all trials so far, so that a series cut short keeps them."""
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in fields(TrialResult)])
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def announce_trial(schedule: list[ScheduledTrial], i: int) -> None:
    """Print which trial comes next and what to do with the cubes while its policy loads."""
    scheduled = schedule[i]
    is_new_layout = i == 0 or schedule[i - 1].layout != scheduled.layout
    setup = "lay the cubes out anew" if is_new_layout else "put back what the arm moved"
    print(
        f"\n=== Trial {i + 1}/{len(schedule)}, layout {scheduled.layout}: {scheduled.color}. "
        f"While the policy loads, {setup}.",
        flush=True,
    )


def run_schedule(
    schedule: list[ScheduledTrial],
    policy: str,
    run_args: list[str],
    folder: Path,
    results: list[TrialResult],
) -> None:
    """Run the trials one by one, asking after each how it went, until they are done or a stop is typed.

    Appends to `results` in place, so that the caller has them if Ctrl+C or the end of the input stops this. Every
    result also goes to RESULTS_FILE in `folder`, and a tally is printed after it.
    """
    i = 0
    while i < len(schedule):
        announce_trial(schedule, i)
        scheduled = schedule[i]
        summary = run_trial(
            policy,
            scheduled.color,
            folder / f"trial-{i + 1:02d}-{scheduled.color}.json",
            run_args,
        )
        if summary is None:
            if input("No trial ran. Enter runs it again, q stops: ").strip().lower() == STOP:
                return
            continue
        code = ask_outcome()
        if code == STOP:
            return
        if code == RUN_AGAIN:
            continue
        result = make_result(
            i + 1,
            scheduled,
            OUTCOMES[code],
            summary,
        )
        results.append(result)
        write_results(folder / RESULTS_FILE, results)
        print(tally(results), flush=True)
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )  # run_policy.py's --color is not --colors
    parser.add_argument("--policy", required=True, help="Hub id or local pretrained_model folder")
    parser.add_argument(
        "--colors", default="red,blue,green,yellow", help="Comma-separated keys of tasks in config.json"
    )
    parser.add_argument(
        "--layouts",
        type=int,
        default=3,
        help="Cube layouts; each serves every colour once",
    )
    args, run_args = parser.parse_known_args()

    colors = args.colors.split(",")
    schedule = make_schedule(colors, args.layouts)
    folder = OUTPUT_FOLDER / f"series-{time.strftime(TIMESTAMP_FORMAT)}"
    folder.mkdir(parents=True)
    results_path = folder / RESULTS_FILE
    print(
        f"{len(schedule)} trials, {args.layouts} layouts x {len(colors)} colours; run_policy.py "
        f"{' '.join(run_args)}\nResults: {results_path}",
        flush=True,
    )
    results = []
    try:
        run_schedule(
            schedule,
            args.policy,
            run_args,
            folder,
            results,
        )
    except (KeyboardInterrupt, EOFError):
        print("\nSeries stopped.", flush=True)
    if results:
        print(f"\n{tally(results)}\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
