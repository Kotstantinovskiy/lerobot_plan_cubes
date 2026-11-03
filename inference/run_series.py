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
from pathlib import Path
import signal
import subprocess
import sys
import time

INFERENCE = Path(__file__).resolve().parent
PROJECT = INFERENCE.parent
OUTCOMES = {"p": "placed", "g": "grasped", "m": "missed", "w": "wrong cube"}
PROMPT = "Outcome: p placed, g grasped, m missed, w wrong cube; r run again, q stop: "


def tally(rows):
    """Placed, grasped (placed or not) and wrong-cube counts over all trials, then per colour."""
    def counts(subset):
        def number(*outcomes):
            return sum(row["outcome"] in outcomes for row in subset)

        n = len(subset)
        return f"placed {number('placed')}/{n}, grasped {number('placed', 'grasped')}/{n}, " \
               f"wrong cube {number('wrong cube')}/{n}"

    colors = dict.fromkeys(row["color"] for row in rows)
    return "\n".join([f"All: {counts(rows)}"]
                     + [f"  {color}: {counts([row for row in rows if row['color'] == color])}" for color in colors])


def run_trial(policy, color, summary, run_args):
    """Runs run_policy.py; its summary, or None if no trial ran (cancelled, or failed before the arm moved)."""
    summary.unlink(missing_ok=True)
    child = subprocess.Popen([sys.executable, str(INFERENCE / "run_policy.py"), "--policy", policy, "--color", color,
                              "--wait", "--summary", str(summary), *run_args], cwd=PROJECT)
    # Ctrl+C reaches the trial too, which folds the arm back; here it is ignored. Only once the trial has started:
    # it would inherit an ignored Ctrl+C.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        child.wait()
    finally:
        signal.signal(signal.SIGINT, previous)
    if not summary.exists():
        return None
    result = json.loads(summary.read_text())
    return result if result["stats"]["steps"] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)  # run_policy.py's --color is not --colors
    parser.add_argument("--policy", required=True, help="Hub id or local pretrained_model folder")
    parser.add_argument("--colors", default="red,blue,green,yellow",
                        help="Comma-separated keys of tasks in config.json")
    parser.add_argument("--layouts", type=int, default=3, help="Cube layouts; each serves every colour once")
    args, run_args = parser.parse_known_args()
    colors = args.colors.split(",")
    schedule = [(layout, colors[(layout - 1 + k) % len(colors)])
                for layout in range(1, args.layouts + 1) for k in range(len(colors))]
    out = PROJECT / "camera_check" / f"series-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    results = out / "results.csv"
    print(f"{len(schedule)} trials, {args.layouts} layouts x {len(colors)} colours; run_policy.py "
          f"{' '.join(run_args)}\nResults: {results}", flush=True)
    rows, i = [], 0
    try:
        while i < len(schedule):
            layout, color = schedule[i]
            new_layout = i == 0 or schedule[i - 1][0] != layout
            setup = "lay the cubes out anew" if new_layout else "put back what the arm moved"
            print(f"\n=== Trial {i + 1}/{len(schedule)}, layout {layout}: {color}. While the policy loads, {setup}.",
                  flush=True)
            result = run_trial(args.policy, color, out / f"trial-{i + 1:02d}-{color}.json", run_args)
            if result is None:
                if input("No trial ran. Enter runs it again, q stops: ").strip().lower() == "q":
                    break
                continue
            while (code := input(PROMPT).strip().lower()) not in (*OUTCOMES, "r", "q"):
                pass
            if code == "q":
                break
            if code == "r":
                continue
            seams = result["seams"]
            rows.append({"trial": i + 1, "layout": layout, "color": color, "outcome": OUTCOMES[code],
                         "target_box": result["target_box"],
                         "seam_median": seams and round(seams["median"], 1),
                         "seam_p90": seams and round(seams["p90"], 1),
                         "steps": result["stats"]["steps"], "video": result["video"]})
            with results.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(tally(rows), flush=True)
            i += 1
    except (KeyboardInterrupt, EOFError):
        print("\nSeries stopped.", flush=True)
    if rows:
        print(f"\n{tally(rows)}\nResults: {results}", flush=True)


if __name__ == "__main__":
    main()
