#!/usr/bin/env python3
"""Shorten the idle start of episodes: append each again from a later frame, then delete the original.

The start frame keeps `--keep-idle-s` seconds before the first frame where any joint command moves more
than 3 units from the episode's first command. The dataset is copied to camera_check/ first; frames are
read from that copy. Actions and states are copied bit for bit; images are decoded and re-encoded in
H.264 with g=2 and crf=30, as during recording. Trimmed episodes end up last, so indices shift.
"""

import argparse
from fractions import Fraction
from pathlib import Path
import shutil

import av
import numpy as np

from lerobot.datasets import dataset_tools
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from record import load_config

PROJECT = Path(__file__).resolve().parents[1]
MOVE_THRESHOLD = 3.0


def keep_episodes_h264(input_path, output_path, episodes_to_keep, fps, vcodec=None, pix_fmt="yuv420p"):
    """dataset_tools._keep_episodes_from_video_with_av with the recording's codec settings.

    Frames are selected by frame number: the original compares float seconds, and a boundary frame at
    50.65 s read as 50.6499… was kept, shifting the next episode's video by one frame.
    """
    ranges = sorted((round(start * fps), round(end * fps)) for start, end in episodes_to_keep)
    with av.open(str(input_path)) as src, av.open(str(output_path), mode="w") as out:
        v_in = src.streams.video[0]
        v_out = out.add_stream("h264", rate=Fraction(fps).limit_denominator(1000), options={"g": "2", "crf": "30"})
        v_out.width, v_out.height, v_out.pix_fmt = v_in.codec_context.width, v_in.codec_context.height, pix_fmt
        v_out.time_base = Fraction(1, int(fps))
        count, index = 0, 0
        for frame in src.decode(v_in):
            number = round(float(frame.pts * frame.time_base) * fps)
            while index < len(ranges) and number >= ranges[index][1]:
                index += 1
            if index >= len(ranges):
                break
            if number < ranges[index][0]:
                continue
            new = frame.reformat(width=v_out.width, height=v_out.height, format=pix_fmt)
            new.pts, new.time_base = count, Fraction(1, int(fps))
            for packet in v_out.encode(new):
                out.mux(packet)
            count += 1
        for packet in v_out.encode():
            out.mux(packet)


def start_frame(dataset, episode, keep_idle_s):
    entry = dataset.meta.episodes[episode]
    actions = np.stack(dataset.hf_dataset.select(range(entry["dataset_from_index"], entry["dataset_to_index"]))["action"])
    moving = np.abs(actions - actions[0]).max(axis=1) > MOVE_THRESHOLD
    if not moving.any():
        raise ValueError(f"Episode {episode} never moves")
    return max(0, int(np.argmax(moving)) - round(keep_idle_s * dataset.meta.fps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--keep-idle-s", type=float, default=3.5)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be cut; change nothing")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["dataset_root"])
    repo_id = config["repo_id"]
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    total = dataset.meta.total_episodes
    if len(set(args.episodes)) != len(args.episodes) or not set(args.episodes) <= set(range(total)):
        parser.error(f"Episodes must be distinct indices below {total}")
    cuts = {ep: start_frame(dataset, ep, args.keep_idle_s) for ep in args.episodes}
    for ep, cut in cuts.items():
        length = dataset.meta.episodes[ep]["length"]
        print(f"Episode {ep}: drop first {cut} of {length} frames ({cut / dataset.meta.fps:.1f} s)", flush=True)
    if args.dry_run:
        return

    backup = PROJECT / "camera_check" / f"backup-{root.name}-{total}ep-before-trim-{'-'.join(map(str, args.episodes))}"
    if backup.exists():
        raise FileExistsError(f"{backup} already exists")
    shutil.copytree(root, backup)
    print(f"Backup: {backup}", flush=True)
    source = LeRobotDataset(repo_id, root=backup, video_backend="pyav")
    target = LeRobotDataset(repo_id, root=root, video_backend="pyav", vcodec="h264", batch_encoding_size=1)
    video_keys = source.meta.video_keys
    for ep, cut in cuts.items():
        entry = source.meta.episodes[ep]
        task = entry["tasks"][0]
        for index in range(entry["dataset_from_index"] + cut, entry["dataset_to_index"]):
            item = source[index]
            frame = {"observation.state": item["observation.state"], "action": item["action"], "task": task}
            for key in video_keys:
                frame[key] = (item[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            target.add_frame(frame)
        target.save_episode()
        print(f"Episode {ep} appended as {target.meta.total_episodes - 1}", flush=True)
    target.finalize()

    dataset_tools._keep_episodes_from_video_with_av = keep_episodes_h264
    staging = root.with_name(root.name + ".trimmed")
    dataset_tools.delete_episodes(LeRobotDataset(repo_id, root=root, video_backend="pyav"),
                                  args.episodes, output_dir=staging, repo_id=repo_id)
    shutil.rmtree(root)  # its pre-trim state is in the backup
    staging.rename(root)
    print(f"Deleted originals {args.episodes}; dataset now has {LeRobotDataset(repo_id, root=root).meta.total_episodes} "
          f"episodes", flush=True)


if __name__ == "__main__":
    main()
