#!/usr/bin/env python3
"""Shorten the idle start of episodes: append each again from a later frame, then delete the original.

The start frame keeps `--keep-idle-s` seconds before the first frame where any joint command moves more
than 3 units from the episode's first command. The dataset is copied to camera_check/ first; frames are
read from that copy. Actions and states are copied bit for bit; images are decoded and re-encoded in
H.264 with g=2 and crf=30, as during recording. Trimmed episodes end up last, so indices shift.
"""

import argparse
import shutil
from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
from lerobot.datasets import dataset_tools
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from project_config import CONFIG_PATH, PROJECT, load_config

MOVE_THRESHOLD = 3.0
VIDEO_BACKEND = "pyav"
VIDEO_CODEC = "h264"
ENCODER_OPTIONS = {"g": "2", "crf": "30"}  # as during recording
MAX_RATE_DENOMINATOR = 1000
MAX_PIXEL_VALUE = 255
CHANNELS_LAST = (1, 2, 0)  # the axes of a decoded image, for add_frame


def frames_in_ranges(frames: Iterable[Any], frame_ranges: list[tuple[int, int]], fps: float) -> Iterator[Any]:
    """Yield the decoded frames whose number is in one of the sorted (first, end) ranges; `end` is not included."""
    range_index = 0
    for frame in frames:
        number = round(float(frame.pts * frame.time_base) * fps)
        while range_index < len(frame_ranges) and number >= frame_ranges[range_index][1]:
            range_index += 1
        if range_index >= len(frame_ranges):
            return
        if number >= frame_ranges[range_index][0]:
            yield frame


def keep_episodes_h264(input_path, output_path, episodes_to_keep, fps, vcodec=None, pix_fmt="yuv420p"):
    """dataset_tools._keep_episodes_from_video_with_av with the recording's codec settings.

    Frames are selected by frame number: the original compares float seconds, and a boundary frame at
    50.65 s read as 50.6499… was kept, shifting the next episode's video by one frame.

    `vcodec` is there for the original's callers; the output is always H.264.
    """
    frame_ranges = sorted((round(start * fps), round(end * fps)) for start, end in episodes_to_keep)
    time_base = Fraction(1, int(fps))
    with av.open(str(input_path)) as source, av.open(str(output_path), mode="w") as output:
        input_stream = source.streams.video[0]
        output_stream = output.add_stream(
            VIDEO_CODEC,
            rate=Fraction(fps).limit_denominator(MAX_RATE_DENOMINATOR),
            options=ENCODER_OPTIONS,
        )
        output_stream.width = input_stream.codec_context.width
        output_stream.height = input_stream.codec_context.height
        output_stream.pix_fmt = pix_fmt
        output_stream.time_base = time_base
        kept_frames = frames_in_ranges(source.decode(input_stream), frame_ranges, fps)
        for new_pts, frame in enumerate(kept_frames):
            new_frame = frame.reformat(width=output_stream.width, height=output_stream.height, format=pix_fmt)
            new_frame.pts = new_pts
            new_frame.time_base = time_base
            for packet in output_stream.encode(new_frame):
                output.mux(packet)
        for packet in output_stream.encode():
            output.mux(packet)


def first_moving_frame(actions: np.ndarray) -> int | None:
    """Index of the first row of `actions` (frames, joints) that differs from the first row by over MOVE_THRESHOLD.

    None when the actions never move that far.
    """
    is_moving = np.abs(actions - actions[0]).max(axis=1) > MOVE_THRESHOLD
    if not is_moving.any():
        return None
    return int(np.argmax(is_moving))


def start_frame(dataset: LeRobotDataset, episode: int, keep_idle_s: float) -> int:
    """How many first frames to drop to leave `keep_idle_s` before the episode moves; ValueError if it never does."""
    entry = dataset.meta.episodes[episode]
    rows = range(entry["dataset_from_index"], entry["dataset_to_index"])
    actions = np.stack(dataset.hf_dataset.select(rows)["action"])
    first_move = first_moving_frame(actions)
    if first_move is None:
        raise ValueError(f"Episode {episode} never moves")
    return max(0, first_move - round(keep_idle_s * dataset.meta.fps))


def are_distinct_indices(episodes: list[int], total: int) -> bool:
    """Whether no episode repeats and all of them are valid indices of a dataset of `total` episodes."""
    return len(set(episodes)) == len(episodes) and set(episodes) <= set(range(total))


def print_cuts(dataset: LeRobotDataset, cuts: dict[int, int]) -> None:
    for episode, cut in cuts.items():
        length = dataset.meta.episodes[episode]["length"]
        print(f"Episode {episode}: drop first {cut} of {length} frames ({cut / dataset.meta.fps:.1f} s)", flush=True)


def back_up_dataset(root: Path, total: int, episodes: list[int]) -> Path:
    """Copy the dataset to camera_check/ and return the copy; FileExistsError if it is already there."""
    episode_list = "-".join(map(str, episodes))
    backup = PROJECT / "camera_check" / f"backup-{root.name}-{total}ep-before-trim-{episode_list}"
    if backup.exists():
        raise FileExistsError(f"{backup} already exists")
    shutil.copytree(root, backup)
    print(f"Backup: {backup}", flush=True)
    return backup


def frame_to_append(item: dict[str, Any], task: str, video_keys: list[str]) -> dict[str, Any]:
    """Turn a frame read from a dataset into what add_frame takes: images as uint8 (height, width, channels)."""
    frame = {"observation.state": item["observation.state"], "action": item["action"], "task": task}
    for key in video_keys:
        frame[key] = (item[key].permute(CHANNELS_LAST).numpy() * MAX_PIXEL_VALUE).round().astype(np.uint8)
    return frame


def append_trimmed_episodes(source: LeRobotDataset, target: LeRobotDataset, cuts: dict[int, int]) -> None:
    """Append each episode of `cuts` ({episode: first frames to drop}) of `source` to the end of `target`."""
    video_keys = source.meta.video_keys
    for episode, cut in cuts.items():
        entry = source.meta.episodes[episode]
        task = entry["tasks"][0]
        for index in range(entry["dataset_from_index"] + cut, entry["dataset_to_index"]):
            target.add_frame(frame_to_append(source[index], task, video_keys))
        target.save_episode()
        print(f"Episode {episode} appended as {target.meta.total_episodes - 1}", flush=True)
    target.finalize()


def delete_original_episodes(repo_id: str, root: Path, episodes: list[int]) -> None:
    dataset_tools._keep_episodes_from_video_with_av = keep_episodes_h264
    staging = root.with_name(root.name + ".trimmed")
    dataset_tools.delete_episodes(
        LeRobotDataset(repo_id, root=root, video_backend=VIDEO_BACKEND),
        episodes,
        output_dir=staging,
        repo_id=repo_id,
    )
    shutil.rmtree(root)  # its pre-trim state is in the backup
    staging.rename(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument("--keep-idle-s", type=float, default=3.5)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be cut; change nothing")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["dataset_root"])
    repo_id = config["repo_id"]
    dataset = LeRobotDataset(repo_id, root=root, video_backend=VIDEO_BACKEND)
    total = dataset.meta.total_episodes
    if not are_distinct_indices(args.episodes, total):
        parser.error(f"Episodes must be distinct indices below {total}")
    cuts = {episode: start_frame(dataset, episode, args.keep_idle_s) for episode in args.episodes}
    print_cuts(dataset, cuts)
    if args.dry_run:
        return

    backup = back_up_dataset(root, total, args.episodes)
    source = LeRobotDataset(repo_id, root=backup, video_backend=VIDEO_BACKEND)
    target = LeRobotDataset(
        repo_id,
        root=root,
        video_backend=VIDEO_BACKEND,
        vcodec=VIDEO_CODEC,
        batch_encoding_size=1,
    )
    append_trimmed_episodes(source, target, cuts)

    delete_original_episodes(repo_id, root, args.episodes)
    episodes_left = LeRobotDataset(repo_id, root=root).meta.total_episodes
    print(f"Deleted originals {args.episodes}; dataset now has {episodes_left} episodes", flush=True)


if __name__ == "__main__":
    main()
