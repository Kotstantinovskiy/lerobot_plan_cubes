#!/usr/bin/env python3
"""Let a SmolVLA or pi0.5 policy drive the follower inside a safety envelope; nothing is added to the dataset.

A trained checkpoint normalizes with the statistics saved in it, those of the data it trained on, whatever dataset lies
here now; a base policy from the Hub uses our dataset's. Every command is clipped to the joint range
seen in our demonstrations and to `--max-step` per frame. When the run ends, after Ctrl+C or an error,
the follower folds back to `rest_pose` from config.json (`--no-rest` skips it; a second Ctrl+C drops
torque at once), then torque is disabled. A side-by-side top/wrist video of the run is saved under camera_check/.

A policy trained with target boxes (target_box.json next to its weights) gets the same box: before the arm
moves, the cube of the commanded colour is found on the top camera and its box is drawn on every top frame.
A policy trained on the cropped top view (top_crop.json) gets the same crop; the video keeps the full frame with the
crop outlined in grey.

By default a chunk is computed when the previous one has run out (`--n-action-steps` actions), and the arm stands
still for the ~1.2 s that takes on a Mac. With `--replan K` the next chunk is computed in the background once K actions
of the current one have run, from the observation at that moment; when it is ready, the actions for the steps that
passed meanwhile are dropped and the arm switches to it without stopping. Replanning starts once the arm has moved
from its starting pose: the demonstrations keep 3.5 s of stillness before the first motion, so from a still arm the
policy plans ~2 s of waiting before it moves, and replanning from still frames would only ever run the wait.

A chunk computed in the background is a fresh sample that knows nothing of the one the arm is running: where it takes
over, the command jumps (on recorded episodes 2-3 times a normal step) and now and then turns to another decision.
With `--rtc` (Real-Time Chunking, for flow-matching policies such as SmolVLA) the background chunk is guided to agree
with what is left of the running one: its actions for the steps that pass while it is computed are held to the running
chunk's, and the following ones, up to `--rtc-horizon`, blend from those into its own. The run ends with the seams: the
largest joint change where a background chunk took over.

With `--server HOST:PORT` the chunks come from inference/policy_server.py on a GPU machine, and everything else stays
here. The joint envelope and the box and crop settings come from the server too, so this machine needs neither a
checkpoint nor a dataset, and `--policy` and `--device` have no effect.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import math
from pathlib import Path
import sys
import time

import av
import cv2
import draccus
import numpy as np
from safetensors.numpy import load_file
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.constants import OBS_IMAGES, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
from record import PROJECT, load_config, positive_int  # noqa: E402
from rest import fold_to_rest  # noqa: E402
import cube_boxes  # noqa: E402  (inference/ is the script's own directory)
import pi05_inputs  # noqa: E402
from remote_policy import RemotePolicy  # noqa: E402
import top_crop  # noqa: E402

# Our camera keys -> the keys the base policy was pretrained with, as in training/train.py.
RENAME = {"observation.images.top": "observation.images.camera1",  # SmolVLA: camera1 = top, camera2 = wrist
          "observation.images.wrist": "observation.images.camera2"}
PI05_RENAME = {"observation.images.top": "observation.images.base_0_rgb",
               "observation.images.wrist": "observation.images.left_wrist_0_rgb"}


def target_box_style(policy_path):
    """Drawing style if the policy was trained with target boxes, else None."""
    path = Path(policy_path) / "target_box.json"
    return json.loads(path.read_text()) if path.is_file() else None


def find_target(top_rgb, color):
    if top_rgb.shape[:2] != (720, 1280):
        raise RuntimeError(f"Target boxes are tuned for a 1280x720 top camera, got {top_rgb.shape[1]}x{top_rgb.shape[0]}")
    boxes = cube_boxes.detect(top_rgb)
    if boxes[color] is None:
        raise RuntimeError(f"No {color} cube found on the top camera; nothing to mark as the target")
    found = ", ".join(f"{c} {b}" for c, b in boxes.items() if b is not None)
    print(f"Target box: {color} {boxes[color]} (found: {found})", flush=True)
    return boxes[color]


def saved_stats(policy_path):
    """The normalization statistics saved in a trained checkpoint ({feature: {stat: array}}), None for a Hub id.
    A dataset that has grown since training has other ones: 88 and 108 episodes differ by up to 4.8 joint units."""
    files = sorted(Path(policy_path).glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not files:
        return None
    stats = {}
    for key, values in load_file(files[0]).items():
        feature, stat = key.rsplit(".", 1)
        stats.setdefault(feature, {})[stat] = values
    return stats


def load_policy(path, device, n_action_steps, ds_meta, rtc=None):
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    cfg.device = device
    cfg.n_action_steps = n_action_steps
    if rtc is not None:
        cfg.rtc_config = rtc
    if hasattr(cfg, "load_vlm_weights"):
        cfg.load_vlm_weights = False  # the policy checkpoint already holds every VLM weight
    is_pi05 = cfg.type == "pi05"
    if is_pi05:
        rename = PI05_RENAME
    else:
        rename = RENAME if "observation.images.camera1" in cfg.input_features else {}
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map=rename)
    if rtc is not None and getattr(policy, "rtc_processor", None) is None:
        raise RuntimeError(f"{cfg.type} has no Real-Time Chunking; it is for flow-matching policies such as SmolVLA")
    policy.eval()
    features = {**policy.config.input_features, **policy.config.output_features}
    stats = saved_stats(path) or ds_meta.stats
    preprocessor_overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": rename},
        "normalizer_processor": {"stats": stats, "features": features,
                                 "norm_map": policy.config.normalization_mapping},
    }
    if is_pi05:  # the checkpoint names no tokenizer, training was given one
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer": pi05_inputs.load_tokenizer()}
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=path, dataset_stats=stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides={
            "unnormalizer_processor": {"stats": stats, "features": policy.config.output_features,
                                       "norm_map": policy.config.normalization_mapping},
        },
    )
    if is_pi05:
        pre = resize_cameras_first(pre)
    return policy, pre, post


def resize_cameras_first(pre):
    """pi0.5's preprocessor behind a step that shrinks the camera frames as in training (inference/pi05_inputs.py)."""
    def resized_pre(observation):
        frames = {key: pi05_inputs.resize(value) for key, value in observation.items() if key.startswith(OBS_IMAGES)}
        return pre({**observation, **frames})

    return resized_pre


def compute_chunk(policy, pre, post, frame, device, task, robot_type, **guidance):
    """The whole action chunk for a frame (chunk_size x joints): in dataset units and as the model gave it (normalized,
    what RTC compares). `guidance`: RTC's inference_delay and prev_chunk_left_over."""
    with torch.no_grad():  # not inference_mode: RTC's guidance takes a gradient, which that mode rules out
        batch = pre(prepare_observation_for_inference(dict(frame), device, task, robot_type))
        raw = policy.predict_action_chunk(batch, **guidance)
        chunk = post(raw.clone())
    return chunk[0].float().cpu().numpy().astype(float), raw[0]


def dataset_observation(ds_meta, root, episode, offset_s):
    """A recorded frame in the same format as robot.get_observation(), for --dry-run."""
    dataset = LeRobotDataset(ds_meta.repo_id, root=root, video_backend="pyav")
    # round: an offset worked out as frames / fps can come back a hair under its frame
    item = dataset[int(ds_meta.episodes[episode]["dataset_from_index"]) + round(offset_s * ds_meta.fps)]
    obs = dict(zip(ds_meta.features["observation.state"]["names"], item["observation.state"].tolist()))
    for key in ("top", "wrist"):
        image = item[f"observation.images.{key}"].permute(1, 2, 0).numpy()
        obs[key] = (image * 255).round().astype(np.uint8)
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--policy", default="lerobot/smolvla_base", help="Hub id or local pretrained_model folder")
    parser.add_argument("--color", required=True, help="Key from tasks in config.json")
    parser.add_argument("--seconds", type=positive_int, default=20)
    # 2.0 stalled the elbow against gravity (the per-frame position error was too small to move it).
    parser.add_argument("--max-step", type=float, default=5.0, help="Max change per frame per joint (normalized units)")
    parser.add_argument("--n-action-steps", type=positive_int, default=50,
                        help="Actions run from each chunk before the next one (without --replan)")
    parser.add_argument("--replan", type=positive_int,
                        help="Compute the next chunk in the background after this many actions of the current one")
    parser.add_argument("--rtc", action="store_true",
                        help="Guide each background chunk to continue the running one (Real-Time Chunking)")
    parser.add_argument("--rtc-horizon", type=positive_int, default=20,
                        help="Actions of the running chunk the next one is guided to agree with; keep it above the "
                             "steps one inference takes (~14 on a Mac)")
    parser.add_argument("--rtc-weight", type=float, default=10.0,
                        help="Upper bound of the RTC guidance; 10 suits SmolVLA's 10 denoising steps")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--server", help="HOST:PORT of policy_server.py, to compute chunks there instead of here")
    parser.add_argument("--dry-run", action="store_true", help="Run one inference on a recorded frame; no robot")
    parser.add_argument("--no-rest", action="store_true", help="Leave the arm where the run ends instead of folding it")
    parser.add_argument("--wait", action="store_true",
                        help="Once the policy is loaded, wait for Enter before the arm is powered (run_series.py)")
    parser.add_argument("--summary", type=Path, help="Also write the run's summary to this JSON file")
    args = parser.parse_args()
    if args.rtc and not (args.replan or args.dry_run):
        parser.error("--rtc guides the chunks computed in the background; add --replan")

    config = load_config(args.config)
    task = config["tasks"][args.color]
    remote = RemotePolicy(args.server) if args.server else None
    if remote:
        if args.rtc and not remote.info["rtc"]:
            parser.error("--rtc needs policy_server.py started with --rtc")
        ds_meta = None
        features, fps = remote.info["features"], remote.info["fps"]
        low, high = (np.asarray(remote.info[k], dtype=float) for k in ("low", "high"))
        envelope_source, policy_name = "the policy's training data", f"{remote.info['policy']} at {args.server}"
    else:
        ds_meta = LeRobotDatasetMetadata(config["repo_id"], root=config["dataset_root"])
        features, fps = ds_meta.features, ds_meta.fps
        low, high = (np.asarray(ds_meta.stats["action"][k], dtype=float) for k in ("min", "max"))
        envelope_source, policy_name = f"{ds_meta.total_episodes} episodes", f"{args.policy} on {args.device}"
    names = features["action"]["names"]
    print(f"Task: {task}\nPolicy: {policy_name}\nJoint envelope from {envelope_source}: " + ", ".join(f"{n} [{lo:.0f}, {hi:.0f}]" for n, lo, hi in zip(names, low, high)), flush=True)
    robot = None
    if not args.dry_run:  # build the robot first so configuration errors surface before the slow model load
        fields = dict(config["robot"], max_relative_target=args.max_step)
        robot = make_robot_from_config(draccus.decode(RobotConfig, fields))
        if set(robot.calibration) != set(robot.bus.motors):
            raise RuntimeError(f"Missing or incomplete follower calibration: {robot.calibration_fpath}")
        # Torque_Enable occasionally gets no status packet right after the cameras start (seen on ids 4 and 6);
        # rewriting it is idempotent, so allow a few retries instead of aborting the connection. The first
        # Present_Position read after connecting fails the same way now and then.
        robot.bus.enable_torque = functools.partial(robot.bus.enable_torque, num_retry=3)
        robot.bus.sync_read = functools.partial(robot.bus.sync_read, num_retry=3)
    if remote:
        box_style, crop = remote.info["target_box"], remote.info["top_crop"]
    else:
        rtc = RTCConfig(enabled=True, execution_horizon=args.rtc_horizon, max_guidance_weight=args.rtc_weight,
                        prefix_attention_schedule=RTCAttentionSchedule.EXP) if args.rtc else None
        policy, pre, post = load_policy(args.policy, args.device, args.n_action_steps, ds_meta, rtc)
        device = get_safe_torch_device(args.device)
        box_style, crop = target_box_style(args.policy), top_crop.load(args.policy)
    box = None
    if crop:
        print(f"Top camera cropped to {crop['box']} (x0, y0, x1, y1)", flush=True)

    def observe(obs):
        """Model input from a robot observation; the box is drawn in place, so the video shows it too."""
        if box is not None:
            cube_boxes.draw_rgb(obs["top"], box, box_style)
        seen = obs if crop is None else {**obs, "top": top_crop.crop_hwc(obs["top"], crop)}
        # Copies: with --replan the frame is read by the background inference while the loop draws on the next one.
        seen = {k: np.array(v) if isinstance(v, np.ndarray) else v for k, v in seen.items()}
        return build_dataset_frame(features, seen, prefix=OBS_STR)

    def plan(frame, **guidance):
        """compute_chunk's two forms of the action chunk, here or on the server, and the seconds it took."""
        if remote:
            return remote.plan(frame, task, **guidance)
        start = time.perf_counter()
        chunk, raw = compute_chunk(policy, pre, post, frame, device, task, ds_meta.robot_type, **guidance)
        return chunk, raw, time.perf_counter() - start

    def recorded_observation(offset_s):
        if remote:
            return remote.recorded_observation(episode=1, offset_s=offset_s)
        return dataset_observation(ds_meta, config["dataset_root"], episode=1, offset_s=offset_s)

    if args.dry_run:
        obs = recorded_observation(3)
        if box_style:
            box = find_target(obs["top"], args.color)
        chunk, _, infer = plan(observe(obs))
        clipped = int((np.clip(chunk[0], low, high) != chunk[0]).sum())
        print(f"Dry run: one inference {infer:.2f}s, {clipped} joints clipped.\n"
              f"  state  {np.round([obs[n] for n in names], 1).tolist()}\n  action {np.round(chunk[0], 1).tolist()}",
              flush=True)
        if args.rtc:  # a chunk taking over from another as with --replan: planned `delay` steps on, run from there
            def frame_at(offset_s):
                return observe(recorded_observation(offset_s))

            first, first_raw, infer = plan(frame_at(6))  # the arm is moving by then
            delay = min(math.ceil(infer * fps), len(first) // 2)
            later = frame_at(6 + delay / fps)
            plain = plan(later)[0]
            guided, _, infer = plan(later, inference_delay=delay, prev_chunk_left_over=first_raw[delay:])
            seams = [np.abs(c[delay] - first[2 * delay - 1]).max() for c in (plain, guided)]
            print(f"RTC check on frames {delay} steps apart: seam {seams[0]:.1f} without, {seams[1]:.1f} with (largest "
                  f"joint change where the later chunk takes over); guided inference {infer:.2f}s", flush=True)
        return

    if args.wait:
        try:
            input("Policy loaded. Set up the cubes, then press Enter to start (Ctrl+C cancels): ")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled before the arm moved.", flush=True)
            return
    video_path = PROJECT / "camera_check" / f"policy-run-{time.strftime('%Y%m%d-%H%M%S')}-{args.color}.mp4"
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=fps, options={"crf": "28"})
    stream.width, stream.height, stream.pix_fmt = 1280, 360, "yuv420p"
    stats = {"steps": 0, "chunks": 0, "max_infer_s": 0.0, "clipped_steps": 0, "late_steps": 0, "dropped_actions": 0}
    seams = []  # largest joint change where a background chunk took over
    background = ThreadPoolExecutor(max_workers=1) if args.replan else None
    try:
        # With torque still off: check calibration and replace any stale goal, so enabling torque cannot jump.
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Calibration mismatch; refusing automatic calibration")
        if any(robot.bus.sync_read("Torque_Enable").values()):
            raise RuntimeError("Expected motors at rest with torque disabled")
        robot.bus.sync_write("Goal_Position", robot.bus.sync_read("Present_Position"))
        robot.bus.disconnect(disable_torque=False)
        robot.connect(calibrate=False)
        if box_style:  # the arm holds its pose; the box is found once, before it moves, as in the recordings
            box = find_target(robot.get_observation()["top"], args.color)
        replanning = (f"next chunk in the background after {args.replan} actions" if args.replan
                      else f"{args.n_action_steps} actions per chunk")
        if args.rtc:
            replanning += f", RTC over {args.rtc_horizon} actions, guidance up to {args.rtc_weight:g}"
        print(f"Policy controls the arm for {args.seconds} s (max {args.max_step} units/frame, {replanning}). "
              f"Ctrl+C stops.", flush=True)
        queue, ran, pending = [], 0, None  # actions left, steps since the chunk's observation, (job, its step)
        left = None  # the queue's actions as the model gave them (normalized), which RTC holds the next chunk to
        delays = deque(maxlen=5)  # steps the last background chunks took to come
        first_pose, moved = None, False
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds:
            tick = time.perf_counter()
            obs = robot.get_observation()
            pose = np.array([obs[n] for n in names])
            first_pose = pose if first_pose is None else first_pose
            moved = moved or np.abs(pose - first_pose).max() > 3  # the motion threshold of recording/trim_start.py
            frame = observe(obs)
            chunk = None
            if pending is not None and (pending[0].done() or not queue):  # waits only if the current chunk ran out
                (chunk, raw, infer), taken_at = pending[0].result(), pending[1]
                pending = None
                # The actions for the steps run since its observation are stale; the last one is kept in any case.
                ran = min(stats["steps"] - taken_at, len(chunk) - 1)
                delays.append(ran)
                chunk, raw = chunk[ran:], raw[ran:]
                stats["dropped_actions"] += ran
                seams.append(np.abs(chunk[0] - values).max())  # values: the action sent last step
            elif not queue:  # every chunk without --replan, and with it until the arm has moved
                chunk, raw, infer = plan(frame)
                ran = 0
                chunk, raw = chunk[:args.n_action_steps], raw[:args.n_action_steps]
            if chunk is not None:
                queue, left = list(chunk), raw
                stats["chunks"] += 1
                stats["max_infer_s"] = max(stats["max_infer_s"], infer)
            if args.replan and moved and pending is None and ran >= args.replan:
                guidance = {}
                if args.rtc:  # the new chunk's first actions are held to those the arm runs while it is computed
                    guidance = {"inference_delay": max(delays, default=math.ceil(infer * fps)),
                                "prev_chunk_left_over": left}
                pending = (background.submit(plan, frame, **guidance), stats["steps"])
            values = queue.pop(0)
            left = left[1:]
            ran += 1
            clipped = np.clip(values, low, high)
            robot.send_action(dict(zip(names, clipped.tolist())))
            stats["steps"] += 1
            stats["clipped_steps"] += bool((clipped != values).any())
            if crop is not None:  # after act(): only the video shows the outline
                x0, y0, x1, y1 = crop["box"]
                cv2.rectangle(obs["top"], (x0, y0), (x1 - 1, y1 - 1), (128, 128, 128), 2)
            side = np.hstack([cv2.resize(obs[k], (640, 360), interpolation=cv2.INTER_AREA) for k in ("top", "wrist")])
            for packet in stream.encode(av.VideoFrame.from_ndarray(side, format="rgb24")):
                container.mux(packet)
            if stats["steps"] % fps == 0:
                print(f"{time.perf_counter() - start:5.1f}s step {stats['steps']}, new chunks {stats['chunks']}, "
                      f"clipped steps {stats['clipped_steps']}", flush=True)
            elapsed = time.perf_counter() - tick
            stats["late_steps"] += elapsed > 1 / fps
            precise_sleep(max(0.0, 1 / fps - elapsed))
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.", flush=True)
    finally:
        if robot.bus.is_connected:  # robot.is_connected is also False when only a camera has failed
            try:
                if not args.no_rest and all(robot.bus.sync_read("Torque_Enable").values()):
                    print("Folding back to rest; Ctrl+C drops torque at once.", flush=True)
                    fold_to_rest(robot.bus, config["rest_pose"], fps, log=lambda s: print(s, flush=True))
            except (Exception, KeyboardInterrupt) as error:
                print(f"Return to rest stopped: {error!r}", flush=True)
            try:
                robot.disconnect()
            except Exception:
                robot.bus.disconnect(disable_torque=True)
            print("Follower disconnected; torque disabled.", flush=True)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        if background is not None:
            background.shutdown(cancel_futures=True)
        seam_stats = {"count": len(seams), "median": float(np.median(seams)),
                      "p90": float(np.percentile(seams, 90)), "max": float(max(seams))} if seams else None
        if seam_stats:
            print(f"Seams over {len(seams)} chunk changes: median {seam_stats['median']:.1f}, 90th percentile "
                  f"{seam_stats['p90']:.1f}, max {seam_stats['max']:.1f} units", flush=True)
        print(f"Summary: {stats}\nVideo: {video_path}", flush=True)
        if args.summary:
            args.summary.write_text(json.dumps({"color": args.color, "target_box": box, "stats": stats,
                                                "seams": seam_stats, "video": str(video_path), "args": vars(args)},
                                               indent=1, default=str))


if __name__ == "__main__":
    main()
