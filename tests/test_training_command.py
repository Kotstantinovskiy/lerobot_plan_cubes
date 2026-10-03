"""The training command line and environment of training/train.py, pinned without starting anything.

subprocess.call is replaced, the policy and dataset folders are small fakes in a temporary directory, and neither the
network nor a GPU is touched.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import train  # noqa: E402

RUN_NAME = "pinned-run"
CONFIG = {"repo_id": "local/color_cubes", "dataset_root": "/data/color_cubes"}
TRAINER_SCRIPT = str(Path(train.__file__).with_name("lerobot_train_fast.py"))
SMOLVLA_RENAME_MAP = (
    '{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
)
PI05_RENAME_MAP = (
    '{"observation.images.top": "observation.images.base_0_rgb", '
    '"observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
)
IMAGE_TRANSFORMS_JSON = (
    '{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.5, 1.5]}}, '
    '"contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.6, 1.4]}}, '
    '"saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}, '
    '"lighting": {"weight": 1.0, "type": "LightingGradient", "kwargs": {"gain": [0.6, 1.4]}}, '
    '"affine": {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}}}'
)


def make_args(**overrides) -> argparse.Namespace:
    """What build_command reads from the command line: a new SmolVLA run with its defaults, unless overridden."""
    values = {
        "policy": Path("/policies/smolvla_base"),
        "device": "cuda",
        "steps": 20_000,
        "batch_size": 64,
        "save_freq": 2_500,
        "log_freq": 100,
        "num_workers": 8,
        "seed": 1000,
        "video_backend": "pyav",
        "no_augment": False,
        "episodes": None,
        "full_finetune": False,
        "wandb": False,
        "gpu": None,
        "precision": "bf16",
        "resume": False,
        "run_name": RUN_NAME,
        "extra": [],
    }
    return argparse.Namespace(**{**values, **overrides})


def make_pi05_args() -> argparse.Namespace:
    return make_args(policy=Path("/policies/pi05_base"), steps=40_000, batch_size=16, save_freq=5_000)


def new_run_options(*, policy_folder, dataset_root, rename_map, batch_size, steps, save_freq, output_dir) -> list[str]:
    """The options every new run gets, in command line order."""
    return [
        f"--policy.path={policy_folder}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.scheduler_decay_steps={steps}",
        f"--dataset.repo_id={CONFIG['repo_id']}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=true",
        f"--dataset.image_transforms.tfs={IMAGE_TRANSFORMS_JSON}",
        f"--rename_map={rename_map}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        "--log_freq=100",
        "--num_workers=8",
        "--seed=1000",
        f"--output_dir={output_dir}",
        f"--job_name={RUN_NAME}",
        "--wandb.enable=false",
    ]


def smolvla_options(*, policy_folder, dataset_root, output_dir) -> list[str]:
    base_options = new_run_options(
        policy_folder=policy_folder,
        dataset_root=dataset_root,
        rename_map=SMOLVLA_RENAME_MAP,
        batch_size=64,
        steps=20000,
        save_freq=2500,
        output_dir=output_dir,
    )
    return [*base_options, "--policy.load_vlm_weights=false"]


def pi05_options(*, policy_folder, dataset_root, output_dir) -> list[str]:
    return new_run_options(
        policy_folder=policy_folder,
        dataset_root=dataset_root,
        rename_map=PI05_RENAME_MAP,
        batch_size=16,
        steps=40000,
        save_freq=5000,
        output_dir=output_dir,
    )


class BuildCommandTests(unittest.TestCase):
    output_dir = train.PROJECT / "outputs" / "train" / RUN_NAME

    def test_new_smolvla_run_passes_every_option_in_order(self):
        command, _ = train.build_command(make_args(), CONFIG, "smolvla")
        expected_options = smolvla_options(
            policy_folder="/policies/smolvla_base",
            dataset_root=CONFIG["dataset_root"],
            output_dir=self.output_dir,
        )
        self.assertEqual(command, [sys.executable, TRAINER_SCRIPT, *expected_options])

    def test_new_pi05_run_has_its_camera_names_and_no_vlm_option(self):
        command, _ = train.build_command(make_pi05_args(), CONFIG, "pi05")
        expected_options = pi05_options(
            policy_folder="/policies/pi05_base",
            dataset_root=CONFIG["dataset_root"],
            output_dir=self.output_dir,
        )
        self.assertEqual(command, [sys.executable, TRAINER_SCRIPT, *expected_options])

    def test_output_dir_is_the_run_folder_under_outputs_train(self):
        _, output_dir = train.build_command(make_args(), CONFIG, "smolvla")
        self.assertEqual(output_dir, self.output_dir)

    def test_resume_passes_only_the_saved_config(self):
        command, _ = train.build_command(make_args(resume=True), CONFIG, "smolvla")
        saved_config = self.output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        self.assertEqual(command, [sys.executable, TRAINER_SCRIPT, f"--config_path={saved_config}", "--resume=true"])

    def test_no_augment_turns_the_transforms_off(self):
        command, _ = train.build_command(make_args(no_augment=True), CONFIG, "smolvla")
        self.assertIn("--dataset.image_transforms.enable=false", command)

    def test_wandb_turns_logging_on(self):
        command, _ = train.build_command(make_args(wandb=True), CONFIG, "smolvla")
        self.assertIn("--wandb.enable=true", command)

    def test_optional_options_follow_the_base_ones_in_order(self):
        args = make_args(episodes=[0, 1, 2], full_finetune=True, extra=["--policy.gradient_checkpointing=true"])
        command, _ = train.build_command(args, CONFIG, "smolvla")
        expected_end = [
            "--wandb.enable=false",
            "--policy.load_vlm_weights=false",
            "--dataset.episodes=[0, 1, 2]",
            "--policy.train_expert_only=false",
            "--policy.freeze_vision_encoder=false",
            "--policy.gradient_checkpointing=true",
        ]
        self.assertEqual(command[-6:], expected_end)

    def test_extra_arguments_follow_the_resume_options(self):
        command, _ = train.build_command(make_args(resume=True, extra=["--steps=50000"]), CONFIG, "smolvla")
        self.assertEqual(command[-2:], ["--resume=true", "--steps=50000"])

    def test_several_gpus_start_one_process_each_through_accelerate(self):
        command, _ = train.build_command(make_args(gpu="0,3"), CONFIG, "smolvla")
        expected_start = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes=2",
            "--num_machines=1",
            "--dynamo_backend=no",
            "--mixed_precision=bf16",
            TRAINER_SCRIPT,
        ]
        self.assertEqual(command[:9], expected_start)

    def test_several_gpus_without_autocast_tell_accelerate_so(self):
        command, _ = train.build_command(make_args(gpu="0,3", precision="fp32"), CONFIG, "smolvla")
        self.assertIn("--mixed_precision=no", command)

    def test_one_gpu_runs_the_script_directly(self):
        command, _ = train.build_command(make_args(gpu="3"), CONFIG, "smolvla")
        self.assertEqual(command[:2], [sys.executable, TRAINER_SCRIPT])

    def test_empty_gpu_value_runs_the_script_directly(self):
        command, _ = train.build_command(make_args(gpu=""), CONFIG, "smolvla")
        self.assertEqual(command[:2], [sys.executable, TRAINER_SCRIPT])


class MainTestCase(unittest.TestCase):
    """main() on a fake project: a config, a dataset with three episodes and one policy folder of each type."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.project = Path(temp_dir.name).resolve()
        self.dataset = self.project / "dataset"
        self.meta = self.dataset / "meta"
        self.meta.mkdir(parents=True)
        (self.meta / "info.json").write_text(json.dumps({"total_episodes": 3}))
        self.config_path = self.project / "config.json"
        self.config_path.write_text(json.dumps({"repo_id": CONFIG["repo_id"], "dataset_root": str(self.dataset)}))
        self.smolvla_folder = self.make_policy_folder(self.project / "smolvla", "smolvla")
        self.pi05_folder = self.make_policy_folder(self.project / "pi05", "pi05")
        self.output_dir = self.project / "outputs" / "train" / RUN_NAME
        self.saved_model = self.output_dir / "checkpoints" / "last" / "pretrained_model"

        self.start_patch(patch.object(train, "PROJECT", self.project))
        self.start_patch(patch("os.cpu_count", return_value=8))
        self.start_patch(patch("importlib.util.find_spec", return_value=object()))
        self.start_patch(patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True))
        self.start_patch(patch("subprocess.Popen", side_effect=AssertionError("train.py must not start a process")))
        self.trainer_call = self.start_patch(patch.object(subprocess, "call", return_value=0))

    def start_patch(self, patcher):
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    @staticmethod
    def make_policy_folder(folder: Path, policy_type: str) -> Path:
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"type": policy_type}))
        return folder

    def run_without_run_name(self, *arguments: str) -> tuple[int, str, str]:
        """Run main() with the fake config; gives its exit code, stdout and stderr."""
        argv = ["train.py", "--config", str(self.config_path), *arguments]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exit_code = train.main()
            except SystemExit as exit_request:
                exit_code = exit_request.code
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def run_main(self, *arguments: str) -> tuple[int, str, str]:
        return self.run_without_run_name("--run-name", RUN_NAME, *arguments)

    def run_smolvla(self, *arguments: str) -> tuple[int, str, str]:
        return self.run_main("--policy", str(self.smolvla_folder), *arguments)

    def run_pi05(self, *arguments: str) -> tuple[int, str, str]:
        return self.run_main("--policy", str(self.pi05_folder), *arguments)

    def write_box_labels(self, *episodes: int) -> None:
        labels = {"camera": "observation.images.top", "episodes": {str(episode): {} for episode in episodes}}
        (self.meta / "cube_boxes.json").write_text(json.dumps(labels))

    def launched_env(self) -> dict[str, str]:
        return self.trainer_call.call_args.kwargs["env"]


class LaunchedCommandTests(MainTestCase):
    def test_smolvla_defaults_reach_the_trainer(self):
        self.run_smolvla()
        expected_options = smolvla_options(
            policy_folder=self.smolvla_folder,
            dataset_root=self.dataset,
            output_dir=self.output_dir,
        )
        self.assertEqual(self.trainer_call.call_args.args[0], [sys.executable, TRAINER_SCRIPT, *expected_options])

    def test_pi05_defaults_reach_the_trainer(self):
        self.run_pi05()
        expected_options = pi05_options(
            policy_folder=self.pi05_folder,
            dataset_root=self.dataset,
            output_dir=self.output_dir,
        )
        self.assertEqual(self.trainer_call.call_args.args[0], [sys.executable, TRAINER_SCRIPT, *expected_options])

    def test_box_and_crop_leave_the_command_as_it_is(self):
        self.write_box_labels(0, 1, 2)
        self.run_smolvla()
        plain_command = self.trainer_call.call_args.args[0]
        self.run_smolvla("--target-box", "--top-crop")
        self.assertEqual(self.trainer_call.call_args.args[0], plain_command)

    def test_given_values_beat_the_policy_defaults(self):
        self.run_pi05("--batch-size", "32", "--steps", "20000", "--save-freq", "2500")
        command = self.trainer_call.call_args.args[0]
        self.assertIn("--batch_size=32", command)
        self.assertIn("--steps=20000", command)
        self.assertIn("--save_freq=2500", command)

    def test_dataset_flag_replaces_the_root_from_the_config(self):
        other_dataset = self.project / "other"
        (other_dataset / "meta").mkdir(parents=True)
        (other_dataset / "meta" / "info.json").write_text("{}")
        self.run_smolvla("--dataset", str(other_dataset))
        self.assertIn(f"--dataset.root={other_dataset}", self.trainer_call.call_args.args[0])

    def test_arguments_after_the_separator_are_passed_on_unchanged(self):
        self.run_smolvla("--", "--policy.gradient_checkpointing=true")
        self.assertEqual(self.trainer_call.call_args.args[0][-1], "--policy.gradient_checkpointing=true")

    def test_worker_count_is_capped_at_64(self):
        with patch("os.cpu_count", return_value=128):
            self.run_smolvla()
        self.assertIn("--num_workers=64", self.trainer_call.call_args.args[0])

    def test_worker_count_is_one_when_the_core_count_is_unknown(self):
        with patch("os.cpu_count", return_value=None):
            self.run_smolvla()
        self.assertIn("--num_workers=1", self.trainer_call.call_args.args[0])

    def test_default_run_name_holds_the_start_time(self):
        with patch("time.strftime", return_value="20261105-120000"):
            self.run_without_run_name("--policy", str(self.smolvla_folder))
        self.assertIn("--job_name=run-20261105-120000", self.trainer_call.call_args.args[0])

    def test_training_runs_from_the_project_root(self):
        self.run_smolvla()
        self.assertEqual(self.trainer_call.call_args.kwargs["cwd"], self.project)

    def test_exit_code_of_the_trainer_is_returned(self):
        self.trainer_call.return_value = 7
        exit_code, _, _ = self.run_smolvla()
        self.assertEqual(exit_code, 7)

    def test_start_prints_the_dataset_and_the_run_folder(self):
        _, stdout, _ = self.run_smolvla()
        self.assertEqual(stdout, f"Датасет: {self.dataset}\nЗапуск: {self.output_dir}\n")

    def test_outputs_folder_is_created_but_not_the_run_folder(self):
        self.run_smolvla()
        self.assertTrue(self.output_dir.parent.is_dir())
        self.assertFalse(self.output_dir.exists())


class EnvironmentTests(MainTestCase):
    python_first_path = f"{Path(sys.executable).parent}{os.pathsep}/usr/bin"

    def test_plain_smolvla_run_uses_bf16_without_box_and_crop(self):
        self.run_smolvla()
        expected = {
            "PATH": self.python_first_path,
            "ACCELERATE_MIXED_PRECISION": "bf16",
            "TARGET_BOX": "0",
            "TOP_CROP": "0",
            "POLICY_TYPE": "smolvla",
        }
        self.assertEqual(self.launched_env(), expected)

    def test_pi05_run_with_box_and_crop_turns_both_on(self):
        self.write_box_labels(0, 1, 2)
        self.run_pi05("--target-box", "--top-crop")
        expected = {
            "PATH": self.python_first_path,
            "ACCELERATE_MIXED_PRECISION": "bf16",
            "TARGET_BOX": "1",
            "TOP_CROP": "1",
            "POLICY_TYPE": "pi05",
        }
        self.assertEqual(self.launched_env(), expected)

    def test_smolvla_run_with_crop_only_leaves_the_box_off(self):
        self.run_smolvla("--top-crop")
        self.assertEqual((self.launched_env()["TARGET_BOX"], self.launched_env()["TOP_CROP"]), ("0", "1"))

    def test_gpu_flag_selects_the_visible_devices(self):
        self.run_smolvla("--gpu", "0,3")
        self.assertEqual(self.launched_env()["CUDA_VISIBLE_DEVICES"], "0,3")

    def test_empty_gpu_value_hides_all_devices(self):
        self.run_smolvla("--gpu", "")
        self.assertEqual(self.launched_env()["CUDA_VISIBLE_DEVICES"], "")

    def test_fp32_turns_autocast_off(self):
        self.run_smolvla("--precision", "fp32")
        self.assertEqual(self.launched_env()["ACCELERATE_MIXED_PRECISION"], "no")

    def test_variables_of_this_process_are_kept(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        self.run_smolvla()
        self.assertEqual(self.launched_env()["HF_HUB_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
