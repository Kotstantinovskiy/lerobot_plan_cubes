"""Loading a checkpoint and computing a chunk, with LeRobot's factories replaced: no checkpoint is ever read."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import local_policy  # noqa: E402
import pi05_inputs  # noqa: E402

SMOLVLA_TYPE = "smolvla"
DEVICE = torch.device("cpu")
DATASET_STATS = {"action": {"min": np.zeros(2)}}


def make_policy_config(policy_type=SMOLVLA_TYPE, input_features=()):
    return SimpleNamespace(
        type=policy_type,
        input_features=dict.fromkeys(input_features),
        output_features={},
        normalization_mapping={},
    )


def make_policy(config, rtc_processor="rtc"):
    return SimpleNamespace(config=config, eval=MagicMock(), rtc_processor=rtc_processor)


class TargetBoxStyleTests(unittest.TestCase):
    def test_style_saved_next_to_the_weights_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "target_box.json").write_text(json.dumps({"thickness": 6}))
            self.assertEqual(local_policy.target_box_style(directory), {"thickness": 6})

    def test_checkpoint_without_a_style_gets_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(local_policy.target_box_style(directory))


class SavedStatsTests(unittest.TestCase):
    def test_hub_id_or_folder_without_statistics_gets_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(local_policy.saved_stats(directory))

    def test_statistics_are_grouped_by_feature(self):
        arrays = {"action.min": np.zeros(2), "action.max": np.ones(2), "observation.state.mean": np.ones(3)}
        with tempfile.TemporaryDirectory() as directory:
            save_file(arrays, Path(directory) / "policy_preprocessor_step_3_normalizer_processor.safetensors")
            stats = local_policy.saved_stats(directory)
        self.assertEqual(sorted(stats["action"]), ["max", "min"])
        self.assertEqual(sorted(stats["observation.state"]), ["mean"])

    def test_first_file_by_name_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {"action.min": np.zeros(2)},
                Path(directory) / "policy_preprocessor_step_1_normalizer_processor.safetensors",
            )
            save_file(
                {"action.max": np.ones(2)},
                Path(directory) / "policy_preprocessor_step_2_normalizer_processor.safetensors",
            )
            stats = local_policy.saved_stats(directory)
        self.assertEqual(list(stats["action"]), ["min"])


class CameraRenameTests(unittest.TestCase):
    def test_pi05_gets_its_own_camera_names(self):
        config = make_policy_config(pi05_inputs.POLICY_TYPE)
        self.assertEqual(local_policy.camera_rename_map(config), local_policy.PI05_CAMERA_RENAME)

    def test_smolvla_base_gets_camera1_and_camera2(self):
        config = make_policy_config(input_features=["observation.images.camera1"])
        self.assertEqual(local_policy.camera_rename_map(config), local_policy.SMOLVLA_CAMERA_RENAME)

    def test_checkpoint_trained_on_our_names_needs_no_renaming(self):
        config = make_policy_config(input_features=["observation.images.top"])
        self.assertEqual(local_policy.camera_rename_map(config), {})


class RtcConfigTests(unittest.TestCase):
    def test_settings_reach_the_config(self):
        rtc = local_policy.make_rtc_config(7, 2.5)
        self.assertEqual((rtc.enabled, rtc.execution_horizon, rtc.max_guidance_weight), (True, 7, 2.5))

    def test_attention_schedule_is_exponential(self):
        self.assertEqual(local_policy.make_rtc_config(7, 2.5).prefix_attention_schedule.value, "EXP")


class LoadPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = make_policy_config(input_features=["observation.images.top"])
        self.config.load_vlm_weights = True
        self.policy = make_policy(self.config)
        self.preprocess, self.postprocess = MagicMock(), MagicMock()
        self.make_processors = MagicMock(return_value=(self.preprocess, self.postprocess))
        self.start_patch(patch.object(local_policy.PreTrainedConfig, "from_pretrained", return_value=self.config))
        self.start_patch(patch.object(local_policy, "make_policy", return_value=self.policy))
        self.start_patch(patch.object(local_policy, "make_pre_post_processors", self.make_processors))
        self.start_patch(patch.object(local_policy, "saved_stats", return_value=None))
        self.start_patch(patch.object(pi05_inputs, "load_tokenizer", return_value="tokenizer"))
        self.dataset_meta = SimpleNamespace(stats=DATASET_STATS)

    def start_patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    def load(self, rtc=None):
        return local_policy.load_policy(
            "some/policy",
            "cpu",
            7,
            self.dataset_meta,
            rtc,
        )

    def test_result_unpacks_into_policy_and_processors(self):
        policy, preprocess, postprocess = self.load()
        self.assertEqual((policy, preprocess, postprocess), (self.policy, self.preprocess, self.postprocess))

    def test_policy_is_in_eval_mode(self):
        self.load()
        self.policy.eval.assert_called_once_with()

    def test_action_steps_and_device_go_into_the_config(self):
        self.load()
        self.assertEqual((self.config.n_action_steps, self.config.device), (7, "cpu"))

    def test_vlm_weights_are_left_to_the_checkpoint(self):
        self.load()
        self.assertFalse(self.config.load_vlm_weights)

    def test_rtc_goes_into_the_config(self):
        self.load("rtc settings")
        self.assertEqual(self.config.rtc_config, "rtc settings")

    def test_rtc_on_a_policy_without_it_is_refused(self):
        self.policy.rtc_processor = None
        with self.assertRaisesRegex(RuntimeError, "no Real-Time Chunking"):
            self.load("rtc settings")

    def test_base_policy_normalizes_with_the_dataset(self):
        self.load()
        self.assertIs(self.make_processors.call_args.kwargs["dataset_stats"], DATASET_STATS)

    def test_trained_checkpoint_normalizes_with_its_own_statistics(self):
        with patch.object(local_policy, "saved_stats", return_value={"action": {}}):
            self.load()
        self.assertEqual(self.make_processors.call_args.kwargs["dataset_stats"], {"action": {}})

    def test_only_pi05_gets_a_tokenizer(self):
        self.load()
        self.assertNotIn("tokenizer_processor", self.make_processors.call_args.kwargs["preprocessor_overrides"])

    def test_pi05_gets_the_downloaded_tokenizer(self):
        self.config.type = pi05_inputs.POLICY_TYPE
        self.load()
        overrides = self.make_processors.call_args.kwargs["preprocessor_overrides"]
        self.assertEqual(overrides["tokenizer_processor"], {"tokenizer": "tokenizer"})

    def test_pi05_preprocessor_gets_the_frames_shrunk_first(self):
        self.config.type = pi05_inputs.POLICY_TYPE
        _, preprocess, _ = self.load()
        preprocess({"observation.images.top": torch.zeros(1, 3, 720, 1280)})
        shrunk = self.preprocess.call_args.args[0]["observation.images.top"]
        self.assertEqual(shrunk.shape, (1, 3, pi05_inputs.IMAGE_SIZE, pi05_inputs.IMAGE_SIZE))


class ComputeChunkTests(unittest.TestCase):
    def setUp(self):
        self.policy = MagicMock()
        self.policy.predict_action_chunk.return_value = torch.ones(1, 5, 2, dtype=torch.float64)

    def compute(self, frame, **guidance):
        return local_policy.compute_chunk(
            self.policy,
            lambda batch: batch,
            lambda raw_chunk: raw_chunk * 10,
            frame,
            DEVICE,
            "Pick up",
            "so101_follower",
            **guidance,
        )

    def test_chunk_comes_in_dataset_units_without_the_batch_dimension(self):
        chunk, _ = self.compute({"observation.state": np.zeros(2, np.float32)})
        self.assertEqual((chunk.shape, chunk[0, 0]), ((5, 2), 10.0))

    def test_chunk_comes_as_the_model_gave_it_too(self):
        _, raw_chunk = self.compute({"observation.state": np.zeros(2, np.float32)})
        self.assertEqual((raw_chunk.shape, raw_chunk[0, 0].item()), (torch.Size([5, 2]), 1.0))

    def test_task_and_robot_type_reach_the_model(self):
        self.compute({"observation.state": np.zeros(2, np.float32)})
        batch = self.policy.predict_action_chunk.call_args.args[0]
        self.assertEqual((batch["task"], batch["robot_type"]), ("Pick up", "so101_follower"))

    def test_guidance_reaches_the_model(self):
        self.compute({"observation.state": np.zeros(2, np.float32)}, inference_delay=3)
        self.assertEqual(self.policy.predict_action_chunk.call_args.kwargs, {"inference_delay": 3})

    def test_given_frame_stays_as_it_was(self):
        state = np.zeros(2, np.float32)
        frame = {"observation.state": state}
        self.compute(frame)
        self.assertIs(frame["observation.state"], state)


class DatasetObservationTests(unittest.TestCase):
    def setUp(self):
        self.dataset_meta = SimpleNamespace(
            repo_id="local/cubes",
            fps=20,
            episodes={1: {"dataset_from_index": 100}},
            features={"observation.state": {"names": ["shoulder.pos", "elbow.pos"]}},
        )
        self.dataset = MagicMock()
        self.dataset.__getitem__.return_value = {
            "observation.state": torch.tensor([1.5, 2.5]),
            "observation.images.top": torch.ones(3, 4, 6),
            "observation.images.wrist": torch.zeros(3, 2, 2),
        }

    def read(self, offset_s):
        with patch.object(local_policy, "LeRobotDataset", return_value=self.dataset):
            return local_policy.dataset_observation(
                self.dataset_meta,
                "root",
                1,
                offset_s,
            )

    def test_offset_counts_from_the_start_of_the_episode(self):
        self.read(3)
        self.dataset.__getitem__.assert_called_once_with(160)

    def test_offset_a_hair_under_a_frame_still_means_that_frame(self):
        self.read(61 / 20 - 1e-9)
        self.dataset.__getitem__.assert_called_once_with(161)

    def test_joints_come_by_name(self):
        self.assertEqual(self.read(3)["elbow.pos"], 2.5)

    def test_camera_images_come_as_uint8_height_width_channels(self):
        image = self.read(3)["top"]
        self.assertEqual((image.shape, image.dtype, image[0, 0, 0]), ((4, 6, 3), np.uint8, 255))


if __name__ == "__main__":
    unittest.main()
