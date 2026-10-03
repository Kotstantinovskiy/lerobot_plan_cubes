"""training/lerobot_train_fast.py against the LeRobot functions it replaces, with fakes: nothing trains or downloads.

Importing the script patches LeRobot for the whole process, so setUpModule imports it for this file only and
tearDownModule undoes the patches. Needs LeRobot, as the script does.
"""

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import accelerate.utils
import torch
from lerobot.datasets import transforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

TOP_CAMERA = "observation.images.top"
WRIST_CAMERA = "observation.images.wrist"
EMPTY_PI05_SLOT = "observation.images.right_wrist_0_rgb"
PI05_RENAME_MAP = {TOP_CAMERA: "observation.images.base_0_rgb", WRIST_CAMERA: "observation.images.left_wrist_0_rgb"}
TARGET_BOX = [500, 300, 545, 345]
PATCHED_BY_THE_SCRIPT = [
    (LeRobotDataset, "__getitem__"),
    (LeRobotDataset, "_query_videos"),
    (torch.utils.data, "DataLoader"),
    (transforms, "make_transform_from_config"),
    (lerobot_train, "make_policy"),
    (lerobot_train, "make_pre_post_processors"),
    (lerobot_train, "save_checkpoint"),
    (accelerate.utils, "DistributedDataParallelKwargs"),
]
LEROBOT_ATTRIBUTES = [(owner, name, getattr(owner, name)) for owner, name in PATCHED_BY_THE_SCRIPT]

fast_trainer = None  # the script, imported by setUpModule


def setUpModule():
    global fast_trainer
    fast_trainer = importlib.import_module("lerobot_train_fast")


def tearDownModule():
    for owner, name, value in LEROBOT_ATTRIBUTES:
        setattr(owner, name, value)
    del sys.modules["lerobot_train_fast"]


def make_image_transforms():
    """One transform of each type train.py passes to LeRobot, configured as LeRobot parses them."""
    transform_configs = {
        "brightness": transforms.ImageTransformConfig(type="ColorJitter", kwargs={"brightness": [0.5, 1.5]}),
        "lighting": transforms.ImageTransformConfig(type="LightingGradient", kwargs={"gain": [0.6, 1.4]}),
        "affine": transforms.ImageTransformConfig(type="RandomAffine", kwargs={"degrees": [-5.0, 5.0]}),
    }
    config = transforms.ImageTransformsConfig(enable=True, max_num_transforms=3, tfs=transform_configs)
    return transforms.ImageTransforms(config)


class InstalledPatchesTests(unittest.TestCase):
    def test_lerobot_train_calls_the_replacements(self):
        replaced = (lerobot_train.make_policy, lerobot_train.make_pre_post_processors, lerobot_train.save_checkpoint)
        expected = (fast_trainer.make_policy, fast_trainer.make_pre_post_processors, fast_trainer.save_checkpoint)
        self.assertEqual(replaced, expected)

    def test_dataset_samples_come_from_the_replacement(self):
        self.assertIs(LeRobotDataset.__getitem__, fast_trainer.getitem)

    def test_lerobot_train_builds_the_replaced_data_loader(self):
        self.assertIs(torch.utils.data.DataLoader, fast_trainer.DataLoader)

    def test_image_transforms_are_built_by_the_replacement(self):
        self.assertIs(transforms.make_transform_from_config, fast_trainer.make_transform)

    def test_several_gpus_keep_gradients_in_the_communication_buckets(self):
        kwargs = accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=True)
        self.assertTrue(kwargs.gradient_as_bucket_view)


class LightingGradientTests(unittest.TestCase):
    def test_frame_keeps_its_shape(self):
        lighting = fast_trainer.LightingGradient(gain=[0.6, 1.4])
        self.assertEqual(lighting(torch.rand(3, 20, 30)).shape, (3, 20, 30))

    def test_result_stays_in_the_image_range(self):
        lighting = fast_trainer.LightingGradient(gain=[1.2, 1.4])
        self.assertEqual(lighting(torch.ones(3, 20, 30)).max().item(), 1.0)

    def test_gain_of_one_leaves_the_frame_as_it_is(self):
        frame = torch.rand(3, 20, 30)
        lighting = fast_trainer.LightingGradient(gain=[1.0, 1.0])
        self.assertTrue(torch.equal(lighting(frame), frame))

    def test_make_transform_knows_the_lighting_type(self):
        config = transforms.ImageTransformConfig(type="LightingGradient", kwargs={"gain": [0.6, 1.4]})
        self.assertIsInstance(fast_trainer.make_transform(config), fast_trainer.LightingGradient)

    def test_make_transform_leaves_other_types_to_lerobot(self):
        config = transforms.ImageTransformConfig(type="ColorJitter", kwargs={"brightness": [0.5, 1.5]})
        self.assertNotIsInstance(fast_trainer.make_transform(config), fast_trainer.LightingGradient)


class SamplerTests(unittest.TestCase):
    def test_permutations_follow_each_other(self):
        indices = list(fast_trainer.PermutationChainSampler(3, epochs=2))
        self.assertEqual((sorted(indices[:3]), sorted(indices[3:])), ([0, 1, 2], [0, 1, 2]))

    def test_length_counts_all_epochs(self):
        self.assertEqual(len(fast_trainer.PermutationChainSampler(5, epochs=2)), 10)

    def test_shuffling_data_loader_uses_the_chain_sampler(self):
        loader = fast_trainer.DataLoader(list(range(6)), batch_size=2, shuffle=True)
        self.assertIsInstance(loader.sampler, fast_trainer.PermutationChainSampler)

    def test_data_loader_keeps_a_given_sampler(self):
        sampler = torch.utils.data.SequentialSampler(list(range(6)))
        loader = fast_trainer.DataLoader(list(range(6)), batch_size=2, shuffle=False, sampler=sampler)
        self.assertIs(loader.sampler, sampler)


class MakePolicyTests(unittest.TestCase):
    def make_policy(self, policy_type: str) -> types.SimpleNamespace:
        """Call the replacement with the keywords lerobot_train uses; gives the config LeRobot's function received."""
        cfg = types.SimpleNamespace(
            type=policy_type,
            input_features={PI05_RENAME_MAP[TOP_CAMERA]: 1, PI05_RENAME_MAP[WRIST_CAMERA]: 2, EMPTY_PI05_SLOT: 3},
        )
        cfg.image_features = dict(cfg.input_features)
        dataset_meta = types.SimpleNamespace(camera_keys=[TOP_CAMERA, WRIST_CAMERA])
        with patch.object(fast_trainer, "original_make_policy") as original:
            fast_trainer.make_policy(cfg=cfg, ds_meta=dataset_meta, rename_map=PI05_RENAME_MAP)
        return original.call_args.args[0]

    def test_pi05_loses_the_camera_slot_the_dataset_has_no_camera_for(self):
        self.assertNotIn(EMPTY_PI05_SLOT, self.make_policy("pi05").input_features)

    def test_pi05_keeps_the_slots_of_its_cameras(self):
        self.assertEqual(
            list(self.make_policy("pi05").input_features),
            [PI05_RENAME_MAP[TOP_CAMERA], PI05_RENAME_MAP[WRIST_CAMERA]],
        )

    def test_smolvla_keeps_all_its_features(self):
        self.assertIn(EMPTY_PI05_SLOT, self.make_policy("smolvla").input_features)


class MakeProcessorsTests(unittest.TestCase):
    def make_processors(self, policy_type: str, **kwargs) -> dict:
        """Call the replacement with the keywords lerobot_train uses; gives what LeRobot's function received."""
        cfg = types.SimpleNamespace(type=policy_type)
        with (
            patch.object(fast_trainer, "original_make_processors") as original,
            patch.object(fast_trainer.pi05_inputs, "load_tokenizer", return_value="tokenizer"),
        ):
            fast_trainer.make_pre_post_processors(policy_cfg=cfg, pretrained_path="base", **kwargs)
        return original.call_args.kwargs

    def test_pi05_gets_the_tokenizer(self):
        overrides = self.make_processors("pi05")["preprocessor_overrides"]
        self.assertEqual(overrides, {"tokenizer_processor": {"tokenizer": "tokenizer"}})

    def test_pi05_keeps_the_other_overrides(self):
        given = {"device_processor": {"device": "cuda"}}
        overrides = self.make_processors("pi05", preprocessor_overrides=given)["preprocessor_overrides"]
        self.assertEqual(overrides["device_processor"], {"device": "cuda"})

    def test_smolvla_gets_no_overrides(self):
        self.assertEqual(self.make_processors("smolvla"), {"pretrained_path": "base"})


class SaveCheckpointTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.checkpoints = Path(temp_dir.name)
        (self.checkpoints / "000100" / "training_state").mkdir(parents=True)
        self.newest = self.checkpoints / "000200"

    def save_newest(self, *, target_box: bool = False, top_crop: bool = False) -> None:
        """Call the replacement with the keywords lerobot_train uses; LeRobot's function only makes the folders."""

        def make_folders(checkpoint_dir, **_):
            (checkpoint_dir / "pretrained_model").mkdir(parents=True)
            (checkpoint_dir / "training_state").mkdir()

        with (
            patch.object(fast_trainer, "original_save_checkpoint", make_folders),
            patch.object(fast_trainer, "TARGET_BOX", target_box),
            patch.object(fast_trainer, "TOP_CROP", top_crop),
        ):
            fast_trainer.save_checkpoint(checkpoint_dir=self.newest, step=200, cfg=None, policy=None, optimizer=None)

    def test_older_checkpoint_loses_its_optimizer_state(self):
        self.save_newest()
        self.assertFalse((self.checkpoints / "000100" / "training_state").exists())

    def test_newest_checkpoint_keeps_its_optimizer_state(self):
        self.save_newest()
        self.assertTrue((self.newest / "training_state").is_dir())

    def test_plain_run_adds_no_settings(self):
        self.save_newest()
        self.assertEqual(list((self.newest / "pretrained_model").iterdir()), [])

    def test_box_run_saves_the_drawing_style(self):
        self.save_newest(target_box=True)
        saved_style = json.loads((self.newest / "pretrained_model" / "target_box.json").read_text())
        self.assertEqual(saved_style, fast_trainer.cube_boxes.STYLE)

    def test_crop_run_saves_the_crop(self):
        self.save_newest(top_crop=True)
        saved_crop = json.loads((self.newest / "pretrained_model" / "top_crop.json").read_text())
        self.assertEqual(saved_crop, fast_trainer.top_crop.CROP)


class QueryVideosTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        (root / "meta").mkdir()
        labels = {"episodes": {"7": {"target": "red", "frame": 0, "boxes": {"red": TARGET_BOX, "blue": None}}}}
        (root / "meta" / "cube_boxes.json").write_text(json.dumps(labels))
        self.dataset = types.SimpleNamespace(root=root)

    def query_episode_7(self, *, target_box: bool, top_crop: bool) -> dict:
        """The replacement's frames, from a black top frame and wrist frame that LeRobot's function "decodes"."""
        decoded = {TOP_CAMERA: torch.zeros(3, 720, 1280), WRIST_CAMERA: torch.zeros(3, 1080, 1920)}
        with (
            patch.object(fast_trainer, "original_query_videos", return_value=decoded),
            patch.object(fast_trainer, "TARGET_BOX", target_box),
            patch.object(fast_trainer, "TOP_CROP", top_crop),
        ):
            return fast_trainer.query_videos(self.dataset, {TOP_CAMERA: [0.0]}, 7)

    def test_box_of_the_episode_target_is_drawn_on_the_top_frame(self):
        frames = self.query_episode_7(target_box=True, top_crop=False)
        expected = fast_trainer.cube_boxes.draw_chw(torch.zeros(3, 720, 1280), TARGET_BOX)
        self.assertTrue(torch.equal(frames[TOP_CAMERA], expected))

    def test_wrist_frame_gets_no_box(self):
        frames = self.query_episode_7(target_box=True, top_crop=False)
        self.assertEqual(frames[WRIST_CAMERA].sum().item(), 0.0)

    def test_crop_is_cut_from_the_frame_with_the_box(self):
        frames = self.query_episode_7(target_box=True, top_crop=True)
        boxed_frame = fast_trainer.cube_boxes.draw_chw(torch.zeros(3, 720, 1280), TARGET_BOX)
        self.assertTrue(torch.equal(frames[TOP_CAMERA], fast_trainer.top_crop.crop_chw(boxed_frame)))

    def test_labels_are_read_once(self):
        self.query_episode_7(target_box=True, top_crop=False)
        (self.dataset.root / "meta" / "cube_boxes.json").unlink()
        self.query_episode_7(target_box=True, top_crop=False)
        self.assertEqual(self.dataset._target_boxes, {7: TARGET_BOX})


class GetItemTests(unittest.TestCase):
    def setUp(self):
        self.image_transforms = make_image_transforms()
        self.dataset = types.SimpleNamespace(
            image_transforms=self.image_transforms,
            meta=types.SimpleNamespace(camera_keys=[TOP_CAMERA, WRIST_CAMERA]),
        )

    def get_item(self, *, policy_type: str = "smolvla") -> dict:
        def decoded_sample(dataset, _):
            self.transforms_during_decoding = dataset.image_transforms
            return {TOP_CAMERA: torch.rand(3, 720, 1280), WRIST_CAMERA: torch.rand(3, 1080, 1920)}

        with (
            patch.object(fast_trainer, "original_getitem", decoded_sample),
            patch.object(fast_trainer, "POLICY_TYPE", policy_type),
        ):
            return fast_trainer.getitem(self.dataset, 0)

    def test_wrist_camera_gets_no_geometric_transform(self):
        wrist_transforms = fast_trainer.camera_transforms(self.dataset, self.image_transforms, WRIST_CAMERA)
        self.assertEqual(list(wrist_transforms.transforms), ["brightness", "lighting"])

    def test_top_camera_gets_all_transforms(self):
        top_transforms = fast_trainer.camera_transforms(self.dataset, self.image_transforms, TOP_CAMERA)
        self.assertEqual(list(top_transforms.transforms), ["brightness", "lighting", "affine"])

    def test_lerobot_decodes_without_running_the_transforms(self):
        self.get_item()
        self.assertIsNone(self.transforms_during_decoding)

    def test_transforms_are_back_after_decoding(self):
        self.get_item()
        self.assertIs(self.dataset.image_transforms, self.image_transforms)

    def test_transforms_are_back_after_a_decoding_error(self):
        with patch.object(fast_trainer, "original_getitem", side_effect=KeyError), self.assertRaises(KeyError):
            fast_trainer.getitem(self.dataset, 0)
        self.assertIs(self.dataset.image_transforms, self.image_transforms)

    def test_smolvla_frames_are_shrunk_to_512(self):
        sample = self.get_item(policy_type="smolvla")
        self.assertEqual((sample[TOP_CAMERA].shape, sample[WRIST_CAMERA].shape), ((3, 512, 512), (3, 512, 512)))

    def test_pi05_frames_are_shrunk_to_224(self):
        sample = self.get_item(policy_type="pi05")
        self.assertEqual((sample[TOP_CAMERA].shape, sample[WRIST_CAMERA].shape), ((3, 224, 224), (3, 224, 224)))


if __name__ == "__main__":
    unittest.main()
