"""What training/train.py does around the training command: dry runs, resuming, refusals, parsing --episodes.

Runs on the fake project of test_training_command.py: nothing is started, and neither the network nor a GPU is touched.
"""

import argparse
import os
import shlex
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import train  # noqa: E402
from test_training_command import TRAINER_SCRIPT, MainTestCase, smolvla_options  # noqa: E402


class DryRunTests(MainTestCase):
    def test_dry_run_prints_the_command_and_starts_nothing(self):
        exit_code, stdout, _ = self.run_smolvla("--dry-run")
        expected_options = smolvla_options(
            policy_folder=self.smolvla_folder,
            dataset_root=self.dataset,
            output_dir=self.output_dir,
        )
        command = shlex.join([sys.executable, TRAINER_SCRIPT, *expected_options])
        self.assertEqual(stdout, f"Датасет: {self.dataset}\nЗапуск: {self.output_dir}\n{command}\n")
        self.assertEqual(exit_code, 0)
        self.trainer_call.assert_not_called()

    def test_dry_run_needs_no_dataset(self):
        (self.meta / "info.json").unlink()
        exit_code, _, _ = self.run_smolvla("--dry-run")
        self.assertEqual(exit_code, 0)

    def test_dry_run_needs_no_lerobot(self):
        with patch("importlib.util.find_spec", return_value=None):
            exit_code, _, _ = self.run_smolvla("--dry-run")
        self.assertEqual(exit_code, 0)

    def test_dry_run_does_not_create_the_outputs_folder(self):
        self.run_smolvla("--dry-run")
        self.assertFalse((self.project / "outputs").exists())


class ResumeTests(MainTestCase):
    def setUp(self):
        super().setUp()
        self.make_policy_folder(self.saved_model, "pi05")

    def test_resume_passes_the_saved_config_to_the_trainer(self):
        self.run_main("--resume")
        expected = [
            sys.executable,
            TRAINER_SCRIPT,
            f"--config_path={self.saved_model / 'train_config.json'}",
            "--resume=true",
        ]
        self.assertEqual(self.trainer_call.call_args.args[0], expected)

    def test_policy_type_comes_from_the_checkpoint(self):
        self.run_main("--resume")
        self.assertEqual(self.launched_env()["POLICY_TYPE"], "pi05")

    def test_box_and_crop_stay_off_without_their_files_in_the_checkpoint(self):
        self.run_main("--resume")
        self.assertEqual((self.launched_env()["TARGET_BOX"], self.launched_env()["TOP_CROP"]), ("0", "0"))

    def test_target_box_file_in_the_checkpoint_turns_the_box_on(self):
        (self.saved_model / "target_box.json").write_text("{}")
        self.run_main("--resume")
        self.assertEqual((self.launched_env()["TARGET_BOX"], self.launched_env()["TOP_CROP"]), ("1", "0"))

    def test_top_crop_file_in_the_checkpoint_turns_the_crop_on(self):
        (self.saved_model / "top_crop.json").write_text("{}")
        self.run_main("--resume")
        self.assertEqual((self.launched_env()["TARGET_BOX"], self.launched_env()["TOP_CROP"]), ("0", "1"))

    def test_flags_turn_box_and_crop_on_even_without_their_files(self):
        self.run_main("--resume", "--target-box", "--top-crop")
        self.assertEqual((self.launched_env()["TARGET_BOX"], self.launched_env()["TOP_CROP"]), ("1", "1"))

    def test_resume_needs_no_dataset(self):
        (self.meta / "info.json").unlink()
        exit_code, _, _ = self.run_main("--resume")
        self.assertEqual(exit_code, 0)

    def test_resume_keeps_the_precision_of_the_checkpoint(self):
        (self.saved_model / train.PRECISION_FILE).write_text("fp32")
        self.run_main("--resume")
        self.assertEqual(self.launched_env()["ACCELERATE_MIXED_PRECISION"], "no")

    def test_precision_flag_overrides_the_checkpoint(self):
        (self.saved_model / train.PRECISION_FILE).write_text("fp32")
        self.run_main("--resume", "--precision", "bf16")
        self.assertEqual(self.launched_env()["ACCELERATE_MIXED_PRECISION"], "bf16")

    def test_older_checkpoint_gets_the_default_precision(self):
        self.run_main("--resume")
        self.assertEqual(self.launched_env()["ACCELERATE_MIXED_PRECISION"], "bf16")

    def test_older_checkpoint_says_which_precision_it_got(self):
        _, stdout, _ = self.run_main("--resume")
        self.assertIn("не хранит точность обучения: продолжаю с bf16", stdout)

    def test_resume_on_several_gpus_tells_accelerate_the_precision_of_the_checkpoint(self):
        (self.saved_model / train.PRECISION_FILE).write_text("fp32")
        self.run_main("--resume", "--gpu", "0,3")
        self.assertIn("--mixed_precision=no", self.trainer_call.call_args.args[0])


class NewRunPrecisionTests(MainTestCase):
    def test_new_run_defaults_to_bf16(self):
        self.run_smolvla()
        self.assertEqual(self.launched_env()["ACCELERATE_MIXED_PRECISION"], "bf16")

    def test_new_run_on_several_gpus_tells_accelerate_bf16(self):
        self.run_smolvla("--gpu", "0,3")
        self.assertIn("--mixed_precision=bf16", self.trainer_call.call_args.args[0])

    def test_new_run_says_nothing_about_a_checkpoint(self):
        _, stdout, _ = self.run_smolvla()
        self.assertNotIn("не хранит точность", stdout)


class RelativePolicyTests(MainTestCase):
    def test_relative_policy_path_reaches_the_trainer_as_absolute(self):
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.project)
        self.run_main("--policy", "smolvla")
        self.assertIn(f"--policy.path={self.smolvla_folder}", self.trainer_call.call_args.args[0])

    def test_absolute_policy_path_stays_as_it_is(self):
        self.run_smolvla()
        self.assertIn(f"--policy.path={self.smolvla_folder}", self.trainer_call.call_args.args[0])


class RefusalTests(MainTestCase):
    def test_resume_without_a_checkpoint_stops_with_exit_code_2(self):
        result = self.run_main("--resume")
        expected_error = f"Ошибка: Нет чекпойнта для продолжения: {self.saved_model.parent}\n"
        self.assertEqual(result, (2, "", expected_error))

    def test_missing_config_stops_with_exit_code_2(self):
        exit_code, _, stderr = self.run_main("--config", str(self.project / "absent.json"))
        self.assertEqual(exit_code, 2)
        self.assertTrue(stderr.startswith("Ошибка: "))

    def test_unknown_policy_type_stops_with_exit_code_2(self):
        folder = self.make_policy_folder(self.project / "act", "act")
        result = self.run_main("--policy", str(folder))
        expected_error = f"Ошибка: {folder}: policy type act, train.py knows smolvla, pi05\n"
        self.assertEqual(result, (2, "", expected_error))

    def test_missing_lerobot_stops_before_the_dataset_check(self):
        (self.meta / "info.json").unlink()
        with patch("importlib.util.find_spec", return_value=None):
            exit_code, _, stderr = self.run_smolvla()
        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr, "Ошибка: LeRobot не установлен в этом Python. См. training/README.md\n")

    def test_missing_dataset_stops_with_exit_code_2(self):
        (self.meta / "info.json").unlink()
        exit_code, _, stderr = self.run_smolvla()
        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr, f"Ошибка: Нет датасета: {self.dataset}. См. training/fetch_dataset.sh\n")

    def test_existing_run_folder_stops_with_exit_code_2(self):
        self.output_dir.mkdir(parents=True)
        exit_code, _, stderr = self.run_smolvla()
        expected_error = (
            f"Ошибка: Папка запуска уже существует: {self.output_dir}. Укажи другой --run-name или --resume\n"
        )
        self.assertEqual((exit_code, stderr), (2, expected_error))

    def test_target_box_without_labels_stops_with_exit_code_2(self):
        exit_code, _, stderr = self.run_smolvla("--target-box")
        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr, "Ошибка: Нет meta/cube_boxes.json: сначала python training/label_boxes.py\n")

    def test_target_box_with_unlabelled_episodes_names_them(self):
        self.write_box_labels(1)
        exit_code, _, stderr = self.run_smolvla("--target-box")
        self.assertEqual(exit_code, 2)
        self.assertEqual(
            stderr, "Ошибка: В meta/cube_boxes.json нет эпизодов 0, 2: заново python training/label_boxes.py\n"
        )

    def test_target_box_only_needs_labels_of_the_chosen_episodes(self):
        self.write_box_labels(0, 1)
        exit_code, _, _ = self.run_smolvla("--target-box", "--episodes", "0,1")
        self.assertEqual(exit_code, 0)

    def test_refusal_starts_nothing(self):
        self.output_dir.mkdir(parents=True)
        self.run_smolvla()
        self.trainer_call.assert_not_called()


class EpisodeListTests(unittest.TestCase):
    def test_numbers_are_sorted_without_repeats(self):
        self.assertEqual(train.episode_list("2, 0,2,1"), [0, 1, 2])

    def test_text_is_refused_with_an_example(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "Ожидается список номеров через запятую"):
            train.episode_list("first")

    def test_negative_number_is_refused(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "неотрицательный"):
            train.episode_list("-1,2")

    def test_empty_list_is_refused(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "хотя бы один"):
            train.episode_list(" , ")


if __name__ == "__main__":
    unittest.main()
