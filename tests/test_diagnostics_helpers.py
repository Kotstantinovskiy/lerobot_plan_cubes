"""The pure helpers of the diagnostics scripts: summaries of timings and what is printed; no camera or port opens."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))
import benchmark_capture  # noqa: E402
import camera_backend_trial  # noqa: E402
import compare_camera_backends  # noqa: E402
import diagnose_cameras  # noqa: E402

WRIST_CONFIG = {"width": 1920, "height": 1080, "macos_device_name": "Wrist Cam"}


class CompareBackendsTests(unittest.TestCase):
    def test_gap_summary_gives_rate_longest_gap_and_long_gap_count(self):
        summary = camera_backend_trial.gap_summary([0.1, 0.6, 0.5], frames=30, elapsed_s=2.0)
        self.assertEqual(summary, {"fps": 15.0, "max_gap_ms": 600.0, "gaps_over_500ms": 1})

    def test_mode_warnings_keep_only_lines_about_the_mode_whatever_their_case(self):
        log_text = "Input #0, avfoundation\nSelected framerate (29.97) is not supported\nFALLBACK to 640x480\n"
        warnings = camera_backend_trial.ffmpeg_mode_warnings(log_text)
        self.assertEqual(warnings, ["Selected framerate (29.97) is not supported", "FALLBACK to 640x480"])

    def test_trial_summary_keeps_the_keys_present_in_the_printed_order(self):
        report = {"frames": 10, "trial": "1-lerobot", "config": {}, "identity_after": {}}
        self.assertEqual(list(compare_camera_backends.trial_summary(report)), ["trial", "frames"])

    def test_ffmpeg_command_asks_for_the_exact_duration_size_and_device(self):
        duration = SimpleNamespace(timescale=30000030, value=1000000)
        command = camera_backend_trial.ffmpeg_command(WRIST_CONFIG, duration)
        self.assertEqual(command[command.index("-framerate") + 1], "30000030/1000000")
        self.assertEqual(command[command.index("-video_size") + 1], "1920x1080")
        self.assertEqual(command[command.index("-i") + 1], "Wrist Cam:none")

    def test_worker_command_passes_the_top_camera_flag_only_when_asked(self):
        without_top = compare_camera_backends.worker_command(
            "ffmpeg",
            Path("/out/1-ffmpeg"),
            20,
            with_top=False,
        )
        with_top = compare_camera_backends.worker_command(
            "ffmpeg",
            Path("/out/1-ffmpeg"),
            20,
            with_top=True,
        )
        self.assertEqual(without_top[2:], ["--worker", "ffmpeg", "--out", "/out/1-ffmpeg", "--seconds", "20"])
        self.assertEqual(with_top, [*without_top, "--with-top"])


class BenchmarkSummaryTests(unittest.TestCase):
    def test_work_summary_compares_each_frame_with_the_budget(self):
        timings = benchmark_capture.CaptureTimings(
            frame_count=4,
            elapsed_s=2.0,
            work_times_s=[0.1, 0.1, 0.1, 0.7],
            stages_ms={},
        )
        summary = benchmark_capture.work_summary(timings, fps=2)
        self.assertEqual(summary["actual_fps"], 2.0)
        self.assertEqual(summary["over_budget_fraction"], 0.25)

    def test_stage_summary_counts_delayed_reads_of_cameras_only(self):
        stages_ms = {"follower_read": [300.0], "top": [10.0, 250.0], "enqueue": [1.0]}
        summary = benchmark_capture.stage_summary(stages_ms, ["top"])
        self.assertEqual(summary["camera_reads_over_200ms"], {"top": 1})
        self.assertEqual(summary["stages_mean_ms"]["top"], 130.0)
        self.assertEqual(summary["stages_max_ms"]["follower_read"], 300.0)


class DiagnoseCamerasTests(unittest.TestCase):
    def test_periodic_line_shows_only_frames_and_timeouts(self):
        statuses = {"top": {"frames": 5, "timeouts": 1, "errors": [], "shape": [480, 640, 3]}}
        self.assertEqual(diagnose_cameras.frame_counts(statuses), {"top": {"frames": 5, "timeouts": 1}})


if __name__ == "__main__":
    unittest.main()
