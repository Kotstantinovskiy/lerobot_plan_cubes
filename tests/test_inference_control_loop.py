"""The control loop of a policy run, with a fake robot, policy and video: no port, camera or checkpoint is touched."""

import sys
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "inference"))
sys.path.insert(0, str(PROJECT / "recording"))

import arm_control  # noqa: E402
from arm_control import (  # noqa: E402
    ArmRun,
    PendingChunk,
    PolicyView,
    RunState,
    drive_arm,
    is_replan_due,
    pop_next_action,
    rtc_guidance,
    seam_size,
    start_chunk,
    summarize_seams,
    take_background_chunk,
    track_motion,
)
from inference_arm_fakes import (  # noqa: E402
    ELBOW,
    FPS,
    MOVED_POSE,
    SHOULDER,
    STILL_POSE,
    FakeClock,
    FakeRobot,
    InlineExecutor,
    make_args,
    make_chunk,
    make_facts,
    make_video,
    silently,
)

OLD_CROP = {"camera": "observation.images.top", "box": [100, 100, 300, 200]}


def state_with_pending(steps_run: int, observed_at_step: int, chunk) -> RunState:
    job = Future()
    job.set_result(chunk)
    state = RunState(last_action=np.array([52.0, 52.0]), pending=PendingChunk(job, observed_at_step))
    state.stats.steps = steps_run
    return state


class SeamTests(unittest.TestCase):
    def test_seam_is_the_largest_joint_change(self):
        self.assertEqual(seam_size(np.array([1.0, 5.0]), np.array([2.0, 1.0])), 4.0)

    def test_no_seams_give_no_summary(self):
        self.assertIsNone(summarize_seams([]))

    def test_summary_holds_count_median_percentile_and_maximum(self):
        summary = summarize_seams([1.0, 2.0, 3.0])
        self.assertEqual(summary, {"count": 3, "median": 2.0, "p90": 2.8, "max": 3.0})


class TrackMotionTests(unittest.TestCase):
    def test_first_pose_has_not_moved(self):
        state = RunState()
        track_motion(state, np.array(STILL_POSE), 3.0)
        self.assertFalse(state.has_moved)

    def test_change_within_the_threshold_is_not_a_move(self):
        state = RunState(first_pose=np.array(STILL_POSE))
        track_motion(state, np.array([53.0, 50.0]), 3.0)
        self.assertFalse(state.has_moved)

    def test_change_past_the_threshold_is_a_move(self):
        state = RunState(first_pose=np.array(STILL_POSE))
        track_motion(state, np.array([53.5, 50.0]), 3.0)
        self.assertTrue(state.has_moved)

    def test_arm_back_at_its_first_pose_still_counts_as_moved(self):
        state = RunState(first_pose=np.array(STILL_POSE), has_moved=True)
        track_motion(state, np.array(STILL_POSE), 3.0)
        self.assertTrue(state.has_moved)


class TakeBackgroundChunkTests(unittest.TestCase):
    def test_actions_for_the_steps_already_run_are_dropped(self):
        state = state_with_pending(steps_run=5, observed_at_step=2, chunk=make_chunk(60.0))
        take_background_chunk(state)
        self.assertEqual(state.actions[0].tolist(), [63.0, 63.0])

    def test_dropped_actions_are_counted(self):
        state = state_with_pending(steps_run=5, observed_at_step=2, chunk=make_chunk(60.0))
        take_background_chunk(state)
        self.assertEqual(state.stats.dropped_actions, 3)

    def test_last_action_is_kept_when_the_job_took_longer_than_its_chunk(self):
        state = state_with_pending(steps_run=500, observed_at_step=0, chunk=make_chunk(60.0, length=20))
        take_background_chunk(state)
        self.assertEqual(len(state.actions), 1)

    def test_seam_is_measured_from_the_action_taken_last_step(self):
        state = state_with_pending(steps_run=5, observed_at_step=2, chunk=make_chunk(60.0))
        take_background_chunk(state)
        self.assertEqual(state.seams, [11.0])

    def test_steps_since_observation_start_at_the_dropped_actions(self):
        state = state_with_pending(steps_run=5, observed_at_step=2, chunk=make_chunk(60.0))
        take_background_chunk(state)
        self.assertEqual(state.steps_since_observation, 3)

    def test_job_is_no_longer_pending(self):
        state = state_with_pending(steps_run=5, observed_at_step=2, chunk=make_chunk(60.0))
        take_background_chunk(state)
        self.assertIsNone(state.pending)

    def test_chunk_is_counted_with_its_inference_time(self):
        state = state_with_pending(steps_run=5, observed_at_step=2, chunk=make_chunk(60.0, seconds=0.7))
        take_background_chunk(state)
        self.assertEqual((state.stats.chunks, state.stats.max_infer_s), (1, 0.7))


class ChunkBookkeepingTests(unittest.TestCase):
    def test_longest_inference_is_kept(self):
        state = RunState()
        start_chunk(state, make_chunk(seconds=0.7))
        start_chunk(state, make_chunk(seconds=0.2))
        self.assertEqual(state.stats.max_infer_s, 0.7)

    def test_popped_action_is_remembered_for_the_next_seam(self):
        state = RunState()
        start_chunk(state, make_chunk(60.0))
        action = pop_next_action(state)
        self.assertIs(state.last_action, action)

    def test_raw_actions_stay_in_step_with_the_actions(self):
        state = RunState()
        start_chunk(state, make_chunk(60.0, length=5))
        pop_next_action(state)
        self.assertEqual(len(state.raw_actions), len(state.actions))


class ReplanTimingTests(unittest.TestCase):
    def test_without_replan_option_it_is_never_due(self):
        state = RunState(has_moved=True, steps_since_observation=100)
        self.assertFalse(is_replan_due(state, None))

    def test_still_arm_waits(self):
        state = RunState(has_moved=False, steps_since_observation=100)
        self.assertFalse(is_replan_due(state, 5))

    def test_one_job_at_a_time(self):
        state = RunState(has_moved=True, steps_since_observation=100, pending=PendingChunk(Future(), 0))
        self.assertFalse(is_replan_due(state, 5))

    def test_too_few_steps_wait(self):
        state = RunState(has_moved=True, steps_since_observation=4)
        self.assertFalse(is_replan_due(state, 5))

    def test_due_after_the_replan_steps(self):
        state = RunState(has_moved=True, steps_since_observation=5)
        self.assertTrue(is_replan_due(state, 5))


class RtcGuidanceTests(unittest.TestCase):
    def test_delay_is_the_largest_of_the_recent_ones(self):
        state = RunState(last_infer_seconds=9.0)
        state.recent_delays.extend([3, 7, 5])
        self.assertEqual(rtc_guidance(state, FPS)["inference_delay"], 7)

    def test_delay_without_a_history_comes_from_the_inference_time(self):
        state = RunState(last_infer_seconds=0.3)
        self.assertEqual(rtc_guidance(state, FPS)["inference_delay"], 3)

    def test_only_the_latest_delays_count(self):
        state = RunState()
        state.recent_delays.extend([99, 1, 1, 1, 1, 1])
        self.assertEqual(rtc_guidance(state, FPS)["inference_delay"], 1)

    def test_left_over_is_what_remains_of_the_running_chunk(self):
        state = RunState(raw_actions=np.ones((4, 2)))
        self.assertIs(rtc_guidance(state, FPS)["prev_chunk_left_over"], state.raw_actions)


class DriveArmTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.start_patch(patch.object(arm_control.time, "perf_counter", self.clock.perf_counter))
        self.start_patch(patch.object(arm_control, "precise_sleep", self.clock.sleep))
        self.background = InlineExecutor()

    def start_patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_loop(self, robot, plan, args, view=None):
        run = ArmRun(
            robot=robot,
            plan=plan,
            background=self.background,
            args=args,
            facts=make_facts(),
            view=view or PolicyView(None, None),
            video=make_video(),
        )
        state = RunState()
        silently(drive_arm, run, state)
        return state

    def test_one_action_is_sent_per_frame(self):
        robot = FakeRobot()
        self.run_loop(robot, MagicMock(return_value=make_chunk()), make_args(seconds=2))
        self.assertEqual(len(robot.sent), 2 * FPS)

    def test_actions_go_out_by_joint_name(self):
        robot = FakeRobot()
        self.run_loop(robot, MagicMock(return_value=make_chunk(60.0)), make_args())
        self.assertEqual(robot.sent[1], {SHOULDER: 61.0, ELBOW: 61.0})

    def test_commands_are_clipped_to_the_envelope(self):
        robot = FakeRobot()
        self.run_loop(robot, MagicMock(return_value=make_chunk(99.0)), make_args())
        self.assertEqual(robot.sent[3][SHOULDER], 100.0)

    def test_clipped_steps_are_counted(self):
        state = self.run_loop(FakeRobot(), MagicMock(return_value=make_chunk(99.0)), make_args())
        self.assertEqual(state.stats.clipped_steps, FPS - 2)

    def test_next_chunk_is_computed_when_the_action_steps_run_out(self):
        plan = MagicMock(return_value=make_chunk())
        self.run_loop(FakeRobot(), plan, make_args(n_action_steps=3))
        self.assertEqual(plan.call_count, 3)

    def test_only_the_action_steps_of_a_chunk_run(self):
        robot = FakeRobot()
        self.run_loop(robot, MagicMock(return_value=make_chunk(60.0)), make_args(n_action_steps=3))
        self.assertEqual(robot.sent[3][SHOULDER], 60.0)

    def test_replanning_waits_until_the_arm_has_moved(self):
        plan = MagicMock(return_value=make_chunk())
        self.run_loop(FakeRobot(), plan, make_args(replan=2))
        self.assertEqual(self.background.submitted_guidance, [])

    def test_moved_arm_gets_its_next_chunk_from_the_background(self):
        plan = MagicMock(side_effect=[make_chunk(50.0), make_chunk(60.0), make_chunk(70.0), make_chunk(80.0)])
        robot = FakeRobot((STILL_POSE, MOVED_POSE))
        self.run_loop(robot, plan, make_args(replan=2))
        self.assertEqual(self.background.submitted_guidance[0], {})

    def test_stale_action_of_a_background_chunk_is_skipped(self):
        plan = MagicMock(side_effect=[make_chunk(50.0), make_chunk(60.0), make_chunk(70.0), make_chunk(80.0)])
        robot = FakeRobot((STILL_POSE, MOVED_POSE))
        self.run_loop(robot, plan, make_args(replan=2))
        self.assertEqual(robot.sent[3][SHOULDER], 61.0)

    def test_takeover_records_the_jump_from_the_last_action(self):
        plan = MagicMock(side_effect=[make_chunk(50.0), make_chunk(60.0), make_chunk(70.0), make_chunk(80.0)])
        robot = FakeRobot((STILL_POSE, MOVED_POSE))
        state = self.run_loop(robot, plan, make_args(replan=2))
        self.assertEqual(state.seams[0], 9.0)

    def test_rtc_guides_the_background_chunk(self):
        plan = MagicMock(side_effect=[make_chunk(50.0), make_chunk(60.0), make_chunk(70.0), make_chunk(80.0)])
        robot = FakeRobot((STILL_POSE, MOVED_POSE))
        self.run_loop(robot, plan, make_args(replan=2, rtc=True))
        self.assertEqual(self.background.submitted_guidance[0]["inference_delay"], 2)

    def test_slow_step_counts_as_late(self):
        robot = FakeRobot()
        robot.before_observation = lambda: self.clock.sleep(0.5)
        state = self.run_loop(robot, MagicMock(return_value=make_chunk()), make_args())
        self.assertEqual(state.stats.late_steps, 2)

    def test_interrupted_loop_leaves_what_it_counted_in_the_state(self):
        robot = FakeRobot()
        robot.failing_send, robot.send_error = 3, KeyboardInterrupt()
        run = ArmRun(
            robot=robot,
            plan=MagicMock(return_value=make_chunk()),
            background=None,
            args=make_args(),
            facts=make_facts(),
            view=PolicyView(None, None),
            video=make_video(),
        )
        state = RunState()
        with self.assertRaises(KeyboardInterrupt):
            drive_arm(run, state)
        self.assertEqual(state.stats.steps, 2)

    def test_outline_of_the_crop_stays_out_of_the_model_frame(self):
        plan = MagicMock(return_value=make_chunk())
        self.run_loop(
            FakeRobot(),
            plan,
            make_args(),
            PolicyView(None, OLD_CROP),
        )
        self.assertEqual(plan.call_args.args[0]["observation.images.top"].max(), 0)

    def test_outline_of_the_crop_reaches_the_video(self):
        video = make_video()
        run = ArmRun(
            robot=FakeRobot(),
            plan=MagicMock(return_value=make_chunk()),
            background=None,
            args=make_args(),
            facts=make_facts(),
            view=PolicyView(None, OLD_CROP),
            video=video,
        )
        silently(drive_arm, run, RunState())
        self.assertGreater(video.stream.encode.call_args.args[0].to_ndarray().max(), 0)


if __name__ == "__main__":
    unittest.main()
