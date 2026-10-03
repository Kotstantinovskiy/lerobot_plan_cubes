"""policy_server.py's answers, with a fake policy and connection: no checkpoint, port or tunnel is touched."""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import policy_server  # noqa: E402
from remote_policy import pack, unpack  # noqa: E402

DEVICE = torch.device("cpu")
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": [6]},
    "observation.images.top": {"dtype": "video", "shape": [4, 6, 3]},
    "action": {"dtype": "float32", "shape": [6]},
}


def make_compute():
    """compute_chunk as the server fills it in: two forms of a 5-step chunk."""
    return MagicMock(return_value=(np.full((5, 6), 7.0), torch.ones(5, 6)))


class BlankFrameTests(unittest.TestCase):
    def test_cameras_are_uint8(self):
        self.assertEqual(policy_server.make_blank_frame(FEATURES)["observation.images.top"].dtype, np.uint8)

    def test_the_rest_is_float32(self):
        self.assertEqual(policy_server.make_blank_frame(FEATURES)["observation.state"].dtype, np.float32)

    def test_only_observations_are_in_the_frame(self):
        self.assertNotIn("action", policy_server.make_blank_frame(FEATURES))


class WarmUpTests(unittest.TestCase):
    def test_a_fresh_process_computes_two_blank_chunks(self):
        compute = make_compute()
        policy_server.warm_up(compute, {"blank": 0})
        self.assertEqual(compute.call_count, policy_server.WARM_UP_CHUNKS)

    def test_warm_up_gives_the_last_chunk_as_the_model_gave_it(self):
        compute = make_compute()
        raw_chunk = policy_server.warm_up(compute, {"blank": 0})
        self.assertEqual(raw_chunk.shape, (5, 6))

    def test_guided_warm_up_continues_after_the_delay(self):
        compute = make_compute()
        policy_server.warm_up_guidance(compute, {"blank": 0}, torch.ones(50, 6))
        left_over = compute.call_args.kwargs["prev_chunk_left_over"]
        self.assertEqual(left_over.shape[0], 50 - policy_server.WARM_UP_RTC_DELAY)


class ClientInfoTests(unittest.TestCase):
    def make_info(self, saved):
        args = SimpleNamespace(policy="/srv/policy", rtc=True)
        dataset_stats = {"action": {"min": np.zeros(2), "max": np.ones(2)}}
        ds_meta = SimpleNamespace(features=FEATURES, fps=20, stats=dataset_stats)
        with (
            patch.object(policy_server, "saved_stats", return_value=saved),
            patch.object(policy_server, "target_box_style", return_value={"thickness": 6}),
            patch.object(policy_server.top_crop, "load", return_value=None),
        ):
            return policy_server.make_client_info(args, "smolvla", ds_meta)

    def test_envelope_comes_from_the_checkpoint_statistics(self):
        info = self.make_info({"action": {"min": np.full(2, 3.0), "max": np.full(2, 4.0)}})
        self.assertEqual((info["low"], info["high"]), ([3.0, 3.0], [4.0, 4.0]))

    def test_base_policy_gets_the_envelope_of_the_dataset(self):
        info = self.make_info(None)
        self.assertEqual((info["low"], info["high"]), ([0.0, 0.0], [1.0, 1.0]))

    def test_info_can_travel_as_json(self):
        info = self.make_info(None)
        self.assertEqual(json.loads(json.dumps(info))["fps"], 20)


class AnswerTests(unittest.TestCase):
    def make_chunk_request(self, **guidance):
        jpeg = cv2.imencode(".jpg", np.zeros((4, 6, 3), np.uint8))[1]
        return {
            "task": np.array("Pick up"),
            "observation.state": np.zeros(6, np.float32),
            "observation.images.top": jpeg,
            **guidance,
        }

    def answer(self, compute, request):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            message = policy_server.answer_chunk(request, compute, DEVICE)
        return unpack(message), printed.getvalue()

    def test_answer_holds_both_forms_of_the_chunk(self):
        answer, _ = self.answer(make_compute(), self.make_chunk_request())
        self.assertEqual((answer["chunk"][0, 0], answer["raw"][0, 0]), (7.0, 1.0))

    def test_task_reaches_the_model_as_text(self):
        compute = make_compute()
        self.answer(compute, self.make_chunk_request())
        self.assertEqual(compute.call_args.kwargs["task"], "Pick up")

    def test_camera_image_is_decoded_for_the_model(self):
        compute = make_compute()
        self.answer(compute, self.make_chunk_request())
        self.assertEqual(compute.call_args.kwargs["frame"]["observation.images.top"].shape, (4, 6, 3))

    def test_plain_chunk_is_not_called_guided(self):
        _, printed = self.answer(make_compute(), self.make_chunk_request())
        self.assertNotIn("(guided)", printed)

    def test_guided_chunk_says_so(self):
        request = self.make_chunk_request(inference_delay=np.array(3))
        _, printed = self.answer(make_compute(), request)
        self.assertIn("(guided)", printed)

    def test_guidance_is_taken_out_of_the_request(self):
        request = {"inference_delay": np.array(3), "prev_chunk_left_over": np.ones((2, 6)), "kept": 1}
        guidance = policy_server.pop_guidance(request, DEVICE)
        self.assertEqual((request, guidance["inference_delay"]), ({"kept": 1}, 3))

    def test_left_over_becomes_a_tensor(self):
        request = {"prev_chunk_left_over": np.ones((2, 6))}
        self.assertIsInstance(policy_server.pop_guidance(request, DEVICE)["prev_chunk_left_over"], torch.Tensor)

    def test_request_without_guidance_gives_none(self):
        self.assertEqual(policy_server.pop_guidance({"kept": 1}, DEVICE), {})

    def answer_recorded(self):
        request = {"recorded_episode": np.array(2), "recorded_offset_s": np.array(3.5)}
        with patch.object(policy_server, "dataset_observation", return_value={"top": np.zeros(2)}) as observation:
            message = policy_server.answer_recorded_frame(request, "metadata", "root")
        return message, observation

    def test_recorded_frame_comes_from_the_asked_episode_and_offset(self):
        _, observation = self.answer_recorded()
        observation.assert_called_once_with(
            "metadata",
            "root",
            2,
            3.5,
        )

    def test_recorded_frame_is_sent_as_a_message(self):
        message, _ = self.answer_recorded()
        self.assertEqual(list(unpack(message)), ["top"])


class ServeClientTests(unittest.TestCase):
    def serve(self, connection, answer):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            policy_server.serve_client(connection, {"fps": 20}, answer)
        return printed.getvalue()

    def test_client_is_told_the_info_first(self):
        connection = MagicMock()
        connection.recv_bytes.side_effect = EOFError
        self.serve(connection, MagicMock())
        connection.send_bytes.assert_called_once_with(b'{"fps": 20}')

    def test_each_request_gets_an_answer(self):
        connection = MagicMock()
        connection.recv_bytes.side_effect = [pack(number=1), EOFError]
        self.serve(connection, MagicMock(return_value=b"answer"))
        connection.send_bytes.assert_called_with(b"answer")

    def test_client_that_leaves_is_not_an_error(self):
        connection = MagicMock()
        connection.recv_bytes.side_effect = EOFError
        self.assertIn("Client left", self.serve(connection, MagicMock()))

    def test_lost_connection_is_not_an_error(self):
        connection = MagicMock()
        connection.recv_bytes.side_effect = ConnectionResetError
        self.assertIn("Client left", self.serve(connection, MagicMock()))


if __name__ == "__main__":
    unittest.main()
