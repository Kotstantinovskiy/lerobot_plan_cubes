"""What run_policy.py --server and policy_server.py exchange; the connection is a fake, so no port is opened."""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import remote_policy  # noqa: E402
from remote_policy import RemotePolicy, decode_cameras, encode_cameras, pack, unpack  # noqa: E402

CAMERA = "observation.images.top"
JPEG_LOSS_LEVELS = 3
INFO = {"type": "smolvla", "fps": 20}


def make_connection(*answers):
    connection = MagicMock()
    connection.recv_bytes.side_effect = [json.dumps(INFO).encode(), *answers]
    return connection


def connect(connection, info=None) -> RemotePolicy:
    with patch.object(remote_policy, "Client", return_value=connection):
        remote = RemotePolicy("gpu-host:18765")
    if info:
        remote.info = info
    return remote


class MessageTests(unittest.TestCase):
    def test_arrays_survive_the_trip(self):
        message = pack(numbers=np.arange(3))
        self.assertEqual(unpack(message)["numbers"].tolist(), [0, 1, 2])

    def test_numbers_come_back_as_arrays_without_dimensions(self):
        self.assertEqual(unpack(pack(delay=3))["delay"].shape, ())

    def test_message_holding_a_pickle_is_refused(self):
        buffer = io.BytesIO()
        np.savez(buffer, code=np.array([object()], dtype=object))
        with self.assertRaises(ValueError):
            unpack(buffer.getvalue())


class CameraEncodingTests(unittest.TestCase):
    def test_cameras_come_back_nearly_as_they_were(self):
        frame = {CAMERA: np.full((16, 16, 3), (200, 100, 50), np.uint8)}
        decoded = decode_cameras(encode_cameras(frame))
        self.assertLessEqual(np.abs(decoded[CAMERA].astype(int) - frame[CAMERA]).max(), JPEG_LOSS_LEVELS)

    def test_camera_turns_into_fewer_bytes_than_its_pixels(self):
        frame = {CAMERA: np.zeros((64, 64, 3), np.uint8)}
        self.assertLess(encode_cameras(frame)[CAMERA].size, frame[CAMERA].size)

    def test_other_features_are_left_as_they_are(self):
        state = np.zeros(6)
        self.assertIs(encode_cameras({"observation.state": state})["observation.state"], state)


class RemotePolicyTests(unittest.TestCase):
    def test_server_that_does_not_answer_the_connection_says_what_to_check(self):
        with (
            patch.object(remote_policy, "Client", side_effect=ConnectionRefusedError),
            self.assertRaisesRegex(RuntimeError, "is policy_server.py running, the tunnel up"),
        ):
            RemotePolicy("gpu-host:18765")

    def test_port_is_a_number(self):
        with patch.object(remote_policy, "Client", return_value=make_connection()) as client:
            RemotePolicy("gpu-host:18765")
        client.assert_called_once_with(("gpu-host", 18765))

    def test_info_comes_first(self):
        self.assertEqual(connect(make_connection()).info, INFO)

    def test_silent_server_times_out(self):
        connection = make_connection()
        remote = connect(connection)
        connection.poll.return_value = False
        with self.assertRaisesRegex(TimeoutError, "has not answered"):
            remote.recorded_observation(episode=1, offset_s=3)

    def test_plan_gives_the_chunk_by_name(self):
        answer = pack(chunk=np.ones((2, 6)), raw=np.zeros((2, 6)))
        planned = connect(make_connection(answer)).plan({}, "Pick up")
        self.assertEqual((planned.actions[0, 0], planned.raw_actions[0, 0]), (1.0, 0.0))

    def test_plan_sends_the_task_and_the_guidance(self):
        connection = make_connection(pack(chunk=np.ones(1), raw=np.ones(1)))
        connect(connection).plan({}, "Pick up", inference_delay=3)
        request = unpack(connection.send_bytes.call_args.args[0])
        self.assertEqual((str(request["task"]), int(request["inference_delay"])), ("Pick up", 3))

    def test_pi05_frames_are_shrunk_before_they_leave(self):
        connection = make_connection(pack(chunk=np.ones(1), raw=np.ones(1)))
        remote = connect(connection, info={"type": "pi05"})
        with patch.object(remote_policy.pi05_inputs, "shrink_cameras", return_value={}) as shrink:
            remote.plan({CAMERA: np.zeros((8, 8, 3), np.uint8)}, "Pick up")
        shrink.assert_called_once()

    def test_recorded_frame_asks_for_the_episode_and_offset(self):
        connection = make_connection(pack(top=np.zeros(1)))
        connect(connection).recorded_observation(episode=1, offset_s=3)
        request = unpack(connection.send_bytes.call_args.args[0])
        self.assertEqual((int(request["recorded_episode"]), int(request["recorded_offset_s"])), (1, 3))


if __name__ == "__main__":
    unittest.main()
