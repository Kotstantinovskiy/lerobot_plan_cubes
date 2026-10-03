"""The laptop's side of a policy on another machine: what run_policy.py --server and policy_server.py exchange.

Each message is a numpy archive, so nothing that arrives is unpickled. pi0.5 frames are shrunk to the model's 224x224
and every camera image is sent as a JPEG: about 50 KB instead of 7 MB, which an SSH tunnel carries in ~20 ms.
"""

import io
import json
import time
from multiprocessing.connection import Client
from typing import Any, NamedTuple

import cv2
import numpy as np

import pi05_inputs

ANSWER_TIMEOUT_S = 30  # the first chunk after a start takes seconds; much longer means the link or the server is gone
JPEG_QUALITY = 95


class PlannedChunk(NamedTuple):
    """One action chunk (chunk_size x joints), wherever it was computed."""

    actions: np.ndarray  # in dataset units
    raw_actions: Any  # as the model gave them (normalized), what RTC compares; an array or a tensor on its device
    seconds: float  # how long computing it took


def pack(**arrays) -> bytes:
    """Arrays and numbers as one message; not compressed, since camera images barely shrink and cost time."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def unpack(message: bytes) -> dict:
    """The arrays of a message from pack, by name; numbers come back as 0-dimensional arrays."""
    with np.load(io.BytesIO(message), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def encode_cameras(frame: dict) -> dict:
    """A dataset frame with its RGB camera images (HWC uint8) as JPEG bytes."""
    encoded = dict(frame)
    for key, image in frame.items():
        if key.startswith(pi05_inputs.CAMERA_PREFIX):
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            encoded[key] = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
    return encoded


def decode_cameras(frame: dict) -> dict:
    """The reverse of encode_cameras."""
    decoded = dict(frame)
    for key, jpeg in frame.items():
        if key.startswith(pi05_inputs.CAMERA_PREFIX):
            decoded[key] = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return decoded


class RemotePolicy:
    """A connection to policy_server.py. `info` holds what the server knows and the laptop needs: the policy's type,
    target box and crop settings, and the dataset's features, fps and joint envelope."""

    def __init__(self, address: str):
        host, port = address.rsplit(":", 1)
        try:
            self.connection = Client((host, int(port)))
        except ConnectionError as error:
            raise RuntimeError(f"No policy server at {address}: is policy_server.py running, the tunnel up?") from error
        self.info = json.loads(self._receive())

    def _receive(self) -> bytes:
        """The next message. Raises TimeoutError if none comes in time."""
        if not self.connection.poll(ANSWER_TIMEOUT_S):
            raise TimeoutError(f"The policy server has not answered for {ANSWER_TIMEOUT_S} s")
        return self.connection.recv_bytes()

    def plan(self, frame: dict, task: str, **guidance) -> PlannedChunk:
        """The action chunk for a dataset frame. `guidance`: RTC's inference_delay and prev_chunk_left_over."""
        start = time.perf_counter()
        if self.info["type"] == pi05_inputs.POLICY_TYPE:
            frame = pi05_inputs.shrink_cameras(frame)
        self.connection.send_bytes(pack(task=task, **encode_cameras(frame), **guidance))
        answer = unpack(self._receive())
        return PlannedChunk(answer["chunk"], answer["raw"], time.perf_counter() - start)

    def recorded_observation(self, episode: int, offset_s: float) -> dict:
        """A recorded frame from the server's dataset, as robot.get_observation() gives it, for --dry-run."""
        self.connection.send_bytes(pack(recorded_episode=episode, recorded_offset_s=offset_s))
        return unpack(self._receive())
