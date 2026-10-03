"""Camera snapshots share pairing authentication and cannot disrupt control."""

import http.client
import json
import threading

import pytest
from websockets.sync.client import connect
from websockets.sync.server import serve

from viewer import ViewerServer

TOKEN = "c" * 64
JPEG = b"\xff\xd8camera frame\xff\xd9"
DISCONNECTED = {"wrist": {"connected": False}, "top": {"connected": False}}


class FakeCameras:
    def __init__(self):
        self.frames = {"wrist": None, "top": None}
        self.broken = False
        self.frame_requests = []

    def status(self):
        if self.broken:
            raise RuntimeError("Private device failure")
        return {name: {"connected": frame is not None} for name, frame in self.frames.items()}

    def frame(self, name):
        self.frame_requests.append(name)
        if self.broken:
            raise RuntimeError("Private device failure")
        return self.frames[name]


@pytest.fixture
def camera_viewer(tmp_path):
    index = tmp_path / "index.html"
    index.write_text("viewer")
    cameras = FakeCameras()
    viewer = ViewerServer(12345, 0, index, token=TOKEN, cameras=cameras)
    viewer.start()
    try:
        yield viewer, cameras
    finally:
        viewer.stop()


def request(viewer, path, *, token=TOKEN, origin=None, host=None, method="GET"):
    if token is not None:
        path += "?token=" + token
    headers = {}
    if origin is not None:
        headers["Origin"] = origin
    if host is not None:
        headers["Host"] = host
    connection = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=2)
    try:
        connection.request(method, path, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_both_unavailable_cameras_remain_visible_in_status(camera_viewer):
    viewer, _ = camera_viewer
    status, headers, body = request(viewer, "/cameras/status")
    assert status == 200
    assert json.loads(body) == DISCONNECTED
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    for name in ("wrist", "top"):
        status, headers, body = request(viewer, f"/cameras/{name}/frame")
        assert status == 503
        assert body == b"Camera is not connected"
        assert headers["Cache-Control"] == "no-store"


def test_no_capture_service_defaults_to_both_disconnected(camera_viewer):
    viewer, _ = camera_viewer
    viewer.cameras = None
    assert json.loads(request(viewer, "/cameras/status")[2]) == DISCONNECTED
    assert request(viewer, "/cameras/wrist/frame")[0] == 503
    assert request(viewer, "/cameras/top/frame")[0] == 503
    assert request(viewer, "/health")[0] == 200


@pytest.mark.parametrize("name", ["wrist", "top"])
def test_camera_disconnect_and_recovery_never_reuse_previous_frame(camera_viewer, name):
    viewer, cameras = camera_viewer
    other = "top" if name == "wrist" else "wrist"
    cameras.frames[name] = JPEG
    status, headers, body = request(viewer, f"/cameras/{name}/frame")
    assert status == 200
    assert body == JPEG
    assert headers["Content-Type"] == "image/jpeg"
    assert headers["Content-Length"] == str(len(JPEG))
    assert json.loads(request(viewer, "/cameras/status")[2]) == {
        name: {"connected": True}, other: {"connected": False},
    }
    assert request(viewer, f"/cameras/{other}/frame")[0] == 503

    cameras.frames[name] = None
    assert request(viewer, f"/cameras/{name}/frame")[0] == 503
    assert json.loads(request(viewer, "/cameras/status")[2]) == DISCONNECTED

    cameras.frames[name] = JPEG + b"new"
    assert request(viewer, f"/cameras/{name}/frame")[2] == JPEG + b"new"


@pytest.mark.parametrize("path", ["/cameras/status", "/cameras/wrist/frame", "/cameras/top/frame"])
@pytest.mark.parametrize("token", [None, "wrong", TOKEN + "&token=" + TOKEN])
def test_camera_routes_require_the_same_pairing_token(camera_viewer, path, token):
    viewer, cameras = camera_viewer
    cameras.frames["wrist"] = JPEG
    status, headers, body = request(viewer, path, token=token)
    assert status == 401
    assert JPEG not in body
    assert not cameras.frame_requests
    assert "Location" not in headers


@pytest.mark.parametrize("path", ["/cameras/status", "/cameras/wrist/frame", "/cameras/top/frame"])
def test_camera_routes_reject_foreign_origin_and_invalid_host(camera_viewer, path):
    viewer, cameras = camera_viewer
    assert request(viewer, path, origin="http://attacker.invalid")[0] == 403
    assert request(viewer, path, host="attacker.invalid")[0] == 400
    assert not cameras.frame_requests


def test_camera_data_is_not_cached_or_exposed_to_other_origins(camera_viewer):
    viewer, cameras = camera_viewer
    cameras.frames["wrist"] = JPEG
    origin = f"http://127.0.0.1:{viewer.port}"
    for path in ("/cameras/status", "/cameras/wrist/frame"):
        status, headers, _ = request(viewer, path, origin=origin)
        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert "Access-Control-Allow-Origin" not in headers


@pytest.mark.parametrize("path", [
    "/cameras/unknown/frame", "/cameras/wrist", "/cameras/top/frame/", "/cameras/devices",
    "/cameras/../config.json", "/cameras/%2e%2e/config.json", "/cameras/wrist/frame.jpg",
])
def test_camera_routes_do_not_allow_device_enumeration_or_arbitrary_paths(camera_viewer, path):
    viewer, cameras = camera_viewer
    assert request(viewer, path)[0] == 404
    assert not cameras.frame_requests


@pytest.mark.parametrize("path", ["/cameras/status", "/cameras/wrist/frame"])
def test_camera_head_matches_get_without_a_response_body(camera_viewer, path):
    viewer, cameras = camera_viewer
    cameras.frames["wrist"] = JPEG
    _, get_headers, _ = request(viewer, path)
    status, headers, body = request(viewer, path, method="HEAD")
    assert status == 200
    assert headers["Content-Length"] == get_headers["Content-Length"]
    assert body == b""


@pytest.mark.parametrize("frame", [b"", "not bytes", None])
def test_missing_or_invalid_snapshots_return_unavailable(camera_viewer, frame):
    viewer, cameras = camera_viewer
    cameras.frames["wrist"] = frame
    assert request(viewer, "/cameras/wrist/frame")[0] == 503


def test_broken_capture_service_cannot_interrupt_a_robot_websocket(camera_viewer):
    viewer, cameras = camera_viewer

    def echo(connection):
        for message in connection:
            connection.send(message)

    with serve(echo, "127.0.0.1", 0) as backend:
        worker = threading.Thread(target=backend.serve_forever, daemon=True)
        worker.start()
        viewer.websocket_port = backend.socket.getsockname()[1]
        try:
            address = f"ws://127.0.0.1:{viewer.port}/ws?token={TOKEN}"
            with connect(address, proxy=None, open_timeout=2, close_timeout=1) as connection:
                connection.send("before camera failure")
                assert connection.recv(timeout=2) == "before camera failure"
                cameras.broken = True
                assert request(viewer, "/cameras/wrist/frame")[0] == 503
                assert request(viewer, "/cameras/top/frame")[0] == 503
                assert json.loads(request(viewer, "/cameras/status")[2]) == DISCONNECTED
                assert request(viewer, "/health")[0] == 200
                connection.send("after camera failure")
                assert connection.recv(timeout=2) == "after camera failure"
        finally:
            backend.shutdown()
            worker.join(timeout=2)
