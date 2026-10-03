"""Pairing authentication and WebSocket relay without a robot or external network."""

import contextlib
import http.client
import json
import socket
import threading
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect
from websockets.sync.server import serve

from viewer import ViewerServer

TOKEN = "a" * 64


@pytest.fixture
def echo_backend():
    opened = []
    closed = threading.Event()

    def echo(connection):
        opened.append(connection)
        try:
            for message in connection:
                connection.send(message)
        finally:
            closed.set()

    with serve(
        echo,
        "127.0.0.1",
        0,
        subprotocols=["viser-v1.1.1"],
        select_subprotocol=lambda connection, offered: "viser-v1.1.1" if "viser-v1.1.1" in offered else None,
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1], opened, closed
        finally:
            server.shutdown()
            thread.join(timeout=2)


@pytest.fixture
def paired_viewer(tmp_path, echo_backend):
    index = tmp_path / "index.html"
    index.write_text("<!doctype html><title>Paired Teleoperator</title>")
    server = ViewerServer(echo_backend[0], 0, index, token=TOKEN, advertised_host="Robot-Mac.local")
    server.start()
    try:
        yield server
    finally:
        server.stop()


def request(server, path, *, host=None, origin=None, method="GET"):
    headers = {}
    if host is not None:
        headers["Host"] = host
    if origin is not None:
        headers["Origin"] = origin
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=2)
    try:
        connection.request(method, path, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_lan_binding_requires_pairing_and_rejects_unsafe_parameters(tmp_path):
    with pytest.raises(ValueError, match="pairing token"):
        ViewerServer(12345, 0, host="0.0.0.0")
    for token in ("short", "a" * 32 + "&admin=yes", "a" * 257):
        with pytest.raises(ValueError, match="Pairing token"):
            ViewerServer(12345, 0, token=token)
    for host in ("example.com", "8.8.8.8", "example.com.local/", "desk.local:8080", "a..local"):
        with pytest.raises(ValueError, match="Advertised host"):
            ViewerServer(12345, 0, advertised_host=host)


@pytest.mark.parametrize(
    "path", ["/", "/health", "/ws", "/?token=wrong", "/?token=%E2%98%83", f"/health?token={TOKEN}&token={TOKEN}"]
)
def test_authentication_rejects_missing_wrong_or_duplicate_token(paired_viewer, echo_backend, path):
    status, headers, body = request(paired_viewer, path)
    assert status == 401
    assert TOKEN.encode() not in body
    assert "Location" not in headers
    assert not echo_backend[1]


def test_pairing_url_and_canonical_redirect_keep_mac_host_and_same_port(paired_viewer):
    assert paired_viewer.url == f"http://127.0.0.1:{paired_viewer.port}/?token={TOKEN}"
    assert paired_viewer.pairing_url == f"http://Robot-Mac.local:{paired_viewer.port}/?token={TOKEN}"
    host = f"Robot-Mac.local:{paired_viewer.port}"
    status, headers, _ = request(paired_viewer, f"/?token={TOKEN}&websocket=ws://evil.invalid", host=host)
    assert status == 302
    params = parse_qs(urlsplit(headers["Location"]).query, keep_blank_values=True)
    assert params == {
        "token": [TOKEN],
        "websocket": [f"ws://robot-mac.local:{paired_viewer.port}/ws?token={TOKEN}"],
        "darkMode": [""],
        "hideViserLogo": [""],
    }
    status, headers, body = request(paired_viewer, headers["Location"], host=host)
    assert status == 200
    assert body == paired_viewer.index_path.read_bytes()
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in headers


def test_health_and_head_are_authenticated_and_do_not_send_a_head_body(paired_viewer):
    path = f"/health?token={TOKEN}"
    status, headers, body = request(paired_viewer, path)
    assert status == 200
    assert json.loads(body) == {"name": "Teleoperator", "version": 1}
    status, head_headers, body = request(paired_viewer, path, method="HEAD")
    assert status == 200
    assert body == b""
    assert head_headers["Content-Length"] == headers["Content-Length"]


@pytest.mark.parametrize("host", ["attacker.invalid", "8.8.8.8", "127.0.0.1:1", "user@robot.local", "desk.local/path"])
def test_host_header_cannot_select_a_public_or_different_origin(paired_viewer, host):
    assert request(paired_viewer, f"/?token={TOKEN}", host=host)[0] == 400


def test_http_origin_must_match_the_local_page(paired_viewer):
    assert request(paired_viewer, f"/health?token={TOKEN}", origin="https://attacker.invalid")[0] == 403
    origin = f"http://127.0.0.1:{paired_viewer.port}"
    assert request(paired_viewer, f"/health?token={TOKEN}", origin=origin)[0] == 200


def test_origin_default_port_is_normalized_without_accepting_other_url_parts():
    assert ViewerServer._same_origin("http://desk.local", "desk.local:80")
    assert ViewerServer._same_origin("http://desk.local:80", "desk.local:80")
    for origin in (
        "http://user@desk.local",
        "http://desk.local/",
        "http://desk.local?token=secret",
        "https://desk.local",
    ):
        assert not ViewerServer._same_origin(origin, "desk.local:80")


@pytest.mark.parametrize("path", ["/../index.html", "/%2e%2e/index.html", "/other.html", "/viewer.py"])
def test_authenticated_requests_cannot_read_other_files(paired_viewer, path):
    assert request(paired_viewer, path + f"?token={TOKEN}")[0] == 404


def test_proxy_round_trips_binary_and_text_on_the_public_port(paired_viewer, echo_backend):
    address = f"ws://127.0.0.1:{paired_viewer.port}/ws?token={TOKEN}"
    origin = f"http://127.0.0.1:{paired_viewer.port}"
    with connect(address, origin=origin, proxy=None, open_timeout=2, close_timeout=1) as connection:
        connection.send(b"\x00\x01robot frame")
        assert connection.recv(timeout=2) == b"\x00\x01robot frame"
        connection.send("joint command")
        assert connection.recv(timeout=2) == "joint command"
    assert len(echo_backend[1]) == 1
    assert echo_backend[2].wait(timeout=2)


def test_proxy_negotiates_the_viser_version_protocol_on_both_connections(paired_viewer, echo_backend):
    address = f"ws://127.0.0.1:{paired_viewer.port}/ws?token={TOKEN}"
    with connect(address, subprotocols=["viser-v1.1.1"], proxy=None, open_timeout=2, close_timeout=1) as connection:
        connection.send("versioned")
        assert connection.recv(timeout=2) == "versioned"
        assert connection.subprotocol == "viser-v1.1.1"
        assert echo_backend[1][0].subprotocol == "viser-v1.1.1"


@pytest.mark.parametrize(
    "token,origin,status",
    [
        ("wrong", None, 401),
        (TOKEN, "https://attacker.invalid", 403),
        (TOKEN, "null", 403),
    ],
)
def test_rejected_websocket_never_connects_to_the_robot(paired_viewer, echo_backend, token, origin, status):
    address = f"ws://127.0.0.1:{paired_viewer.port}/ws?token={token}"
    with pytest.raises(InvalidStatus) as error, connect(address, origin=origin, proxy=None, open_timeout=2):
        pass
    assert error.value.response.status_code == status
    assert not echo_backend[1]


def test_native_client_without_origin_can_connect_with_valid_token(paired_viewer):
    address = f"ws://127.0.0.1:{paired_viewer.port}/ws?token={TOKEN}"
    with connect(address, proxy=None, open_timeout=2, close_timeout=1) as connection:
        connection.send("native")
        assert connection.recv(timeout=2) == "native"


def test_plain_http_cannot_open_a_backend_websocket(paired_viewer, echo_backend):
    assert request(paired_viewer, f"/ws?token={TOKEN}")[0] == 400
    assert not echo_backend[1]


def test_stop_closes_active_relays_and_releases_the_port(paired_viewer, echo_backend):
    address = f"ws://127.0.0.1:{paired_viewer.port}/ws?token={TOKEN}"
    with contextlib.closing(connect(address, proxy=None, open_timeout=2, close_timeout=1)) as connection:
        connection.send("connected")
        assert connection.recv(timeout=2) == "connected"
        port = paired_viewer.port
        paired_viewer.stop()
        with pytest.raises(ConnectionClosed):
            connection.recv(timeout=2)
        assert echo_backend[2].wait(timeout=2)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def test_bind_failure_is_reported_and_does_not_leave_a_worker(tmp_path):
    index = tmp_path / "index.html"
    index.write_text("viewer")
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        viewer = ViewerServer(12345, occupied.getsockname()[1], index)
        with pytest.raises(OSError):
            viewer.start()
        viewer.stop()
        assert viewer._thread is None


def test_lan_binding_cannot_skip_the_backend_guard():
    with pytest.raises(ValueError, match="guarded backend"):
        ViewerServer(12345, 0, host="0.0.0.0", token=TOKEN)


@pytest.fixture
def real_viser():
    import viser

    from viewer import guard_backend

    server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
    key = guard_backend(server)
    server.gui.add_text("Guard check", initial_value="application data")
    try:
        yield server, key
    finally:
        server.stop()


@pytest.mark.parametrize("headers", [None, {"X-Teleoperator-Gateway": "wrong"}])
def test_direct_viser_connection_is_closed_before_application_bytes(real_viser, headers):
    server, _ = real_viser
    with connect(
        f"ws://127.0.0.1:{server.get_port()}",
        subprotocols=["viser-v1.1.1"],
        additional_headers=headers,
        proxy=None,
        open_timeout=2,
        close_timeout=1,
    ) as connection:
        with pytest.raises(ConnectionClosed) as error:
            connection.recv(timeout=2)
        assert error.value.rcvd.code == 1008
        assert server.get_clients() == {}


def test_authenticated_gateway_receives_real_viser_state_without_exposing_backend_key(tmp_path, real_viser):
    server, key = real_viser
    index = tmp_path / "index.html"
    index.write_text("viewer")
    viewer = ViewerServer(server.get_port(), 0, index, token=TOKEN, backend_key=key)
    viewer.start()
    try:
        with connect(
            f"ws://127.0.0.1:{viewer.port}/ws?token={TOKEN}",
            subprotocols=["viser-v1.1.1"],
            proxy=None,
            open_timeout=2,
            close_timeout=1,
        ) as connection:
            payload = connection.recv(timeout=2)
            assert isinstance(payload, bytes)
            assert payload
            assert key.encode() not in payload
            assert connection.subprotocol == "viser-v1.1.1"
    finally:
        viewer.stop()
