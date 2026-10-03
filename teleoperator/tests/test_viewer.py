"""The viewer serves only its bundle and always selects the local backend."""

import http.client
import socket
from urllib.parse import parse_qs, urlsplit

import pytest

from viewer import ViewerServer


@pytest.fixture
def viewer(tmp_path):
    index = tmp_path / "index.html"
    index.write_text("<!doctype html><title>Teleoperator</title>")
    server = ViewerServer(websocket_port=12345, port=0, index_path=index)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def request(viewer, path, method="GET"):
    connection = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=2)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def canonical_path(viewer):
    return request(viewer, "/")[1]["Location"]


def test_root_redirect_selects_only_the_local_backend_and_dark_theme(viewer):
    status, headers, body = request(viewer, "/")
    assert status == 302
    assert body == b""
    query = parse_qs(urlsplit(headers["Location"]).query, keep_blank_values=True)
    assert query == {"websocket": [f"ws://127.0.0.1:{viewer.port}/ws"], "darkMode": [""], "hideViserLogo": [""]}
    assert headers["Cache-Control"] == "no-store"
    assert viewer.url == f"http://127.0.0.1:{viewer.port}/"


@pytest.mark.parametrize(
    "path",
    [
        "/?websocket=ws://attacker.invalid:80",
        "/?websocket=ws://127.0.0.1:12345&websocket=ws://attacker.invalid:80",
        "/index.html",
    ],
)
def test_noncanonical_query_redirects_to_the_authoritative_backend(viewer, path):
    status, headers, _ = request(viewer, path)
    assert status == 302
    assert headers["Location"] == canonical_path(viewer)


def test_bundle_has_html_type_exact_length_and_no_cache(viewer):
    status, headers, body = request(viewer, canonical_path(viewer))
    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Content-Length"] == str(len(body))
    assert headers["Cache-Control"] == "no-store"
    assert body == viewer.index_path.read_bytes()


def test_head_matches_get_headers_without_sending_bundle(viewer):
    path = canonical_path(viewer)
    status, headers, body = request(viewer, path, "HEAD")
    assert status == 200
    assert body == b""
    assert int(headers["Content-Length"]) == viewer.index_path.stat().st_size
    status, headers, body = request(viewer, "/", "HEAD")
    assert status == 302
    assert headers["Location"] == path
    assert body == b""


@pytest.mark.parametrize("path", ["/favicon.ico", "/../index.html", "/%2e%2e/index.html", "/other.html"])
def test_unrecognized_paths_are_not_served(viewer, path):
    assert request(viewer, path)[0] == 404


def test_start_and_stop_are_idempotent_and_release_the_bound_port(tmp_path):
    index = tmp_path / "index.html"
    index.write_text("viewer")
    server = ViewerServer(12345, 0, index)
    server.stop()
    server.start()
    original_port = server.port
    server.start()
    assert server.port == original_port
    server.stop()
    server.stop()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", original_port))


def test_missing_bundle_fails_without_binding_a_port(tmp_path):
    server = ViewerServer(12345, 0, tmp_path / "missing.html")
    with pytest.raises(RuntimeError, match="Viewer bundle is missing"):
        server.start()
    assert server.port == 0
    server.stop()
