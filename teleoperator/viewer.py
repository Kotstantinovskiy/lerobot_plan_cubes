"""Serve the viewer and proxy its authenticated WebSocket to the local robot."""

import asyncio
from importlib import metadata
import ipaddress
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import parse_qs, urlencode, urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, ClientWSTimeout, WSMsgType, web

_PRIVATE_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
    "127.0.0.0/8", "fc00::/7", "fe80::/10", "::1/128",
))
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"}
_CONNECTIONS = web.AppKey("connections", set)
_BACKEND_HEADER = "X-SO101-Gateway"
_CAMERA_NAMES = ("wrist", "top")
_CAMERA_PATHS = {f"/cameras/{name}/frame": name for name in _CAMERA_NAMES}


def _local_host(host):
    if not isinstance(host, str) or not host or len(host) > 253:
        return False
    host = host.lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or (
            host.endswith(".local") and all(_LABEL.fullmatch(label) for label in host.split("."))
        )
    return any(address.version == network.version and address in network for network in _PRIVATE_NETWORKS)


def _authority(host, port):
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def guard_backend(server):
    """Require a private gateway key before Viser sends or consumes app data.

    Viser 1.1.1 has no public handshake-authentication hook. This isolated adapter
    reads its raw connection map in an awaited infra connect callback, which runs
    before the message producers and consumer. Call once, immediately after
    constructing ViserServer and before adding GUI controls. No package is patched.
    """
    if metadata.version("viser") != "1.1.1":
        raise RuntimeError("Review the backend guard before changing Viser 1.1.1")
    try:
        transport = server._websock_server
        connections = transport._live_connections
    except AttributeError as error:
        raise RuntimeError("Viser backend guard is unavailable") from error
    if not isinstance(connections, dict):
        raise RuntimeError("Viser backend connection map changed")
    key = secrets.token_urlsafe(32)

    @transport.on_client_connect
    async def authenticate(client):
        connection = connections.get(client.client_id)
        if connection is None:
            return
        values = connection.request.headers.get_all(_BACKEND_HEADER) if connection.request is not None else []
        if len(values) != 1 or not values[0].isascii() or not secrets.compare_digest(values[0], key):
            await connection.close(code=1008, reason="Use the paired viewer")

    transport.disconnect_all_clients()
    return key


class ViewerServer:
    def __init__(self, websocket_port, port, index_path=None, *, host="127.0.0.1", token=None, advertised_host=None, backend_key=None, cameras=None):
        if isinstance(websocket_port, bool) or not isinstance(websocket_port, int) or not 1 <= websocket_port <= 65535:
            raise ValueError("WebSocket port must be between 1 and 65535")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("Viewer port must be between 0 and 65535")
        if host not in {"127.0.0.1", "0.0.0.0"}:
            raise ValueError("Viewer host must be 127.0.0.1 or 0.0.0.0")
        if token is not None and (not isinstance(token, str) or re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token) is None):
            raise ValueError("Pairing token must contain 32 to 256 URL-safe characters")
        if host != "127.0.0.1" and token is None:
            raise ValueError("LAN mode requires a pairing token")
        if backend_key is not None and (not isinstance(backend_key, str) or re.fullmatch(r"[A-Za-z0-9_-]{32,256}", backend_key) is None):
            raise ValueError("Backend key must contain 32 to 256 URL-safe characters")
        if host != "127.0.0.1" and backend_key is None:
            raise ValueError("LAN mode requires a guarded backend")
        if advertised_host is not None and not _local_host(advertised_host):
            raise ValueError("Advertised host must be a private IP address or a .local name")
        self.websocket_port = websocket_port
        self.port = port
        self.host = host
        self.token = token
        self.backend_key = backend_key
        self.cameras = cameras
        self.advertised_host = advertised_host or "127.0.0.1"
        self.index_path = Path(index_path) if index_path is not None else Path(__file__).parent / "web" / "dist" / "index.html"
        self._server = None
        self._thread = None
        self._loop = None
        self._stop_event = None
        self._error = None

    def _url_for(self, host):
        query = "?" + urlencode({"token": self.token}) if self.token is not None else ""
        return f"http://{_authority(host, self.port)}/{query}"

    @property
    def url(self):
        return self._url_for("127.0.0.1")

    @property
    def pairing_url(self):
        return self._url_for(self.advertised_host)

    def _request_authority(self, request):
        values = request.headers.getall("Host", [])
        if len(values) != 1 or any(character in values[0] for character in "/\\?#@%"):
            return None
        try:
            parsed = urlsplit("//" + values[0])
            port = parsed.port if parsed.port is not None else 80
            if port != self.port or not _local_host(parsed.hostname):
                return None
        except ValueError:
            return None
        return _authority(parsed.hostname.lower(), self.port)

    @staticmethod
    def _response(status, text):
        return web.Response(status=status, text=text, headers=_HEADERS)

    @staticmethod
    def _same_origin(origin, authority):
        try:
            parsed = urlsplit(origin)
            port = parsed.port if parsed.port is not None else 80
            return (
                parsed.scheme.lower() == "http" and parsed.username is None and parsed.password is None
                and parsed.path == "" and not parsed.query and not parsed.fragment
                and parsed.hostname is not None and _authority(parsed.hostname.lower(), port) == authority
            )
        except ValueError:
            return False

    def _camera_status(self):
        """Camera failures must not interrupt the viewer or robot connection."""
        try:
            states = self.cameras.status() if self.cameras is not None else {}
            connected = {name: states.get(name, {}).get("connected") is True for name in _CAMERA_NAMES}
        except Exception:
            connected = dict.fromkeys(_CAMERA_NAMES, False)
        return web.json_response({name: {"connected": value} for name, value in connected.items()}, headers=_HEADERS)

    def _camera_frame(self, name):
        try:
            frame = self.cameras.frame(name) if self.cameras is not None else None
        except Exception:
            frame = None
        if not isinstance(frame, bytes) or not frame:
            return self._response(503, "Camera is not connected")
        return web.Response(body=frame, content_type="image/jpeg", headers=_HEADERS)

    async def _relay(self, request, session):
        protocols = tuple(
            part.strip() for header in request.headers.getall("Sec-WebSocket-Protocol", [])
            for part in header.split(",") if part.strip()
        )
        if len(protocols) > 8 or any(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value) is None for value in protocols):
            return self._response(400, "Invalid WebSocket protocol")
        client = web.WebSocketResponse(protocols=protocols, heartbeat=5, timeout=0.5, max_msg_size=64 * 1024 * 1024, compress=False)
        if request.method != "GET" or not client.can_prepare(request).ok:
            return self._response(400, "WebSocket upgrade required")
        try:
            upstream = await session.ws_connect(
                f"ws://127.0.0.1:{self.websocket_port}",
                max_msg_size=64 * 1024 * 1024, compress=0, heartbeat=5,
                timeout=ClientWSTimeout(ws_close=0.5),
                protocols=protocols,
                headers={_BACKEND_HEADER: self.backend_key} if self.backend_key is not None else None,
            )
        except (ClientError, OSError, asyncio.TimeoutError):
            return self._response(502, "Robot viewer is unavailable")
        client = web.WebSocketResponse(
            protocols=(upstream.protocol,) if upstream.protocol is not None else (),
            heartbeat=5, timeout=0.5, max_msg_size=64 * 1024 * 1024, compress=False,
        )

        async def copy_messages(source, destination):
            async for message in source:
                if message.type == WSMsgType.TEXT:
                    await destination.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await destination.send_bytes(message.data)
                elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                    break

        tasks = []
        try:
            await client.prepare(request)
            request.app[_CONNECTIONS].add(client)
            tasks = [asyncio.create_task(copy_messages(client, upstream)), asyncio.create_task(copy_messages(upstream, client))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await upstream.close()
            await client.close()
            request.app[_CONNECTIONS].discard(client)
        return client

    async def _serve(self, content, ready):
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        connections = set()
        async with ClientSession(trust_env=False, timeout=ClientTimeout(total=3, connect=1)) as session:
            async def handle(request):
                if request.path not in {"/", "/index.html", "/health", "/ws", "/cameras/status", *_CAMERA_PATHS}:
                    return self._response(404, "Not found")
                authority = self._request_authority(request)
                if authority is None:
                    return self._response(400, "Invalid local host")
                try:
                    query = parse_qs(request.query_string, keep_blank_values=True, max_num_fields=16)
                except ValueError:
                    return self._response(400, "Invalid query")
                if self.token is not None:
                    values = query.get("token", [])
                    if len(values) != 1 or not values[0].isascii() or not secrets.compare_digest(values[0], self.token):
                        return self._response(401, "Pairing required")
                origins = request.headers.getall("Origin", [])
                if origins and (len(origins) != 1 or not self._same_origin(origins[0], authority)):
                    return self._response(403, "Origin not allowed")
                if request.path == "/health":
                    return web.json_response({"name": "SO-101", "version": 1}, headers=_HEADERS)
                if request.path == "/cameras/status":
                    return self._camera_status()
                if request.path in _CAMERA_PATHS:
                    return self._camera_frame(_CAMERA_PATHS[request.path])
                if request.path == "/ws":
                    return await self._relay(request, session)
                websocket = f"ws://{authority}/ws"
                expected = {}
                if self.token is not None:
                    expected["token"] = [self.token]
                    websocket += "?" + urlencode({"token": self.token})
                expected.update(websocket=[websocket], darkMode=[""], hideViserLogo=[""])
                if query != expected:
                    return web.Response(status=302, body=b"", headers={
                        **_HEADERS, "Location": "/?" + urlencode(expected, doseq=True),
                    })
                return web.Response(body=content, content_type="text/html", charset="utf-8", headers=_HEADERS)

            async def shutdown(app):
                await asyncio.gather(*(connection.close(code=1001) for connection in tuple(connections)))

            app = web.Application()
            app[_CONNECTIONS] = connections
            app.router.add_get("/{path:.*}", handle)
            app.on_shutdown.append(shutdown)
            runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
            try:
                await runner.setup()
                await web.TCPSite(runner, self.host, self.port).start()
                self.port = runner.addresses[0][1]
                self._server = runner
                ready.set()
                await self._stop_event.wait()
            finally:
                await runner.cleanup()
                self._server = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.index_path.is_file():
            raise RuntimeError(f"Viewer bundle is missing: {self.index_path}. Build the viewer before starting.")
        content = self.index_path.read_bytes()
        ready = threading.Event()
        self._error = None

        def run():
            try:
                asyncio.run(self._serve(content, ready))
            except BaseException as error:
                self._error = error
            finally:
                ready.set()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        if not ready.wait(timeout=5):
            self.stop()
            raise RuntimeError("Viewer did not start")
        if self._error is not None:
            error = self._error
            self.stop()
            raise error

    def stop(self):
        if self._thread is None:
            return
        if self._thread.is_alive() and self._loop is not None and self._stop_event is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except RuntimeError:
                # A failed startup may have closed its loop just before the
                # worker finishes reporting the original bind error.
                if not self._loop.is_closed():
                    raise
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("Viewer did not stop")
        self._thread = None
        self._loop = None
        self._stop_event = None
