"""Exercise the native connection model against an isolated local HTTP fixture."""

import contextlib
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


class Handler(BaseHTTPRequestHandler):
    leaked = False

    def do_GET(self):
        request = urlsplit(self.path)
        token = parse_qs(request.query).get("token", [""])[0]
        if request.path != "/health":
            Handler.leaked = True
            self.send_error(500)
            return
        if token.startswith("redirect"):
            self.send_response(302)
            self.send_header("Location", "/leak?token=" + token)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        time.sleep(0.3 if token.startswith("slow") else 0.05)
        body = json.dumps({"name": "Teleoperator", "version": 1}).encode()
        self.send_response(401 if token.startswith("bad") else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def log_message(self, format, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    subprocess.run([sys.argv[1], str(server.server_port)], check=True)
    if Handler.leaked:
        raise AssertionError("The native health request followed a redirect")
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)
