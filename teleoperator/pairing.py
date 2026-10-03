"""Local-network pairing address and an in-memory QR code."""

import base64
from html import escape
import io
import ipaddress
import re
import socket
import subprocess
import sys

import qrcode

PRIVATE_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "127.0.0.0/8",
))


def _hostname():
    if sys.platform == "darwin":
        try:
            name = subprocess.check_output(
                ["/usr/sbin/scutil", "--get", "LocalHostName"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip()
            if name:
                return name + ".local"
        except (OSError, subprocess.SubprocessError):
            pass
    return socket.gethostname()


def local_address(value=None):
    """Use the Mac's Bonjour hostname unless a LAN address was supplied."""
    host = value if value is not None else _hostname()
    host = host.strip().lower().rstrip(".")
    if value is None and not host.endswith(".local"):
        host += ".local"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if not host.endswith(".local") or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host):
            raise ValueError("Use a private IPv4 address or the Mac's .local hostname")
        if any(not part or len(part) > 63 or part.startswith("-") or part.endswith("-") for part in host.split(".")):
            raise ValueError("Invalid .local hostname")
        return host
    if address.version != 4 or not any(address in network for network in PRIVATE_NETWORKS):
        raise ValueError("Use a private IPv4 address or the Mac's .local hostname")
    return str(address)


def pairing_html(url):
    """Render a pairing secret without writing it or its QR code to disk."""
    buffer = io.BytesIO()
    image = qrcode.make(url)
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<div class="phone-pairing">'
        f'<img src="data:image/png;base64,{encoded}" width="240" height="240" alt="QR-код подключения iPhone">'
        '<p>iPhone и Mac должны быть в одной Wi-Fi сети. Откройте сканер в приложении.</p>'
        f'<p class="phone-pairing__url">{escape(url)}</p>'
        '<p>Ссылка даёт доступ к пульту и действует до перезапуска сервера.</p></div>'
    )
