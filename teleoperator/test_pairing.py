"""Only addresses reachable on the intended local network are advertised."""

import pytest

from pairing import local_address, pairing_html


@pytest.mark.parametrize("value", ["192.168.1.10", "10.0.0.2", "172.16.5.1", "169.254.2.3", "127.0.0.1", "robot-mac.local"])
def test_local_pairing_addresses(value):
    assert local_address(value) == value


@pytest.mark.parametrize("value", ["8.8.8.8", "0.0.0.0", "192.0.2.1", "example.com", "bad..local", "-bad.local", "host/path.local", "host.local:8080"])
def test_public_or_malformed_addresses_are_not_advertised(value):
    with pytest.raises(ValueError):
        local_address(value)


def test_default_uses_the_mac_bonjour_name(monkeypatch):
    monkeypatch.setattr("pairing._hostname", lambda: "Robot-Mac")
    assert local_address() == "robot-mac.local"


def test_qr_is_self_contained_and_pairing_link_is_escaped():
    page = pairing_html("http://robot.local:8080/?token=one&other=two")
    assert 'src="data:image/png;base64,iVBOR' in page
    assert "token=one&amp;other=two" in page
