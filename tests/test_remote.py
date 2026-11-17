"""The same perception code must work against a browser we did not launch (a remote/hosted Chromium over CDP)."""
import socket
import subprocess
import time
import urllib.request

import pytest
from playwright.sync_api import sync_playwright

from test_pilot import BrowserConfig, BrowserSession


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def test_observe_through_remote_cdp_endpoint(base_url, tmp_path):
    with sync_playwright() as pw:
        exe = pw.chromium.executable_path
    port = _free_port()
    proc = subprocess.Popen(
        [exe, "--headless", "--no-sandbox", f"--remote-debugging-port={port}", f"--user-data-dir={tmp_path}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )   # a separate process stands in for the remote browser
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=0.5)
                break
            except Exception:
                time.sleep(0.1)
        else:
            pytest.skip("could not start a standalone Chromium")
        with BrowserSession(BrowserConfig(cdp_url=f"http://127.0.0.1:{port}")) as s:
            s.goto(base_url + "/")
            text = s.look()
            assert 'textbox "From"' in text and "Booking confirmed" not in text
    finally:
        proc.terminate()
