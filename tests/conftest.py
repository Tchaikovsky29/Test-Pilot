import http.server
import threading
import time
from pathlib import Path

import pytest

from test_pilot import BrowserConfig, BrowserSession

FIXTURES = Path(__file__).parent / "fixtures"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(FIXTURES), **kw)

    def do_GET(self):
        if self.path.startswith("/slow"):
            time.sleep(0.4)  # slow API call: perception must wait for it
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        if self.path in ("/flights", "/"):
            self.path = "/flight.html"
        elif self.path.startswith("/hotels"):
            self.path = "/hotels.html"
        super().do_GET()

    def log_message(self, *a):
        pass


@pytest.fixture(scope="session")
def base_url():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture()
def session():
    try:
        s = BrowserSession(BrowserConfig()).start()
    except Exception as e:  # no Chromium installed
        pytest.skip(f"browser unavailable: {e}")
    yield s
    s.close()
