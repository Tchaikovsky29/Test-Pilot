"""A person logs in once; the tool keeps only the browser session and notices when it no longer works."""
import json

import pytest

from test_pilot import BrowserConfig, BrowserSession
from test_pilot.browser import SessionExpired
from test_pilot.policy import Policy
from test_pilot.sessions import find_session, session_path


@pytest.fixture(scope="module", autouse=True)
def _browser_available():
    try:
        BrowserSession(BrowserConfig()).start().close()        # closed at once: one Playwright per thread
    except Exception as e:
        pytest.skip(f"browser unavailable: {e}")


def test_a_saved_session_logs_in_again_and_holds_no_password(base_url, tmp_path):
    path = str(tmp_path / "org.json")
    with BrowserSession(BrowserConfig()) as s:                  # the person logs in
        s.goto(base_url + "/login.html")
        assert s.login_form_shown()
        s.page.fill("#u", "rep@example.com")
        s.page.fill("#p", "not-saved-anywhere")
        s.page.click("#in")
        s.page.wait_for_selector("text=Welcome back")
        s.save_state(path)
    saved = json.loads(open(path).read())
    assert any(c["name"] == "sid" for c in saved["cookies"]) and "not-saved-anywhere" not in open(path).read()
    with BrowserSession(BrowserConfig(storage_state=path)) as s:  # the tool, later
        s.goto(base_url + "/login.html")
        assert not s.login_form_shown() and s.page.is_visible("text=Welcome back")


def test_an_expired_session_stops_at_the_login_wall(base_url, tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"cookies": [], "origins": []}))
    with BrowserSession(BrowserConfig(storage_state=str(path))) as s:
        with pytest.raises(SessionExpired, match="test-pilot login"):
            s.goto(base_url + "/login.html")


def test_without_a_session_a_login_page_is_just_a_page(base_url):
    with BrowserSession(BrowserConfig()) as s:
        s.goto(base_url + "/login.html")                        # no error: testing a login form is allowed
        assert s.login_form_shown()


def test_sessions_are_found_by_host(tmp_path):
    url = "https://www.Example.my.salesforce.com/lightning/page/home"
    assert session_path(url, tmp_path).name == "example.my.salesforce.com.json"
    assert find_session(url, root=tmp_path) is None
    session_path(url, tmp_path).write_text("{}")
    assert find_session(url, root=tmp_path) == str(session_path(url, tmp_path))
    assert find_session(url, explicit="mine.json", root=tmp_path) == "mine.json"


def test_the_host_a_start_url_redirects_to_is_part_of_the_site():
    p = Policy(start_url="https://org.my.salesforce.com/")
    assert not p.host_allowed("https://org.lightning.force.com/lightning/o/Lead/list")
    p.allow_site_of("https://org.lightning.force.com/lightning/page/home")
    assert p.host_allowed("https://org.lightning.force.com/lightning/o/Lead/list")
    p.allow_site_of("chrome-error://chromewebdata/")
    assert "chromewebdata" not in p.allowed_hosts
