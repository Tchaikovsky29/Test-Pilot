"""Logged-in sessions without credentials. The user logs in themselves in a browser the tool opens (including any
email or authenticator code); only the resulting browser state (cookies and local storage, Playwright's
storage_state) is kept, in a local file per site. The tool never sees or stores a password.

    test-pilot login https://example.my.salesforce.com/      ->  .test-pilot/sessions/example.my.salesforce.com.json

explore, verify and observe then load the session for the URL's host on their own (or the file given by --session).
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

SESSIONS_DIR = Path(".test-pilot") / "sessions"


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def session_path(url: str, root: Path = SESSIONS_DIR) -> Path:
    return root / f"{host_of(url)}.json"


def find_session(url: str, explicit: str | None = None, root: Path = SESSIONS_DIR) -> str | None:
    """The session file to use: the one given, else the saved one for the URL's host, else none."""
    if explicit:
        return explicit
    p = session_path(url, root)
    return str(p) if p.is_file() else None
