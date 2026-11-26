"""Browser factory and session. Local Chromium by default; a remote browser when a CDP endpoint is configured."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

from .models import Observation
from .perceive import Perceiver
from .render import DEFAULT_MAX_CHARS, render
from .text import clip, quote
from .watch import PageWatcher, describe_change


class NavigationError(Exception):
    """The URL is invalid, unreachable, or answered with an error status. The message is safe to show the user."""


def validate_url(url: str) -> str:
    u = url.strip()
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.netloc:
        raise NavigationError(f"Not a valid web address: {url!r}. Use a full http:// or https:// URL.")
    return u


@dataclass
class BrowserConfig:
    headless: bool = True
    cdp_url: str | None = field(default_factory=lambda: os.getenv("TEST_PILOT_CDP_URL") or None)  # remote browser
    viewport: tuple[int, int] = (1280, 800)
    nav_timeout_ms: int = 30_000
    dialog_policy: str = "accept"
    maximize: bool = True
    fresh_context: bool = False  # on a remote browser, open a context of our own instead of reusing the default one


class BrowserSession:
    def __init__(self, config: BrowserConfig | None = None):
        self.config = config or BrowserConfig()
        self.dialogs: list[str] = []
        self.page: Page | None = None

    def __enter__(self) -> "BrowserSession":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> "BrowserSession":
        c = self.config
        self._pw = sync_playwright().start()
        if c.cdp_url:
            self._browser = self._pw.chromium.connect_over_cdp(c.cdp_url)
            # Hosted browsers usually expose a single default context: use it rather than creating another.
            if c.fresh_context or not self._browser.contexts:  # parallel workers must not share cookies or tabs
                self._context = self._browser.new_context()
            else:
                self._context = self._browser.contexts[0]
            self.page = self._context.pages[0] if self._context.pages else self._context.new_page()
        else:
            args = ["--start-maximized"] if c.maximize and not c.headless else []
            self._browser = self._pw.chromium.launch(headless=c.headless, args=args)
            self._context = self._browser.new_context(no_viewport=True)
            self.page = self._context.new_page()
        self.page.set_default_timeout(c.nav_timeout_ms)
        if c.maximize and not c.headless:
            try:
                size = self.page.evaluate("() => [screen.availWidth || innerWidth, screen.availHeight || innerHeight]")
                if isinstance(size, list) and len(size) == 2:
                    width, height = int(size[0]), int(size[1])
                    if width > 0 and height > 0:
                        self.page.set_viewport_size({"width": width, "height": height})
                        c.viewport = (width, height)
            except Exception:
                pass
        self.page.on("dialog", self._on_dialog)
        self.watcher = PageWatcher(self.page)
        self.cdp = self._context.new_cdp_session(self.page)
        self.perceiver = Perceiver(self.page, self.cdp, self.watcher, self.dialogs)
        return self

    def _on_dialog(self, dialog) -> None:
        # An open alert/confirm blocks the page, so it is handled at once and reported in the next observation.
        policy = "accepted" if self.config.dialog_policy == "accept" else "dismissed"
        self.dialogs.append(f"[dialog {dialog.type}] {quote(clip(dialog.message, 300))} ({policy})")
        try:
            dialog.accept() if policy == "accepted" else dialog.dismiss()
        except PlaywrightError:
            pass

    def close(self) -> None:
        for step in ("_context", "_browser"):
            try:
                getattr(self, step).close()
            except Exception:
                pass
        try:
            self._pw.stop()
        except Exception:
            pass

    def goto(self, url: str) -> int | None:
        url = validate_url(url)
        try:
            resp = self.page.goto(url, wait_until="domcontentloaded", timeout=self.config.nav_timeout_ms)
        except PlaywrightError as e:
            code = re.search(r"net::ERR_[A-Z_]+", str(e))
            reason = code.group() if code else ("timed out" if "Timeout" in str(e) else clip(str(e).splitlines()[0], 160))
            raise NavigationError(f"Could not open {url}: {reason}") from None
        self.watcher.settle()
        status = resp.status if resp else None
        if status is not None and status >= 400:
            raise NavigationError(f"{url} answered with HTTP {status}")
        return status

    def observe(self) -> Observation:
        return self.perceiver.observe()

    def look(self, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """The page as the model sees it."""
        return render(self.observe(), max_chars)

    def snapshot(self) -> dict | None:
        try:
            return self.watcher.probe()
        except PlaywrightError:
            return None

    @staticmethod
    def change(before: dict | None, after: dict | None) -> str:
        return describe_change(before, after)
