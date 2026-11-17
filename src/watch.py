"""Knowing when a page has stopped changing, and whether an action changed it.

No fixed sleeps. A MutationObserver counts real DOM changes inside the page; Playwright events track requests in
flight. settle() returns when the document is parsed, the DOM has been quiet for `quiet_ms`, and no recent request is
pending, bounded by `cap_ms`. Idea from Browser Use's ax.py, reimplemented on Playwright.
"""
from __future__ import annotations

import re
import time
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page

_CONTEXT_LOSS = re.compile(
    r"Execution context was destroyed|Cannot find context|Target closed|Target page, context or browser has been closed|"
    r"navigat|No frame",
    re.I,
)
IGNORED_RESOURCES = {"websocket", "eventsource", "media", "manifest"}
LONG_POLL_MS = 1500  # requests older than this are treated as long-poll / analytics, not "loading"

# Installed in every document before page scripts run. Style-only churn counts once per element per 200 ms, so an
# animation loop goes quiet while a menu shown via style.display still counts as a change.
INIT_JS = r"""
(() => {
  if (window.__qa) return;
  const st = window.__qa = {n: 0, last: performance.now(), doc: Math.random()};
  const styled = new WeakMap();
  new MutationObserver((records) => {
    const t = performance.now();
    let real = false;
    for (const r of records) {
      if (r.attributeName && r.oldValue === r.target.getAttributeNS(r.attributeNamespace, r.attributeName)) continue;
      if (r.attributeName !== 'style') real = true;
      else if (t - (styled.get(r.target) ?? -1e9) > 200) real = true;
      if (r.attributeName === 'style') styled.set(r.target, t);
    }
    if (real) { st.last = t; st.n++; }
  }).observe(document, {subtree: true, childList: true, attributes: true, attributeOldValue: true, characterData: true});
  for (const type of ['input', 'change']) addEventListener(type, () => { st.n++; st.last = performance.now(); }, true);
})();
"""

PROBE_JS = r"""() => {
  const s = window.__qa;
  return {
    idle: s ? performance.now() - s.last : 1e9,
    n: s ? s.n : 0,
    doc: s ? s.doc : null,
    ready: document.readyState,
    url: location.href,
    title: document.title,
  };
}"""


def now_ms() -> float:
    return time.monotonic() * 1000


def is_context_loss(e: BaseException) -> bool:
    return bool(_CONTEXT_LOSS.search(str(e)))


class PageWatcher:
    def __init__(self, page: Page):
        self.page = page
        self.inflight: dict[Any, float] = {}
        self.last_net = 0.0
        self.nav_failure: str | None = None
        page.add_init_script(INIT_JS)  # future documents
        try:
            page.evaluate(INIT_JS)  # the document that is already open (remote browsers)
        except PlaywrightError:
            pass
        page.on("request", self._on_request)
        page.on("requestfinished", self._on_done)
        page.on("requestfailed", self._on_failed)

    def _on_request(self, req) -> None:
        if req.resource_type == "document" and req.frame == self.page.main_frame:
            self.nav_failure = None
        if req.resource_type not in IGNORED_RESOURCES:
            self.inflight[req] = now_ms()
            self.last_net = now_ms()

    def _on_done(self, req) -> None:
        self.inflight.pop(req, None)
        self.last_net = now_ms()

    def _on_failed(self, req) -> None:
        self.inflight.pop(req, None)
        self.last_net = now_ms()
        if req.resource_type == "document" and req.frame == self.page.main_frame:
            self.nav_failure = req.failure or "request failed"

    def pending(self) -> int:
        t = now_ms()
        return sum(t - started < LONG_POLL_MS for started in self.inflight.values())

    def probe(self) -> dict:
        return self.page.evaluate(PROBE_JS)

    def settle(self, cap_ms: float = 3000, quiet_ms: float = 150, since: float | None = None) -> dict:
        """`since` (a now_ms() timestamp taken when an action began) makes quiet count only from that moment."""
        start = now_ms()
        limit = cap_ms
        probe = None
        while now_ms() - start < limit:
            try:
                probe = self.probe()
            except PlaywrightError as e:
                if not is_context_loss(e):
                    return {"why": "error", "pending": self.pending(), "ms": now_ms() - start, "probe": None}
                probe = None
                self.page.wait_for_timeout(40)  # navigation in progress: wait for the new document
                continue
            if probe["ready"] == "loading":
                limit = max(cap_ms, 5000)  # navigations need longer than in-page updates
            idle = probe["idle"] if since is None else min(probe["idle"], now_ms() - since)
            net_quiet = now_ms() - self.last_net >= quiet_ms
            if probe["ready"] != "loading" and idle >= quiet_ms and self.pending() == 0 and net_quiet:
                return {"why": "quiet", "pending": 0, "ms": now_ms() - start, "probe": probe}
            self.page.wait_for_timeout(30)  # also lets Playwright deliver request events
        return {"why": "cap", "pending": self.pending(), "ms": now_ms() - start, "probe": probe}


def describe_change(before: dict | None, after: dict | None) -> str:
    """What the next action tells the model: navigated / page changed / no change."""
    if not before or not after:
        return "page changed"
    if after["doc"] != before["doc"] or after["url"] != before["url"]:
        return f"navigated to {after['url']}"
    return "page changed" if after["n"] != before["n"] else "no change"
