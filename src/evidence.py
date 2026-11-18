"""What changed between two looks. This is the raw material for assertions: after "Search" with an empty field, a test
needs to say "an error appeared", and after a valid search, "results appeared". Pure functions, no browser."""
from __future__ import annotations

from urllib.parse import urldefrag

from pydantic import BaseModel, Field

from .models import Observation
from .text import clip, fold

MAX_LINES = 8
MAX_LEN = 140


class Changes(BaseModel):
    navigated: bool = False
    url: str = ""
    appeared: list[str] = Field(default_factory=list)  # headings and text that are new
    disappeared: list[str] = Field(default_factory=list)  # headings and text that are gone
    alerts: list[str] = Field(default_factory=list)  # live-region messages (role=alert/status) that are new
    errors: list[str] = Field(default_factory=list)  # "Field: message" for fields newly in error

    @property
    def empty(self) -> bool:
        return not (self.navigated or self.appeared or self.disappeared or self.alerts or self.errors)


def _texts(obs: Observation, kinds: tuple[str, ...]) -> dict[str, str]:
    return {fold(ln.text): ln.text for ln in obs.lines if ln.kind in kinds}


def _errors(obs: Observation) -> dict[int, str]:
    out: dict[int, str] = {}
    for e in obs.elements:
        msg = e.validation or (e.description if "invalid" in e.states else "")
        if msg:
            out[e.id] = f"{e.display_name or e.widget}: {msg}"
    return out


def diff(before: Observation | None, after: Observation) -> Changes:
    if before is None:
        return Changes()
    ch = Changes()
    ch.navigated = urldefrag(before.url).url != urldefrag(after.url).url
    ch.url = after.url if ch.navigated else ""
    b, a = _texts(before, ("heading", "text")), _texts(after, ("heading", "text"))
    heads_after = {fold(ln.text) for ln in after.lines if ln.kind == "heading"}
    new = [t for k, t in a.items() if k not in b]
    if ch.navigated:  # a whole new page: its headings say enough
        new = [t for t in new if fold(t) in heads_after]
    ch.appeared = [clip(t, MAX_LEN) for t in new]
    ch.disappeared = [] if ch.navigated else [clip(t, MAX_LEN) for k, t in b.items() if k not in a]
    ba, aa = _texts(before, ("alert",)), _texts(after, ("alert",))
    ch.alerts = [clip(t, MAX_LEN) for k, t in aa.items() if k not in ba]
    eb, ea = _errors(before), _errors(after)
    ch.errors = [clip(m, MAX_LEN) for i, m in ea.items() if eb.get(i) != m]
    return ch


def render_changes(ch: Changes) -> str:
    """The block appended after a look so the model (and later the test designer) sees what an action did."""
    if ch.empty:
        return ""
    out = ["[changes since last look]"]
    if ch.navigated:
        out.append(f"navigated to {ch.url}")

    def block(prefix: str, items: list[str]) -> None:
        for t in items[:MAX_LINES]:
            out.append(f"{prefix} {t}")
        if len(items) > MAX_LINES:
            out.append(f"{prefix} (… {len(items) - MAX_LINES} more)")

    block("+", ch.appeared)
    block("-", ch.disappeared)
    block("!", ch.alerts)
    block("error:", ch.errors)
    out.append("[/changes]")
    return "\n" + "\n".join(out)