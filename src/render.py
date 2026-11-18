"""Observation -> the text the model reads. One line per control, in page order:

    [id] widget "name" type=email = "value" required invalid  options: a | b | c
A leading * marks controls that were not in the previous look. Cut-down pages keep the top and the last fifth,
because a dialog's Submit button is usually its last line (a trick from Browser Use's ax.py).
"""
from __future__ import annotations

from .models import Element, Observation
from .text import clip, quote, well_formed

DEFAULT_MAX_CHARS = 4000


def element_line(e: Element) -> str:
    parts = [("*" if e.is_new else "") + f"[{e.id}] {e.widget}"]
    if e.name:
        parts.append(f'"{clip(e.display_name, 150)}"')
    if e.input_type and e.widget in ("textbox", "searchbox") and e.input_type not in ("text", ""):
        parts.append(f"type={e.input_type}")
    if e.value:
        parts.append("= " + quote(clip(e.value, 100)))
    parts.extend(e.states)
    if e.constraints:
        parts.append("[" + " ".join(f"{k}={clip(v, 30)}" for k, v in e.constraints.items()) + "]")
    if e.href:
        parts.append("-> " + clip(e.href, 70))
    if e.validation:
        parts.append("(error: " + quote(clip(e.validation, 120)) + ")")
    if e.description:
        parts.append("(hint: " + quote(clip(e.description, 120)) + ")")
    line = " ".join(parts)
    if e.options:
        more = f" | … {len(e.options) - 6} more" if len(e.options) > 6 else ""
        line += "  options: " + " | ".join(e.options[:6]) + more
    return line


def _fit(src: list[str], budget: float) -> int:
    size = 0
    for i, line in enumerate(src):
        size += len(line) + 1
        if size > budget:
            return i
    return len(src)


# Budget tiers, tried in order until the page fits: (clip text lines to N chars, drop text lines longer than M).
# Controls, headings and short labels are the part the model acts on, so prose goes first, never the controls.
_TIERS = [(None, None), (70, None), (70, 40)]


def _assemble(obs: Observation, text_clip: int | None, drop_over: int | None) -> list[str]:
    by_id = obs.by_id
    body: list[str] = []
    omitted = 0
    in_dialog = False  # a blocking dialog's prose is what the model must read: never dropped, clipped at 160

    def flush_omitted() -> None:
        nonlocal omitted
        if omitted:
            body.append(f"(… {omitted} long text line{'s' if omitted > 1 else ''} omitted)")
            omitted = 0

    for ln in obs.lines:
        if ln.kind == "text":
            if in_dialog:
                body.append(clip(ln.text, 160))
                continue
            if drop_over is not None and len(ln.text) > drop_over:
                omitted += 1
                continue
            flush_omitted()
            body.append(clip(ln.text, text_clip) if text_clip else ln.text)
            continue
        flush_omitted()
        if ln.kind == "overlay":
            in_dialog = not ln.text.startswith("end")
        if ln.kind == "heading":
            body.append(f"## {ln.text}")
        elif ln.kind == "frame":
            body.append(f"--- frame {ln.text} ---")
        elif ln.kind == "overlay":
            body.append(f"--- {ln.text} ---")
        elif ln.kind == "alert":
            body.append(f"! {clip(ln.text, 200)}")  # live-region messages are evidence: never dropped or shortened further
        elif ln.kind == "control" and ln.element_id in by_id:
            body.append(element_line(by_id[ln.element_id]))
    flush_omitted()
    return body


def render(obs: Observation, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    body: list[str] = []
    for text_clip, drop_over in _TIERS:
        body = _assemble(obs, text_clip, drop_over)
        if _fit(body, max_chars) == len(body):
            break
    if not body:
        body = [
            f"(the page did not load: {obs.nav_error})" if obs.nav_error
            else "(nothing in the accessibility tree: the page is blank, or a popup, often in a cross-origin iframe, hides it)"
        ]
    if _fit(body, max_chars) < len(body):  # still too long: keep the top and the last fifth
        head = _fit(body, max_chars * 0.8)
        tail = _fit(body[::-1], max_chars * 0.2)
        body = [*body[:head], f"... {len(body) - head - tail} lines cut here (observe(max_chars=...) for more)", *body[len(body) - tail:]]
    header = f"[state] {obs.title} | {obs.url}"
    if obs.pending_requests:
        header += f" | loading: {obs.pending_requests}"
    for d in obs.dialogs:
        header += f"\n{d}"
    return well_formed(header + "\n" + "\n".join(body) + "\n[/state]")
