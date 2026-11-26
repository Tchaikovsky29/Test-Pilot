"""Observation -> the text the model reads. One line per control, in page order:

    [id] widget "name" type=email = "value" required invalid  options: a | b | c
A leading * marks controls that were not in the previous look.

Three things keep the view short:
  * a blocking modal is shown alone: the page behind it cannot be used, so it is not sent;
  * otherwise only lines near the viewport are listed in full; the rest becomes one outline line per direction
    ("⋮ below the view ..."), and `scroll` brings it into the listing;
  * what is still too long is trimmed in tiers (prose first, never controls), then cut in the middle.
"""
from __future__ import annotations

from .models import Element, Line, Observation
from .text import clip, quote, well_formed

DEFAULT_MAX_CHARS = 4000
ABOVE, BELOW = 0.25, 0.25  # viewport heights beyond the edge that are still listed in full
OUTLINE_SECTIONS = 6


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


# ---- what to leave out before any trimming --------------------------------------------------------------------------

def _dialog_block(obs: Observation) -> tuple[list[Line], int] | None:
    """(the dialog's lines, how many page controls are hidden behind it), when an overlay leaves the page behind it
    unclickable and has at least one control of its own. Then only the dialog matters: nothing else can be clicked.
    An overlay the page can be clicked around (a banner, a popup without a backdrop) is listed first, but the page
    stays. A dialog with no control to press is left alone too: hiding the page could leave the model stuck."""
    if not obs.modal:
        return None
    inside: list[Line] = []
    hidden, in_dialog = 0, False
    for ln in obs.lines:
        if ln.kind == "overlay":
            in_dialog = not ln.text.startswith("end")
            inside.append(ln)
        elif in_dialog:
            inside.append(ln)
        elif ln.kind == "control":
            hidden += 1
    if not any(ln.kind == "control" for ln in inside):
        return None
    return inside, hidden


def _side(ln: Line, lo: float, hi: float, width: float) -> str | None:
    """None when the line is on screen (or its place is unknown), else which way it lies."""
    if ln.y is not None:
        if ln.y < lo:
            return "above"
        if ln.y > hi:
            return "below"
    if width and ln.x is not None and not (-0.1 * width <= ln.x <= width):
        return "aside"  # a horizontal carousel: scrolling down will not reach it
    return None


def _outline(direction: str, sections: list[list]) -> str:
    total = sum(n for _, n in sections)
    parts = [(f"## {clip(h, 40)}" if h else "controls") + (f" ({n})" if n else "") for h, n in sections[:OUTLINE_SECTIONS]]
    if len(sections) > OUTLINE_SECTIONS:
        parts.append(f"+{len(sections) - OUTLINE_SECTIONS} more")
    return f"⋮ {direction} the view, {total} control{'s' if total != 1 else ''}: " + " | ".join(parts) + \
           f"  (scroll {'up' if direction == 'above' else 'down'} to list them)"


def _window(obs: Observation) -> tuple[list[Line], list[str]]:
    """Lines near the viewport, and the outline lines that stand for everything else. A focused field, an invalid
    field, an alert or a dialog is always kept: that is where the model's next move is, wherever it sits."""
    h, w = obs.viewport_h, obs.viewport_w
    if not h:
        return obs.lines, []
    lo, hi = obs.scroll_y - ABOVE * h, obs.scroll_y + (1 + BELOW) * h
    by_id = obs.by_id
    kept: list[Line] = []
    outline: dict[str, list[list]] = {"above": [], "below": []}  # direction -> [[heading or None, controls], ...]
    aside, in_dialog, last_heading = 0, False, None
    for ln in obs.lines:
        if ln.kind == "overlay":
            in_dialog = not ln.text.startswith("end")
            kept.append(ln)
            continue
        e = by_id.get(ln.element_id) if ln.kind == "control" else None
        pinned = in_dialog or ln.kind in ("alert", "frame") or (
            e is not None and ("focused" in e.states or "invalid" in e.states or bool(e.validation)))
        side = None if pinned else _side(ln, lo, hi, w)
        if side is None:
            kept.append(ln)
            if ln.kind == "heading":
                last_heading = ln.text
        elif side == "aside":
            aside += ln.kind == "control"
        elif ln.kind == "heading":
            outline[side].append([ln.text, 0])
        elif ln.kind == "control":
            if not outline[side]:  # the rest of a section whose heading is still in view
                outline[side].append([last_heading, 0])
            outline[side][-1][1] += 1
    tail = [_outline(d, s) for d, s in outline.items() if s]
    if aside:
        tail.append(f"⋮ {aside} more control{'s' if aside > 1 else ''} sit off to the side (a carousel?): use its arrow or next control")
    return kept, tail


# ---- text ------------------------------------------------------------------------------------------------------------

# Budget tiers, tried in order until the page fits: (clip text lines to N chars, drop text lines longer than M).
# Controls, headings and short labels are the part the model acts on, so prose goes first, never the controls.
_TIERS = [(None, None), (70, None), (70, 40)]


def _assemble(obs: Observation, lines: list[Line], text_clip: int | None, drop_over: int | None) -> list[str]:
    by_id = obs.by_id
    body: list[str] = []
    omitted = 0
    in_dialog = False  # a blocking dialog's prose is what the model must read: never dropped, clipped at 160
    for ln in lines:
        if ln.kind == "text":
            if in_dialog:
                body.append(clip(ln.text, 160))
            elif drop_over is not None and len(ln.text) > drop_over:
                omitted += 1
            else:
                body.append(clip(ln.text, text_clip) if text_clip else ln.text)
            continue
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
    if omitted:  # one note for the whole page, not one per gap
        body.append(f"(… {omitted} long text line{'s' if omitted > 1 else ''} omitted)")
    return body


def render(obs: Observation, max_chars: int = DEFAULT_MAX_CHARS, window: bool = True) -> str:
    """`window=False` lists the whole page (the explorer asks for that when it keeps failing, in case the control it
    needs was left out of the view)."""
    lines, tail = obs.lines, []
    dialog = _dialog_block(obs)
    if dialog is not None:
        lines, hidden = dialog
        tail = [f"(page behind the dialog is hidden: {hidden} control{'s' if hidden != 1 else ''}; it returns when the dialog closes)"]
    elif window:
        lines, tail = _window(obs)
    body: list[str] = []
    for text_clip, drop_over in _TIERS:
        body = _assemble(obs, lines, text_clip, drop_over) + tail
        if _fit(body, max_chars) == len(body):
            break
    if not body:
        body = [
            f"(the page did not load: {obs.nav_error})" if obs.nav_error
            else "(nothing in the accessibility tree: the page is blank, or a popup, often in a cross-origin iframe, hides it)"
        ]
    if _fit(body, max_chars) < len(body):  # still too long: keep the top and the last fifth
        head = _fit(body, max_chars * 0.8)
        tail_n = _fit(body[::-1], max_chars * 0.2)
        body = [*body[:head], f"... {len(body) - head - tail_n} lines cut here (observe(max_chars=...) for more)", *body[len(body) - tail_n:]]
    header = f"[state] {obs.title} | {obs.url}"
    if obs.pending_requests:
        header += f" | loading: {obs.pending_requests}"
    for d in obs.dialogs:
        header += f"\n{d}"
    return well_formed(header + "\n" + "\n".join(body) + "\n[/state]")