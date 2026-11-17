"""Perception: turn a live page into an Observation (text only, no screenshots).

Sources, all read over CDP in one pass:
  * Accessibility.getFullAXTree  -> roles, names, states, in page order (main frame + same-process child frames)
  * DOM.getDocument(pierce)      -> tag and attributes per node, <select> options (feeds stable locators)
  * a visible-text probe         -> drops opacity:0 / display:none text that the AX tree still contains
The approach follows Browser Use's ax.py (backendNodeIds as element refs, visible-text filtering, native select and
date handling, '*' for new elements); this is a fresh implementation on Playwright's CDP session.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

from playwright.sync_api import CDPSession
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from .text import clip, fold, is_icon_text, norm
from .locators import candidates
from .models import Element, Line, Observation
from .roles import BLOCKS, CHECKABLE, CONTROLS, DATE_INPUT_TYPES, NATIVE_INPUT_ROLES
from .text import clip, fold, norm
from .watch import PageWatcher, is_context_loss

# Text actually shown, lowercased with whitespace collapsed. The AX tree keeps opacity:0 text (pre-rendered success
# banners), which an agent would then report as success; checkVisibility with checkOpacity drops it.
VISIBLE_JS = r"""() => {
  const out = [], shown = new Map(), roots = [document];
  for (let i = 0; i < roots.length; i++) {
    const walk = document.createTreeWalker(roots[i], NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    for (let n = walk.nextNode(); n; n = walk.nextNode()) {
      if (n.nodeType === Node.ELEMENT_NODE) {
        if (n.shadowRoot) roots.push(n.shadowRoot);
        try { if (n.contentDocument) roots.push(n.contentDocument); } catch (e) {}
        continue;
      }
      let e = n.parentElement;
      if (!e || !n.textContent.trim()) continue;
      while (e.parentElement && getComputedStyle(e).display === 'contents') e = e.parentElement;
      if (!shown.has(e)) shown.set(e, e.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}));
      if (shown.get(e)) out.push(n.textContent);
    }
  }
  return out.join(' ').replace(/\s+/g, ' ').toLowerCase();
}"""


# The topmost thing at the centre of the viewport, climbed to its outermost modal ancestor: role=dialog / aria-modal /
# <dialog open>, or a fixed-position layer with a z-index that covers a fifth of the screen (a typical div modal).
OVERLAY_JS = r"""(() => {
  const w = innerWidth, h = innerHeight;
  const isOverlay = (e) => {
    if (e.matches('[role=dialog],[role=alertdialog],[aria-modal=true],dialog[open]')) return true;
    const cs = getComputedStyle(e);
    if (cs.position !== 'fixed' || !(parseInt(cs.zIndex, 10) > 0)) return false;
    const r = e.getBoundingClientRect();
    return r.width * r.height >= 0.2 * w * h;
  };
  let found = null;
  for (let e = document.elementFromPoint(w / 2, h / 2); e && e !== document.body && e !== document.documentElement; e = e.parentElement)
    if (isOverlay(e)) found = e;
  return found;
})()"""

# Elements that are interactive but expose no accessibility role (div tabs, custom dropdowns, date-picker triggers):
# the page attached a click listener, or styled them cursor:pointer. Chrome's DOM snapshot reports both in one call.
CLICK_SKIP_TAGS = {"HTML", "BODY", "INPUT", "SELECT", "TEXTAREA", "OPTION", "OPTGROUP", "IFRAME", "SCRIPT", "STYLE", "SVG", "PATH", "LABEL"}
MAX_CLICKABLE_TEXT = 160  # longer text means a container with delegated handlers, not a button


def find_clickables(snapshot: dict, viewport_area: float) -> set[int]:
    """backendNodeIds of rendered elements with a click listener or an outermost cursor:pointer."""
    strings = snapshot.get("strings") or []
    s = lambda i: strings[i] if isinstance(i, int) and 0 <= i < len(strings) else ""  # noqa: E731
    out: set[int] = set()
    for doc in snapshot.get("documents") or []:
        nodes, layout = doc.get("nodes") or {}, doc.get("layout") or {}
        parent, backend = nodes.get("parentIndex") or [], nodes.get("backendNodeId") or []
        names, types = nodes.get("nodeName") or [], nodes.get("nodeType") or []
        clicks = set((nodes.get("isClickable") or {}).get("index") or [])
        row = {ni: i for i, ni in enumerate(layout.get("nodeIndex") or [])}
        styles, bounds = layout.get("styles") or [], layout.get("bounds") or []
        pointer = {ni for ni, i in row.items() if i < len(styles) and styles[i] and s(styles[i][0]) == "pointer"}
        for ni in clicks | pointer:
            if ni not in row or ni >= len(types) or types[ni] != 1 or s(names[ni]).upper() in CLICK_SKIP_TAGS:
                continue
            if ni in pointer and ni not in clicks and parent[ni] in pointer:
                continue  # cursor is inherited: only the outermost element of a pointer chain counts
            x, y, w, h = (bounds[row[ni]] + [0, 0, 0, 0])[:4]
            if w * h < 100 or w * h > 0.6 * viewport_area:
                continue  # invisible specks, and page-sized wrappers with delegated handlers
            out.add(backend[ni])
    return out


@dataclass
class DomInfo:
    tag: str
    attrs: dict[str, str]
    options: list[str] = field(default_factory=list)


def _attr_dict(flat: list[str] | None) -> dict[str, str]:
    flat = flat or []
    return {flat[i]: flat[i + 1] for i in range(0, len(flat) - 1, 2)}


def _select_options(select: dict) -> list[str]:
    opts: list[str] = []
    stack = list(reversed(select.get("children") or []))
    while stack:
        n = stack.pop()
        name = (n.get("nodeName") or "").upper()
        if name == "OPTGROUP":
            stack.extend(reversed(n.get("children") or []))
        elif name == "OPTION":
            a = _attr_dict(n.get("attributes"))
            if "disabled" in a:
                continue
            text = norm("".join(c.get("nodeValue", "") for c in n.get("children") or [] if c.get("nodeType") == 3))
            opts.append(a.get("label") or text or a.get("value", ""))
    return opts


def index_dom(root: dict) -> dict[int, DomInfo]:
    """backendNodeId -> tag, attributes (and options for <select>), across iframes and shadow roots."""
    out: dict[int, DomInfo] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        tag = (n.get("nodeName") or "").upper()
        info = DomInfo(tag, _attr_dict(n.get("attributes")))
        if tag == "SELECT":
            info.options = _select_options(n)
        if n.get("backendNodeId") is not None:
            out[n["backendNodeId"]] = info
        stack.extend(n.get("children") or [])
        if n.get("contentDocument"):
            stack.append(n["contentDocument"])
        stack.extend(n.get("shadowRoots") or [])
    return out


def _val(o: Any) -> Any:
    return o.get("value") if isinstance(o, dict) else None


def _prop(n: dict, name: str) -> Any:
    return next(((p.get("value") or {}).get("value") for p in n.get("properties") or [] if p.get("name") == name), None)


def _short_href(href: str, base: str) -> str:
    if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
        return ""
    full = urljoin(base, href)
    a, b = urlparse(full), urlparse(base)
    path = (a.path or "/") + (f"?{a.query}" if a.query else "")
    return path if a.netloc == b.netloc else f"{a.netloc}{path}"


def _origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


class _FrameWalker:
    """Walks one frame's AX tree in page order and appends Lines and Elements."""

    def __init__(self, nodes: list[dict], dom: dict[int, DomInfo], visible: str | None, page_url: str,
                 frame_url: str, previous: set[int] | None, lines: list[Line], elements: list[Element],
                 clickables: set[int] | None = None, overlay: int | None = None):
        self.by_id = {n["nodeId"]: n for n in nodes}
        self.dom, self.visible, self.page_url, self.frame_url = dom, visible, page_url, frame_url
        self.previous, self.lines, self.elements = previous, lines, elements
        self.text: list[str] = []
        self.clickables, self.overlay = clickables or set(), overlay
        self.overlay_lines: list[Line] | None = None  # lines of the blocking dialog, hoisted by the caller
        self.in_control = 0  # >0 while inside a listed control: nothing nested becomes a "clickable"

    def flush(self) -> None:
        if not self.text:
            return
        t = norm(" ".join(self.text))
        self.text.clear()
        if not t or is_icon_text(t):
            return  # empty, or just an icon-font glyph name such as "vpn_key"
        last = self.elements[-1] if self.elements else None
        if self.lines and self.lines[-1].kind == "control" and last and fold(t) in (fold(last.name), fold(last.description)):
            return  # a label printed after its control, which already carries it as its name
        self.lines.append(Line(kind="text", text=clip(t, 300)))

    def walk(self, node: dict | None, inside: str) -> None:
        if not node:
            return
        if self.overlay is not None and node.get("backendDOMNodeId") == self.overlay and self.overlay_lines is None:
            self.flush()
            start = len(self.lines)
            self._walk(node, inside)
            self.flush()
            self.overlay_lines = self.lines[start:]
            del self.lines[start:]
            return
        self._walk(node, inside)

    def _subtree_text(self, node: dict) -> str:
        parts: list[str] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if _val(n.get("role")) == "StaticText" and not n.get("ignored"):
                t = norm(_val(n.get("name")))
                if t and (self.visible is None or fold(t) in self.visible):
                    parts.append(t)
            stack.extend(reversed([self.by_id[c] for c in n.get("childIds") or [] if c in self.by_id]))
        return norm(" ".join(parts))

    def _has_control(self, node: dict) -> bool:
        stack = [self.by_id[c] for c in node.get("childIds") or [] if c in self.by_id]
        while stack:
            n = stack.pop()
            if _val(n.get("role")) in CONTROLS and n.get("backendDOMNodeId") is not None and not n.get("ignored"):
                return True
            stack.extend(self.by_id[c] for c in n.get("childIds") or [] if c in self.by_id)
        return False

    def _walk(self, node: dict, inside: str) -> None:
        role = _val(node.get("role"))
        role = "" if role is None else str(role)
        name = norm(_val(node.get("name")))
        kids = [self.by_id.get(c) for c in node.get("childIds") or []]
        if node.get("ignored"):
            for c in kids:
                self.walk(c, inside)
            return
        if role == "StaticText":
            t = fold(name)
            if t and t not in inside and (self.visible is None or t in self.visible):
                self.text.append(name)
            return
        if role == "heading" and name and fold(name) in inside:
            return  # already said by the card link or control that contains it
        if role == "heading" and name:
            self.flush()
            self.lines.append(Line(kind="heading", text=name))
            for c in kids:
                self.walk(c, fold(name))
            self.flush()
            return
        bid = node.get("backendDOMNodeId")
        info = self.dom.get(bid) if bid is not None else None
        value = norm(_val(node.get("value")))
        focusable = _prop(node, "focusable") and role != "RootWebArea" and (name or value or _prop(node, "editable"))
        is_control = bid is not None and (role in CONTROLS or role in NATIVE_INPUT_ROLES or bool(focusable))
        widget_override = None
        if not is_control and bid in self.clickables and self.in_control == 0 and not self._has_control(node):
            label = name or self._subtree_text(node)
            if label and len(label) <= MAX_CLICKABLE_TEXT:
                is_control, name, widget_override = True, label, "clickable"
        if not is_control:
            if role in BLOCKS:
                self.flush()
                for c in kids:
                    self.walk(c, inside)
                self.flush()
            else:
                for c in kids:
                    self.walk(c, inside)
            return
        self.flush()
        if self.lines and self.lines[-1].kind == "text" and self.lines[-1].text.lower() == name.lower():
            self.lines.pop()  # its label, printed as its name
        el = self._element(node, role, name, value, bid, info, widget_override)
        self.elements.append(el)
        self.lines.append(Line(kind="control", element_id=el.id))
        if el.widget in ("select", "date", "file") or role in NATIVE_INPUT_ROLES:
            return  # children are the browser's own widget parts
        # Only what the line showed counts as said, except for links and clickables: a card link's long text is
        # repeated by its heading and paragraph inside, and printing it again only burns the budget.
        shown = name if (role == "link" or widget_override) else clip(name, 150)
        inside = fold(f"{inside} {shown} {value}")
        self.in_control += 1
        for c in kids:
            self.walk(c, inside)
        self.in_control -= 1
        self.flush()

    def _element(self, node: dict, role: str, name: str, value: str, bid: int, info: DomInfo | None,
                 widget_override: str | None = None) -> Element:
        tag = info.tag if info else ""
        attrs = info.attrs if info else {}
        itype = (attrs.get("type") or "text").lower() if tag == "INPUT" else ""
        editable = bool(_prop(node, "editable"))
        if tag == "SELECT":
            widget = "select"
        elif tag == "INPUT" and itype in DATE_INPUT_TYPES:
            widget = "date"
        elif tag == "INPUT" and itype == "file":
            widget = "file"
        elif role == "generic" and editable:
            widget = "textbox"
        else:
            widget = role
        if widget_override:
            widget = widget_override
        if itype == "password" and value:
            value = "(hidden)"  # secrets never reach the model, the logs or the saved report
        checked, expanded = _prop(node, "checked"), _prop(node, "expanded")
        required = bool(_prop(node, "required")) or "required" in attrs
        invalid = attrs.get("aria-invalid", "").lower() in ("true", "grammar", "spelling")
        native_disabled = "disabled" in attrs and tag in ("INPUT", "SELECT", "BUTTON", "TEXTAREA", "OPTION", "FIELDSET")
        states = [
            s for s in [
                "" if checked is None or role not in CHECKABLE else ("unchecked" if checked == "false" else "checked"),
                "" if expanded is None or widget == "select" else ("expanded" if expanded else "collapsed"),
                "selected" if _prop(node, "selected") else "",
                "required" if required else "",
                "invalid" if invalid else "",
                "readonly" if _prop(node, "readonly") else "",
                "disabled" if native_disabled or (_prop(node, "disabled") and not _prop(node, "focusable")) else "",
                "focused" if _prop(node, "focused") else "",
            ] if s
        ]
        el = Element(
            id=bid, role=role, widget=widget, name=name, value=value, states=states,
            options=info.options if info else [], tag=tag, input_type=itype, required=required,
            description="" if fold(norm(_val(node.get("description")))) in ("", fold(name)) else norm(_val(node.get("description"))),
            href=_short_href(attrs.get("href", ""), self.page_url) if role == "link" else "",
            frame_url=self.frame_url,
            is_new=self.previous is not None and bid not in self.previous,
        )
        el.locators = candidates(el, attrs)
        return el


class Perceiver:
    def __init__(self, page: Page, cdp: CDPSession, watcher: PageWatcher, dialogs: list[str]):
        self.page, self.cdp, self.watcher, self.dialogs = page, cdp, watcher, dialogs
        self._previous: set[int] | None = None
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))  # AX trees nest deeply

    def _retry(self, fn: Callable[[], Any]) -> Any:
        for i in range(21):
            try:
                return fn()
            except PlaywrightError as e:
                if i >= 20 or not is_context_loss(e):
                    raise
                self.page.wait_for_timeout(50)

    def _capture(self) -> list[dict]:
        """AX trees of the main frame and of every same-process child frame (cross-origin frames fail and are skipped)."""
        def run() -> list[dict]:
            nodes = self.cdp.send("Accessibility.getFullAXTree")["nodes"]
            kids: list[tuple[str, str]] = []

            def collect(tree: dict) -> None:
                for child in tree.get("childFrames") or []:
                    kids.append((child["frame"].get("url", ""), child["frame"]["id"]))
                    collect(child)

            collect(self.cdp.send("Page.getFrameTree")["frameTree"])
            frames = [{"url": "", "nodes": nodes}]
            for url, fid in kids:
                try:
                    frames.append({"url": url, "nodes": self.cdp.send("Accessibility.getFullAXTree", {"frameId": fid})["nodes"]})
                except PlaywrightError:
                    frames.append({"url": url, "nodes": []})
            return frames

        return self._retry(run)

    def _overlay_bid(self) -> int | None:
        try:
            r = self.cdp.send("Runtime.evaluate", {"expression": OVERLAY_JS, "returnByValue": False})
            oid = (r.get("result") or {}).get("objectId")
            if not oid:
                return None
            try:
                return self.cdp.send("DOM.describeNode", {"objectId": oid})["node"].get("backendNodeId")
            finally:
                self.cdp.send("Runtime.releaseObject", {"objectId": oid})
        except (PlaywrightError, KeyError):
            return None

    def _clickables(self) -> set[int]:
        try:
            area = self.page.evaluate("() => innerWidth * innerHeight")
            snap = self.cdp.send("DOMSnapshot.captureSnapshot", {"computedStyles": ["cursor"]})
            return find_clickables(snap, area)
        except (PlaywrightError, KeyError, TypeError):
            return set()  # perception degrades to accessibility-tree controls only

    def observe(self, settle: bool = True) -> Observation:
        settled = self.watcher.settle(cap_ms=3000) if settle else {"probe": None, "pending": 0}
        info = settled.get("probe") or self._retry(lambda: self.page.evaluate("() => ({url: location.href, title: document.title})"))
        frames = self._capture()
        dom = index_dom(self._retry(lambda: self.cdp.send("DOM.getDocument", {"depth": -1, "pierce": True}))["root"])
        try:
            visible = self._retry(lambda: self.page.evaluate(VISIBLE_JS))
        except PlaywrightError:
            visible = None
        clickables, overlay = self._clickables(), self._overlay_bid()
        lines: list[Line] = []
        elements: list[Element] = []
        hoisted: list[Line] = []
        for frame in frames:
            start = len(lines)
            # Text from a cross-origin frame is not in `visible`, so it is not filtered there.
            same_origin = not frame["url"] or _origin(frame["url"]) == _origin(info["url"]) or frame["url"] == "about:srcdoc"
            walker = _FrameWalker(frame["nodes"], dom, visible if same_origin else None, info["url"], frame["url"],
                                  self._previous, lines, elements, clickables, overlay)
            if frame["nodes"]:
                walker.walk(frame["nodes"][0], "")
                walker.flush()
            if walker.overlay_lines:
                hoisted = walker.overlay_lines
            if frame["url"] and len(lines) > start:
                lines.insert(start, Line(kind="frame", text=frame["url"]))
        if hoisted:  # a modal dialog covers the page: show it first, whatever its place in the document
            lines[:0] = [Line(kind="overlay", text="dialog blocking the page: handle it first"), *hoisted,
                         Line(kind="overlay", text="end of dialog")]
        self._previous = {e.id for e in elements}
        nav_error = None
        if info["url"].startswith("chrome-error:"):
            nav_error = self.watcher.nav_failure or "unknown network error"
        dialogs, self.dialogs[:] = list(self.dialogs), []
        return Observation(url=info["url"], title=info["title"], lines=lines, elements=elements, dialogs=dialogs,
                           pending_requests=settled.get("pending", 0), nav_error=nav_error, blocked=bool(hoisted))
