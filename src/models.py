"""Data models shared by every stage. Pure data: no browser code in here."""
import hashlib
import json
import re
from typing import Literal
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from .text import clean_name

LocatorKind = Literal["role", "label", "placeholder", "testid", "text", "css"]

class Locator(BaseModel):
    """One way to find an element again. Backend node ids die with the page; these survive into generated tests."""

    kind: LocatorKind
    value: str  # accessible name / label / placeholder / test id / text / css selector
    role: str | None = None  # only for kind="role"
    matches: int | None = Field(default=None, description="live match count; exactly 1 means usable")
    exact: bool = True  # False = substring match (used with the icon-free form of a name)
    visible: bool = False  # narrowed with .filter(visible=True) to drop hidden duplicates
    nth: int | None = None  # last resort for identical siblings (fragile: depends on page order)

    @property
    def fragile(self) -> bool:
        return self.nth is not None

    def python(self, root: str = "page") -> str:
        q = json.dumps(self.value, ensure_ascii=False)
        ex = ", exact=True" if self.exact else ""
        if self.kind == "role":
            base = f"{root}.get_by_role({json.dumps(self.role)}, name={q}{ex})"
        elif self.kind == "label":
            base = f"{root}.get_by_label({q}{ex})"
        elif self.kind == "placeholder":
            base = f"{root}.get_by_placeholder({q}{ex})"
        elif self.kind == "testid":
            base = f"{root}.get_by_test_id({q})"
        elif self.kind == "text":
            base = f"{root}.get_by_text({q}{ex})"
        else:
            base = f"{root}.locator({q})"
        if self.visible:
            base += ".filter(visible=True)"
        if self.nth is not None:
            base += f".nth({self.nth})"
        return base


class Element(BaseModel):
    """An actionable control as the agent sees it. `id` is Chrome's backendNodeId, valid for the current page only."""

    id: int
    role: str  # ARIA role from Chrome's accessibility tree
    widget: str  # what to show the model: select, date, file, textbox, clickable, ...
    name: str = ""
    value: str = ""
    states: list[str] = Field(default_factory=list)  # required, invalid, disabled, checked, ...
    options: list[str] = Field(default_factory=list)  # native <select> options
    tag: str = ""
    input_type: str = ""
    required: bool = False
    description: str = ""  # aria-describedby text: hints and validation messages live here
    validation: str = ""  # the browser's own error for this field, only after the user (or an agent) tried to submit
    constraints: dict[str, str] = Field(default_factory=dict)  # pattern, minlength, maxlength, min, max, step
    href: str = ""
    frame_url: str = ""  # "" for the main frame
    is_new: bool = False  # not present in the previous observation
    locators: list[Locator] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        """The name without a leading icon-font word, for the model to read. Locators keep using `name`."""
        return clean_name(self.name)

    def best_locator(self) -> Locator | None:
        return next((loc for loc in self.locators if loc.matches == 1), None)


class Line(BaseModel):
    kind: Literal["heading", "text", "control", "frame", "overlay", "alert"]
    text: str = ""
    element_id: int | None = None
    x: float | None = None  # left and top edge in document coordinates; None = unknown, so always listed
    y: float | None = None


_VARIABLE = re.compile(r"\d+|[0-9a-f]{8}-[0-9a-f-]{27}")
_SIG_SKIP = {"link", "option", "tab", "menuitem", "treeitem"}


class Observation(BaseModel):
    url: str
    title: str = ""
    lines: list[Line] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    dialogs: list[str] = Field(default_factory=list)
    pending_requests: int = 0
    nav_error: str | None = None
    blocked: bool = False  # a modal dialog covers the page: its lines come first
    modal: bool = False  # ...and it is a real modal (role=dialog / aria-modal / <dialog>): the page behind it is unusable
    scroll_y: float = 0.0  # where the viewport sits in the document
    viewport_w: float = 0.0
    viewport_h: float = 0.0  # 0 = unknown: the view is not windowed

    @property
    def by_id(self) -> dict[int, Element]:
        return {e.id: e for e in self.elements}

    def element(self, element_id: int) -> Element | None:
        return self.by_id.get(element_id)

    def signature(self) -> str:
        """A coarse page-type fingerprint: URL path pattern + form-ish controls + headings, with numbers erased.
        Typing into a field or a different search result list must not change it, and neither does a modal dialog
        that happens to be open (its lines sit between the overlay markers and are left out). Refined later for
        the site graph."""
        path = "/".join("{n}" if _VARIABLE.fullmatch(p) else p for p in urlparse(self.url).path.split("/"))
        dialog_controls: set[int] = set()
        dialog_headings: set[str] = set()
        in_dialog = False
        for ln in self.lines:
            if ln.kind == "overlay":
                in_dialog = not ln.text.startswith("end")
            elif in_dialog and ln.kind == "control" and ln.element_id is not None:
                dialog_controls.add(ln.element_id)
            elif in_dialog and ln.kind == "heading":
                dialog_headings.add(ln.text)
        fields = sorted({(e.widget, _VARIABLE.sub("#", e.name.lower())) for e in self.elements
                         if e.widget not in _SIG_SKIP and e.id not in dialog_controls})
        heads = sorted({_VARIABLE.sub("#", ln.text.lower()) for ln in self.lines
                        if ln.kind == "heading" and ln.text not in dialog_headings})
        raw = json.dumps([path, fields, heads], ensure_ascii=False)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]