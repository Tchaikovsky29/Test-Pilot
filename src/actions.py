"""Actions on the ids the model sees. Every action goes through the same locator the generated test will use, so a
step that works during exploration is a step that will replay. Policy decides first; failures come back as plain
sentences the model can act on ("covered by a dialog: use its controls first"), never as stack traces.
"""
from __future__ import annotations

import json
import re
from typing import Annotated, Literal, Union

from playwright.sync_api import Error as PlaywrightError
from pydantic import BaseModel, Field, TypeAdapter

from .browser import BrowserSession
from .evidence import Changes, diff, render_changes
from .locators import ensure_locator, resolve, root_for
from .models import Element, Locator, Observation
from .policy import Policy
from .render import DEFAULT_MAX_CHARS, render
from .text import clip, quote
from .watch import describe_change, now_ms

CLICK_TIMEOUT_MS = 4000
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
NOT_TEXT_FIELDS = {"button", "link", "checkbox", "radio", "switch", "clickable", "tab", "menuitem", "option", "select"}


# ---- what the model may ask for ---------------------------------------------------------------------------------

class Click(BaseModel):
    kind: Literal["click"] = "click"
    id: int


class Type(BaseModel):
    kind: Literal["type"] = "type"
    id: int
    text: str
    keys: bool = False  # real key presses: needed by autocomplete boxes that ignore a plain fill
    submit: bool = False  # press Enter afterwards


class Select(BaseModel):
    kind: Literal["select"] = "select"
    id: int
    option: str


class Press(BaseModel):
    kind: Literal["press"] = "press"
    key: str  # Enter, Escape, Tab, ArrowDown, ...


class Scroll(BaseModel):
    kind: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"] = "down"


class Wait(BaseModel):
    kind: Literal["wait"] = "wait"
    ms: int = 1000


class GoBack(BaseModel):
    kind: Literal["go_back"] = "go_back"


Action = Annotated[Union[Click, Type, Select, Press, Scroll, Wait, GoBack], Field(discriminator="kind")]
_ACTION = TypeAdapter(Action)


def parse_action(data: dict | str) -> Action:
    """Validate model output; raises pydantic.ValidationError on anything that is not a known action."""
    return _ACTION.validate_json(data) if isinstance(data, str) else _ACTION.validate_python(data)


# ---- what comes back ----------------------------------------------------------------------------------------------

class Step(BaseModel):
    """One successful action, with enough to become a line of a Playwright test."""
    kind: str
    locator: Locator | None = None
    frame_url: str = ""
    value: str | None = None
    keys: bool = False
    element_name: str = ""
    widget: str = ""
    url: str = ""  # page the step ran on

    def code(self) -> str:
        root = "page" if not self.frame_url else f"page.frame(url={json.dumps(self.frame_url)})"
        target = self.locator.python(root) if self.locator else ""
        v = json.dumps(self.value, ensure_ascii=False)
        if self.kind == "click":
            return f"{target}.click()"
        if self.kind == "type":
            return f"{target}.press_sequentially({v})" if self.keys else f"{target}.fill({v})"
        if self.kind == "select":
            return f"{target}.select_option(label={v})"
        if self.kind == "press":
            return f"page.keyboard.press({v})"
        if self.kind == "go_back":
            return "page.go_back()"
        return f"# {self.kind}"


class ActionResult(BaseModel):
    status: Literal["ok", "failed", "blocked", "stopped"]
    message: str
    step: Step | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _fail(msg: str) -> ActionResult:
    return ActionResult(status="failed", message=msg)


class Actor:
    def __init__(self, session: BrowserSession, policy: Policy, click_timeout_ms: int = CLICK_TIMEOUT_MS):
        self.session, self.policy, self.click_timeout_ms = session, policy, click_timeout_ms
        self.obs: Observation | None = None
        self.changes: Changes = Changes()
        self.steps: list[Step] = []
        self.count = 0

    # -- looking ---------------------------------------------------------------------------------------------------
    def look(self, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """The page as the model sees it, followed by what changed since the previous look (new text, alerts,
        field errors). `self.changes` keeps the same information as data, for the test designer."""
        before = self.obs
        self.obs = self.session.observe()
        self.changes = diff(before, self.obs)
        return render(self.obs, max_chars) + render_changes(self.changes)

    # -- the one entry point ---------------------------------------------------------------------------------------
    def act(self, action: Action) -> ActionResult:
        if self.count >= self.policy.max_actions:
            return _fail(f"action budget used up ({self.policy.max_actions}); stop and report what was found")
        self.count += 1
        try:
            return getattr(self, f"_{action.kind}")(action)
        except PlaywrightError as e:
            return _fail(self._explain(e))

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _explain(self, e: PlaywrightError) -> str:
        s = str(e)
        m = re.search(r"(<[a-zA-Z][^\n]{0,100}?>)[^\n]*intercepts pointer events", s)
        if m:
            why = f"it is covered by {clip(m.group(1), 90)}"
            if self.obs and self.obs.blocked:
                why += "; a dialog is blocking the page, so use the dialog's own controls first"
            return why
        if "outside of the viewport" in s:
            return "it is outside the viewport and cannot be scrolled into view"
        if "not visible" in s:
            return "it is not visible"
        if "not enabled" in s or "disabled" in s:
            return "it is disabled"
        if "detached" in s or "Element is not attached" in s:
            return "it disappeared from the page; look again"
        if "Timeout" in s:
            return "timed out waiting for it to become usable"
        return clip(s.splitlines()[0], 200)

    def _target(self, element_id: int):
        if self.obs is None:
            return _fail("look at the page first")
        e = self.obs.element(element_id)
        if e is None:
            return _fail(f"[{element_id}] is not on the page as of the last look; look again")
        loc = ensure_locator(self.session.page, e, self.obs.elements)
        if loc is None:
            if all((c.matches or 0) == 0 for c in e.locators):
                return _fail(f'[{e.id}] {e.widget} "{e.display_name}" is no longer on the page; look again')
            return _fail(f'[{e.id}] {e.widget} "{e.display_name}" cannot be told apart from similar elements')
        return e, loc, resolve(root_for(self.session.page, e.frame_url), loc)

    def _label(self, e: Element) -> str:
        return f'[{e.id}] {e.widget} "{clip(e.display_name, 60)}"'

    def _done(self, what: str, before: dict | None, t0: float, step: Step) -> ActionResult:
        self.session.watcher.settle(cap_ms=3000, since=t0)
        page = self.session.page
        verdict = self.policy.check_url(page.url)
        if not verdict.ok:  # the action carried us off the allowed site: undo it
            try:
                page.go_back(wait_until="domcontentloaded", timeout=5000)
            except PlaywrightError:
                pass
            return ActionResult(status="blocked", message=f"{what}, but it {verdict.reason}; went back")
        change = describe_change(before, self.session.snapshot())
        msg = f"{what} -> {change}"
        if self.session.dialogs:
            msg += " | " + " ".join(self.session.dialogs)
            self.session.dialogs.clear()
        step.url = page.url
        self.steps.append(step)
        return ActionResult(status="ok", message=msg, step=step)

    # -- actions ---------------------------------------------------------------------------------------------------
    def _click(self, a: Click) -> ActionResult:
        t = self._target(a.id)
        if isinstance(t, ActionResult):
            return t
        e, loc, target = t
        v = self.policy.check_click(e)
        if not v.ok:
            return ActionResult(status="blocked", message=f"blocked: {v.reason}")
        if v.critical:
            return ActionResult(status="stopped", message=f"stopped before clicking: {v.reason}")
        before, t0 = self.session.snapshot(), now_ms()
        try:
            target.click(timeout=self.click_timeout_ms)
        except PlaywrightError as ex:
            return _fail(f"could not click {self._label(e)}: {self._explain(ex)}")
        return self._done(f"clicked {self._label(e)}", before, t0,
                          Step(kind="click", locator=loc, frame_url=e.frame_url, element_name=e.name, widget=e.widget))

    def _type(self, a: Type) -> ActionResult:
        t = self._target(a.id)
        if isinstance(t, ActionResult):
            return t
        e, loc, target = t
        if e.widget in NOT_TEXT_FIELDS:
            return _fail(f"{self._label(e)} is not a text field; use click" + (" or select" if e.widget == "select" else ""))
        if "readonly" in e.states:
            return _fail(f"{self._label(e)} is read-only: click it to open its picker or calendar, then click an option")
        if "disabled" in e.states:
            return _fail(f"{self._label(e)} is disabled")
        v = self.policy.check_type(e)
        if v.critical:
            return ActionResult(status="stopped", message=f"stopped before typing: {v.reason}")
        if e.widget == "date" and e.input_type == "date" and not ISO_DATE.fullmatch(a.text):
            return _fail(f"{self._label(e)} is a native date field: type the date as YYYY-MM-DD")
        before, t0 = self.session.snapshot(), now_ms()
        try:
            if a.keys:
                target.clear(timeout=self.click_timeout_ms)
                target.press_sequentially(a.text, delay=40, timeout=self.click_timeout_ms)
            else:
                target.fill(a.text, timeout=self.click_timeout_ms)
            if a.submit:
                target.press("Enter", timeout=self.click_timeout_ms)
        except PlaywrightError as ex:
            return _fail(f"could not type into {self._label(e)}: {self._explain(ex)}")
        return self._done(f"typed {quote(clip(a.text, 60))} into {self._label(e)}" + (" and pressed Enter" if a.submit else ""),
                          before, t0, Step(kind="type", locator=loc, frame_url=e.frame_url, value=a.text, keys=a.keys,
                                           element_name=e.name, widget=e.widget))

    def _select(self, a: Select) -> ActionResult:
        t = self._target(a.id)
        if isinstance(t, ActionResult):
            return t
        e, loc, target = t
        if e.widget != "select":
            return _fail(f"{self._label(e)} is not a native select: click it, then click one of its options")
        label = next((o for o in e.options if o.lower() == a.option.lower().strip()), None)
        if label is None:
            return _fail(f"{self._label(e)} has no option {quote(a.option)}; options: " + " | ".join(e.options))
        before, t0 = self.session.snapshot(), now_ms()
        try:
            target.select_option(label=label, timeout=self.click_timeout_ms)
        except PlaywrightError as ex:
            return _fail(f"could not select in {self._label(e)}: {self._explain(ex)}")
        return self._done(f"selected {quote(label)} in {self._label(e)}", before, t0,
                          Step(kind="select", locator=loc, frame_url=e.frame_url, value=label, element_name=e.name, widget=e.widget))

    def _press(self, a: Press) -> ActionResult:
        before, t0 = self.session.snapshot(), now_ms()
        try:
            self.session.page.keyboard.press(a.key)
        except PlaywrightError as ex:
            return _fail(f"could not press {a.key}: {clip(str(ex).splitlines()[0], 120)}")
        return self._done(f"pressed {a.key}", before, t0, Step(kind="press", value=a.key))

    def _scroll(self, a: Scroll) -> ActionResult:
        page = self.session.page
        h = page.evaluate("() => innerHeight")
        before, t0 = self.session.snapshot(), now_ms()
        page.mouse.wheel(0, h * 0.8 * (1 if a.direction == "down" else -1))
        page.wait_for_timeout(150)
        return self._done(f"scrolled {a.direction}", before, t0, Step(kind="scroll", value=a.direction))

    def _wait(self, a: Wait) -> ActionResult:
        ms = max(200, min(a.ms, 5000))
        before, t0 = self.session.snapshot(), now_ms()
        self.session.page.wait_for_timeout(ms)
        return self._done(f"waited {ms} ms", before, t0, Step(kind="wait", value=str(ms)))

    def _go_back(self, a: GoBack) -> ActionResult:
        before, t0 = self.session.snapshot(), now_ms()
        if self.session.page.go_back(wait_until="domcontentloaded", timeout=8000) is None:
            return _fail("there is no earlier page to go back to")
        return self._done("went back", before, t0, Step(kind="go_back"))