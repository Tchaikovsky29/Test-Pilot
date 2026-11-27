"""Test-case verifiers: each designed case is carried out on the real application in a browser of its own, several at
a time, and turned into a test definition the compiler can write as code (FR-05, FR-08).

Per case:
  1. open the application URL in a fresh browser context;
  2. carry out each step. A step that repeats a recorded exploration step is replayed without the model: the same
     control is found again by its locator or name and the same action is taken. Any other step (or a replay that no
     longer works) is handed to the model, which may take a few actions to complete it, through the same Actor as the
     explorer, so the safety policy and locator checks still apply;
  3. check every expected outcome against the page as it is now and against what the last step changed;
  4. return the steps that actually worked (with verified locators) and the checks that held, as assertions.

The model never writes code. A case ends "verified", "unconfirmed" (it ran, but an expected outcome was not seen),
"failed" (a step could not be done) or "blocked" (the safety policy stopped it, or the site could not be opened).
"""
from __future__ import annotations

import dataclasses
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Callable, Literal

from pydantic import BaseModel, Field, model_validator

from .actions import Action, ActionResult, Actor, Click, GoBack, Press, Select, Step, Type
from .browser import BrowserConfig, BrowserSession, NavigationError
from .designer import AssertKind, CaseStep, DesignResult, ExpectKind, Expectation, Gap, TestCase
from .evidence import Changes, field_errors
from .explorer import ACTIONS_HELP, LIMITS_RULES, PAGE_VIEW_RULES
from .llm import LLMError
from .locators import ensure_locator
from .models import Element, Locator, Observation
from .policy import Policy
from .text import clip, fold, shows

CaseStatus = Literal["verified", "unconfirmed", "failed", "blocked"]
_DUPLICATE = re.compile(r"\bduplicate", re.I)
MAX_AGENT_ACTIONS = 6
SETTLE_AGAIN_MS = 1500  # extra wait before re-reading a page whose saved values had not appeared  # per step: open a dropdown, type, pick the option, ... but not wander off
DEFAULT_CASE_SECONDS = 180.0
SKIPPED_KINDS = ("scroll", "wait")  # exploration moves, not test steps: locators do not need the element on screen


# ---- results ------------------------------------------------------------------------------------------------------

class Assertion(BaseModel):
    """One check the compiler writes as an expect(...). field_value: a field holds this value."""
    kind: AssertKind
    value: str
    locator: Locator | None = None  # the field (field_error) or control (element_visible) it is about
    frame_url: str = ""
    native: bool = False  # a field_error from the browser's own validation bubble: not page text
    label: str = ""  # field_shows: the label the saved value is shown under


class ExpectationResult(BaseModel):
    kind: ExpectKind
    expected: str
    status: Literal["confirmed", "corrected", "not_seen"]
    actual: str = ""  # what the page showed instead (or the corrected value)


class StepResult(BaseModel):
    n: int
    description: str
    status: Literal["done", "failed", "blocked", "stopped", "skipped"]
    how: Literal["replayed", "agent", ""] = ""
    message: str = ""
    actions: int = 0


class VerifiedCase(BaseModel):
    case: TestCase
    status: CaseStatus
    reason: str = ""
    steps: list[StepResult] = Field(default_factory=list)
    expectations: list[ExpectationResult] = Field(default_factory=list)
    definition: list[Step] = Field(default_factory=list)  # the steps that worked: what the compiler writes
    assertions: list[Assertion] = Field(default_factory=list)  # the checks that held
    notices: list[str] = Field(default_factory=list)  # what the site warned about without failing the case (duplicates)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    elapsed_s: float = 0.0


class VerificationResult(BaseModel):
    url: str
    requirement: str
    requirement_points: list[str]
    cases: list[VerifiedCase] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    concurrency: int = 1
    usage: str = ""
    elapsed_s: float = 0.0

    def counts(self) -> dict[str, int]:
        return {s: sum(c.status == s for c in self.cases) for s in ("verified", "unconfirmed", "failed", "blocked")}


# ---- the model's part: carry out one step -------------------------------------------------------------------------

class StepDecision(BaseModel):
    """One action towards the current step, or done, or failed (with the reason)."""
    note: str = ""
    action: Action | None = None
    done: bool = False
    failed: bool = False
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, data):
        if isinstance(data, dict) and "action" not in data and "kind" in data:
            meta = {k: data[k] for k in ("note", "done", "failed", "reason") if k in data}
            return {**meta, "action": {k: v for k, v in data.items() if k not in meta}}
        return data

    @model_validator(mode="after")
    def _one(self):
        if self.done or self.failed:
            self.action = None
        elif self.action is None:
            raise ValueError('reply with an "action", or "done": true, or "failed": true with a "reason"')
        return self


SYSTEM = ("""You carry out one step of a test case in a real web browser, through a text view of the page. The steps \
before it have already been done.

""" + PAGE_VIEW_RULES + """METHOD
1. Do exactly what the current step says, with the value given, and nothing else: do not go on to later steps.
2. A step may take a few actions (open a dropdown, then pick the option; type into an autocomplete box with keys: true, \
then click the matching choice). Check [changes since last look] to see whether the last action worked.
3. When the step is complete, reply done. If it cannot be done on this page (the control is not there, or it keeps \
failing), reply failed with the reason: do not improvise a different test.
4. Steps that test validation must be done as written: if the step says to leave a field empty or enter a wrong value, \
do that.

""" + LIMITS_RULES + ACTIONS_HELP + """REPLY with JSON only, one of:
{"note": "<what you expect this to show>", "action": {...}}
{"done": true, "note": "<what shows the step is complete>"}
{"failed": true, "reason": "<why the step cannot be done>"}""")


def _step_line(s: CaseStep) -> str:
    bits = [s.description]
    if s.target:
        bits.append(f'control: "{s.target}"')
    if s.value:
        bits.append(f'value: "{s.value}"')
    return " | ".join(bits)


def build_step_messages(case: TestCase, n: int, tried: list[str], page_text: str, today: str) -> list[dict]:
    steps = "\n".join(f"{'→' if i == n else ' '} {i}. {_step_line(s)}" for i, s in enumerate(case.steps, 1))
    task = (f"Today is {today}.\nTest case {case.id}: {case.title}\nPreconditions: "
            f"{'; '.join(case.preconditions) or 'none'}\nTest data: "
            f"{', '.join(f'{k} = {v}' for k, v in case.test_data.items()) or 'none'}\nSteps (→ marks the current one):\n{steps}")
    now = f"Current step {n}: {_step_line(case.steps[n - 1])}\n"
    now += ("Actions taken for this step so far:\n" + "\n".join(tried) + "\n") if tried else ""
    now += f"\nCurrent page:\n<page>\n{page_text}\n</page>\n\nReply with JSON only."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task, "cache": True},
            {"role": "user", "content": now}]


# ---- one case -----------------------------------------------------------------------------------------------------

class _Run:
    """One case in progress: the actor, the current page text, and what the last step changed."""

    def __init__(self, actor: Actor):
        self.actor = actor
        self.page_text = actor.look()
        self.step_changes: list[Changes] = []  # what each action of the current step changed
        self.dialogs: list[str] = []  # native alert/confirm messages, reported in action messages
        self.notices: list[str] = []  # warnings worth reporting that do not fail the case

    def do(self, action: Action) -> ActionResult:
        r = self.actor.act(action)
        if "[dialog " in r.message:
            self.dialogs.append(r.message[r.message.index("[dialog "):])
        self.page_text = self.actor.look()
        self.step_changes.append(self.actor.changes)
        ch = self.actor.changes
        for t in ch.alerts + ch.appeared:  # a re-run with the same data: the site's own duplicate warning
            if _DUPLICATE.search(t) and t not in self.notices:
                self.notices.append(clip(t, 200))
        return r

    def last_changes(self) -> Changes:
        """Everything the current step changed, merged: a validation message can appear one action before the end."""
        out = Changes()
        for ch in self.step_changes:
            out.navigated = out.navigated or ch.navigated
            out.url = ch.url or out.url
            out.appeared += ch.appeared
            out.disappeared += ch.disappeared
            out.alerts += ch.alerts
            out.errors += ch.errors
        return out


def _find(obs: Observation, rec: Step) -> Element | None:
    """The element a recorded step used, on the page as it is now: by the recorded locator, else by name."""
    if rec.locator is not None:
        want = (rec.locator.kind, rec.locator.role, rec.locator.value)
        hits = [e for e in obs.elements if e.frame_url == rec.frame_url
                and any((l.kind, l.role, l.value) == want for l in e.locators)]
        if len(hits) == 1:
            return hits[0]
        if hits and rec.locator.nth is not None and rec.locator.nth < len(hits):
            return hits[rec.locator.nth]
    named = [e for e in obs.elements if e.name == rec.element_name and (not rec.widget or e.widget == rec.widget)]
    return named[0] if len(named) == 1 else None


def _replay(run: _Run, rec: Step, value: str) -> ActionResult | None:
    """Repeat a recorded step. None when its control cannot be found (the model then takes over)."""
    if rec.kind == "press":
        return run.do(Press(key=rec.value or "Enter"))
    if rec.kind == "go_back":
        return run.do(GoBack())
    e = None
    for _ in range(3):
        e = _find(run.actor.obs, rec)
        if e is not None:
            break
        run.actor.session.watcher.settle(cap_ms=1500)
        run.page_text = run.actor.look()
    if e is None:
        return None
    if rec.kind == "click":
        return run.do(Click(id=e.id))
    if rec.kind == "type":
        # key presses or not is decided from the field as it is now (a suggestion box gets them): recordings made
        # before that rule typed every box key by key, which could lose the last key
        return run.do(Type(id=e.id, text=value, submit=rec.submit))
    if rec.kind == "select":
        return run.do(Select(id=e.id, option=value))
    return None


def _agent_step(run: _Run, llm, case: TestCase, n: int, deadline: float, today: str) -> tuple[str, str, int]:
    """Let the model carry out step n. Returns (status, message, actions taken)."""
    tried: list[str] = []
    for _ in range(MAX_AGENT_ACTIONS + 1):
        if time.monotonic() > deadline:
            return "failed", "ran out of time", len(tried)
        try:
            d = llm.structured(build_step_messages(case, n, tried, run.page_text, today), StepDecision)
        except LLMError as e:
            return "failed", f"model error: {e}", len(tried)
        if d.done:
            return "done", d.note or "done", len(tried)
        if d.failed:
            return "failed", d.reason or "the model could not do this step", len(tried)
        if len(tried) >= MAX_AGENT_ACTIONS:
            break
        r = run.do(d.action)
        tried.append(f"- {r.message}" if r.ok else f"- {r.status.upper()}: {r.message}")
        if r.status in ("blocked", "stopped"):
            return r.status, r.message, len(tried)
    return "failed", f"not complete after {MAX_AGENT_ACTIONS} actions", len(tried)


def _contains(haystack: list[str], needle: str) -> bool:
    n = fold(needle).strip()
    return bool(n) and any(n in fold(h) for h in haystack if h)


def _locator_for(run: _Run, e: Element) -> Locator | None:
    try:
        return ensure_locator(run.actor.session.page, e, run.actor.obs.elements)
    except Exception:
        return None


class _Seen:
    """The page now and what the last step changed, read once for all of a case's checks."""

    def __init__(self, run: _Run):
        self.run, self.obs, self.last = run, run.actor.obs, run.last_changes()
        self.page = [ln.text for ln in self.obs.lines] + [e.display_name for e in self.obs.elements]
        self.readable = self.obs.readable()
        self.errors = field_errors(self.obs)
        self.alerts = [ln.text for ln in self.obs.lines if ln.kind == "alert"] + self.last.alerts + run.dialogs
        # a page-level message after a step that stayed put: an alert region, a native dialog, or a plain toast
        self.notice = (self.alerts or ([] if self.last.navigated else self.last.appeared))[:1]

    def on(self, e: Element, kind, value: str, **kw) -> Assertion:
        return Assertion(kind=kind, value=value, locator=_locator_for(self.run, e), frame_url=e.frame_url, **kw)


# Each check returns (status, actual, assertion). "corrected" only ever replaces a value the designer guessed.
Outcome = tuple[str, str, Assertion | None]


def _text_visible(s: _Seen, x: Expectation) -> Outcome:
    if _contains(s.page + s.last.appeared + s.last.alerts, x.value):
        return "confirmed", "", Assertion(kind=x.kind, value=x.value)
    e = next((e for e in s.obs.elements if e.value and _contains([e.value], x.value)), None)
    if e is not None:  # held by a field: a typed or picked date is not page text
        return "confirmed", "", s.on(e, "field_value", e.value)
    return "not_seen", "", None


def _text_gone(s: _Seen, x: Expectation) -> Outcome:
    if not _contains(s.page, x.value):
        return "confirmed", "", Assertion(kind=x.kind, value=x.value)
    return "not_seen", "", None


def _url_contains(s: _Seen, x: Expectation) -> Outcome:
    want = fold(x.value).strip()
    if want and want in fold(s.obs.url):
        return "confirmed", "", Assertion(kind=x.kind, value=x.value)
    return "not_seen", s.obs.url, None


def _element_visible(s: _Seen, x: Expectation) -> Outcome:
    e = next((e for e in s.obs.elements if _contains([e.display_name], x.value)), None)
    return ("confirmed", "", s.on(e, x.kind, x.value)) if e is not None else ("not_seen", "", None)


def _field_error(s: _Seen, x: Expectation) -> Outcome:
    hit = next(((e, m) for e, m in s.errors if _contains([m, f"{e.display_name}: {m}"], x.value)), None)
    if hit is not None:
        return "confirmed", "", s.on(hit[0], x.kind, hit[1], native=bool(hit[0].validation))
    if _contains(s.last.errors, x.value):  # reported, but the field is no longer in error
        return "confirmed", "", Assertion(kind="text_visible", value=x.value)
    if not x.observed and s.errors:  # a guessed message: the field's own one is the fact
        e, m = s.errors[0]
        return "corrected", m, s.on(e, x.kind, m, native=bool(e.validation))
    if not x.observed and s.notice:  # many sites report a missing field in a page-level message instead
        msg = clip(s.notice[0], 200)
        return "corrected", msg, Assertion(kind="alert" if s.alerts else "text_visible", value=msg)
    shown = "; ".join(f"{e.display_name}: {m}" for e, m in s.errors[:3])
    return "not_seen", shown, None


def _field_shows(s: _Seen, x: Expectation) -> Outcome:
    """A saved value under its label on the page now (a record page): the persisted data, not just a message."""
    if shows(s.readable, x.label, x.value):
        return "confirmed", "", Assertion(kind=x.kind, label=x.label, value=x.value)
    lab = fold(x.label).strip().rstrip(":")
    at = next((i for i, t in enumerate(s.readable) if fold(t).strip().rstrip(":") == lab), None)
    shown = " | ".join(s.readable[at + 1:at + 3]) if at is not None else f'no "{x.label}" on the page'
    return "not_seen", clip(shown, 200), None


def _alert(s: _Seen, x: Expectation) -> Outcome:
    if _contains(s.alerts, x.value):
        return "confirmed", "", Assertion(kind=x.kind, value=x.value)
    if not x.observed and s.alerts:
        msg = clip(s.alerts[0], 200)
        return "corrected", msg, Assertion(kind=x.kind, value=msg)
    return "not_seen", "", None


_CHECKS = {"text_visible": _text_visible, "text_gone": _text_gone, "url_contains": _url_contains,
           "element_visible": _element_visible, "field_error": _field_error, "alert": _alert,
           "field_shows": _field_shows}


def check(run: _Run, expected: list[Expectation]) -> tuple[list[ExpectationResult], list[Assertion]]:
    """Each expected outcome against the page now and what the last step changed. An outcome the designer only
    guessed (never observed) that is contradicted by a concrete message is corrected to what the page said."""
    seen = _Seen(run)
    if any(x.kind == "field_shows" and _field_shows(seen, x)[0] != "confirmed" for x in expected):
        # a record page fills its values in after it appears: give it a moment and read it again, once
        run.actor.session.watcher.settle(cap_ms=3000)
        run.actor.session.page.wait_for_timeout(SETTLE_AGAIN_MS)
        run.page_text = run.actor.look()
        seen = _Seen(run)
    results, asserts = [], []
    for x in expected:
        status, actual, a = _CHECKS[x.kind](seen, x)
        if status == "not_seen" and not actual and (seen.last.appeared or seen.last.alerts or seen.last.errors):
            actual = clip("; ".join((seen.last.alerts + seen.last.errors + seen.last.appeared)[:4]), 300)
        expected = f"{x.label}: {x.value}" if x.kind == "field_shows" else x.value
        results.append(ExpectationResult(kind=x.kind, expected=expected, status=status, actual=actual))
        if a is not None and not any((b.kind, b.value, b.locator) == (a.kind, a.value, a.locator) for b in asserts):
            asserts.append(a)  # several guesses can resolve to the same message: check it once
    return results, asserts


def verify_case(case: TestCase, url: str, llm, browser: BrowserConfig, *, max_seconds: float = DEFAULT_CASE_SECONDS,
                today: str | None = None) -> VerifiedCase:
    """Carry out one case in a browser of its own. Never raises for problems on the site or with the model."""
    t0 = time.monotonic()
    deadline = t0 + max_seconds
    today = today or date.today().isoformat()
    out = VerifiedCase(case=case, status="failed")

    run: _Run | None = None

    def finish(status: CaseStatus, reason: str = "") -> VerifiedCase:
        out.status, out.reason = status, reason
        out.notices = list(run.notices) if run is not None else []
        out.elapsed_s = round(time.monotonic() - t0, 1)
        u = getattr(llm, "usage", None)
        if u is not None:
            out.llm_calls, out.prompt_tokens = getattr(u, "calls", 0), getattr(u, "prompt_tokens", 0)
            out.completion_tokens, out.cached_tokens = getattr(u, "completion_tokens", 0), getattr(u, "cached_tokens", 0)
        return out

    with BrowserSession(browser) as session:
        try:
            session.goto(url)
        except NavigationError as e:
            return finish("blocked", str(e))
        actor = Actor(session, Policy(start_url=url, allow_external_links=True))
        actor.policy.allow_site_of(session.page.url)  # where the start URL redirected to
        run = _Run(actor)
        for n, step in enumerate(case.steps, 1):
            run.step_changes = []
            res = StepResult(n=n, description=step.description, status="done")
            rec = step.recorded
            if rec is not None and rec.kind in SKIPPED_KINDS:
                res.status, res.message = "skipped", f"{rec.kind}: not needed to reach a control"
                out.steps.append(res)
                continue
            r = _replay(run, rec, step.value or rec.value or "") if rec is not None else None
            if r is not None and r.ok:
                res.how, res.message, res.actions = "replayed", r.message, 1
            elif r is not None and r.status in ("blocked", "stopped"):
                res.how, res.status, res.message, res.actions = "replayed", r.status, r.message, 1
            else:
                status, msg, count = _agent_step(run, llm, case, n, deadline, today)
                res.how, res.status, res.message, res.actions = "agent", status, msg, count
                if r is not None:  # the replay was tried first
                    res.message = f"replay failed ({clip(r.message, 100)}); {msg}"
            out.steps.append(res)
            if res.status in ("blocked", "stopped"):
                out.definition = [s for s in actor.steps if s.kind not in SKIPPED_KINDS]
                return finish("blocked", f"step {n}: {res.message}")
            if res.status == "failed":
                out.definition = [s for s in actor.steps if s.kind not in SKIPPED_KINDS]
                return finish("failed", f"step {n}: {res.message}")

        out.definition = [s for s in actor.steps if s.kind not in SKIPPED_KINDS]
        out.expectations, out.assertions = check(run, case.expected)
        if not case.expected:
            return finish("unconfirmed", "the case has no checkable expected outcome")
        missed = [x for x in out.expectations if x.status == "not_seen"]
        if missed:
            return finish("unconfirmed", "; ".join(
                f'{x.kind} "{clip(x.expected, 60)}" not seen' + (f" (page showed: {clip(x.actual, 120)})" if x.actual else "")
                for x in missed))
        fixed = [x for x in out.expectations if x.status == "corrected"]
        return finish("verified", "; ".join(dict.fromkeys(
            f'{x.kind} corrected to what the page showed: "{clip(x.actual, 100)}"' for x in fixed)))


# ---- all cases, several at a time ---------------------------------------------------------------------------------

def verify(design: DesignResult, llm_factory: Callable[[TestCase], object], browser: BrowserConfig, *,
           concurrency: int = 3, max_seconds: float = DEFAULT_CASE_SECONDS, today: str | None = None,
           on_case: Callable[[VerifiedCase], None] | None = None) -> VerificationResult:
    """Verify every case, up to `concurrency` at once. Each worker gets its own browser context and its own LLM
    client (from `llm_factory`), so nothing is shared between cases but the site itself."""
    t0 = time.monotonic()
    concurrency = max(1, concurrency)
    worker_browser = dataclasses.replace(browser, fresh_context=True)

    def work(case: TestCase) -> VerifiedCase:
        try:
            return verify_case(case, design.url, llm_factory(case), worker_browser, max_seconds=max_seconds, today=today)
        except Exception as e:  # a crashed browser in one worker must not take the others down
            first = str(e).splitlines()[0] if str(e) else type(e).__name__
            return VerifiedCase(case=case, status="failed", reason=f"verifier crashed: {clip(first, 200)}")

    results: list[VerifiedCase] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in as_completed([pool.submit(work, c) for c in design.cases]):
            vc = fut.result()
            results.append(vc)
            if on_case:
                on_case(vc)
    results.sort(key=lambda v: v.case.id)
    calls = sum(v.llm_calls for v in results)
    pt, ct, cached = (sum(getattr(v, k) for v in results) for k in ("prompt_tokens", "completion_tokens", "cached_tokens"))
    return VerificationResult(url=design.url, requirement=design.requirement, requirement_points=design.requirement_points,
                              cases=results, gaps=design.gaps, concurrency=concurrency,
                              usage=f"{calls} call(s), {pt} prompt ({cached} cached) + {ct} completion tokens",
                              elapsed_s=round(time.monotonic() - t0, 1))
