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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Callable, Literal

from pydantic import BaseModel, Field, model_validator

from .actions import Action, ActionResult, Actor, Click, GoBack, Press, Select, Step, Type
from .browser import BrowserConfig, BrowserSession, NavigationError
from .designer import AssertKind, CaseStep, DesignResult, ExpectKind, Expectation, Gap, TestCase
from .evidence import Changes
from .explorer import ACTIONS_HELP, LIMITS_RULES, PAGE_VIEW_RULES
from .llm import LLMError
from .locators import ensure_locator
from .models import Element, Locator, Observation
from .policy import Policy
from .text import clip, fold

CaseStatus = Literal["verified", "unconfirmed", "failed", "blocked"]
MAX_AGENT_ACTIONS = 6  # per step: open a dropdown, type, pick the option, ... but not wander off
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

    def do(self, action: Action) -> ActionResult:
        r = self.actor.act(action)
        if "[dialog " in r.message:
            self.dialogs.append(r.message[r.message.index("[dialog "):])
        self.page_text = self.actor.look()
        self.step_changes.append(self.actor.changes)
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
        return run.do(Type(id=e.id, text=value, keys=rec.keys, submit=rec.submit))
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


def _field_errors(obs: Observation) -> list[tuple[Element, str]]:
    out = []
    for e in obs.elements:
        msg = e.validation or (e.description if "invalid" in e.states else "")
        if msg:
            out.append((e, msg))
    return out


def _locator_for(run: _Run, e: Element) -> Locator | None:
    try:
        return ensure_locator(run.actor.session.page, e, run.actor.obs.elements)
    except Exception:
        return None


def check(run: _Run, expected: list[Expectation]) -> tuple[list[ExpectationResult], list[Assertion]]:
    """Each expected outcome against the page now and what the last step changed. An outcome the designer only
    guessed (never observed) that is contradicted by a concrete message is corrected to what the page said."""
    obs, last = run.actor.obs, run.last_changes()
    page = [ln.text for ln in obs.lines] + [e.display_name for e in obs.elements]
    values = [e.value for e in obs.elements if e.value]  # what fields hold: a typed date is not page text
    errors = _field_errors(obs)
    alerts = [ln.text for ln in obs.lines if ln.kind == "alert"] + last.alerts + run.dialogs
    results, asserts = [], []
    for x in expected:
        res = ExpectationResult(kind=x.kind, expected=x.value, status="not_seen")
        a: Assertion | None = None
        if x.kind == "text_visible" and _contains(page + last.appeared + last.alerts, x.value):
            res.status, a = "confirmed", Assertion(kind=x.kind, value=x.value)
        elif x.kind == "text_visible" and _contains(values, x.value):
            e = next(e for e in obs.elements if e.value and _contains([e.value], x.value))
            res.status = "confirmed"
            a = Assertion(kind="field_value", value=e.value, locator=_locator_for(run, e), frame_url=e.frame_url)
        elif x.kind == "text_gone" and not _contains(page, x.value):
            res.status, a = "confirmed", Assertion(kind=x.kind, value=x.value)
        elif x.kind == "url_contains" and fold(x.value).strip() and fold(x.value).strip() in fold(obs.url):
            res.status, a = "confirmed", Assertion(kind=x.kind, value=x.value)
        elif x.kind == "element_visible":
            e = next((e for e in obs.elements if _contains([e.display_name], x.value)), None)
            if e is not None:
                res.status = "confirmed"
                a = Assertion(kind=x.kind, value=x.value, locator=_locator_for(run, e), frame_url=e.frame_url)
        elif x.kind == "field_error":
            hit = next(((e, m) for e, m in errors if _contains([m, f"{e.display_name}: {m}"], x.value)), None)
            if hit is None and errors and not x.observed:  # a guessed message: the page's own one is the fact
                hit = errors[0]
                res.status, res.actual = "corrected", hit[1]
            elif hit is not None:
                res.status = "confirmed"
            elif not errors and not x.observed and (last.alerts or run.dialogs or (last.appeared and not last.navigated)):
                # Sites often report a missing field in a page-level message (an alert region, a native dialog, or a
                # plain toast) rather than on the field itself. A toast only counts if the step stayed on the page:
                # a failed validation does not navigate.
                msg = (last.alerts or run.dialogs or last.appeared)[0]
                res.status, res.actual = "corrected", clip(msg, 200)
                a = Assertion(kind="alert" if (last.alerts or run.dialogs) else "text_visible", value=res.actual)
            if hit is not None:
                e, m = hit
                a = Assertion(kind=x.kind, value=m, locator=_locator_for(run, e), frame_url=e.frame_url,
                              native=bool(e.validation))
            elif _contains(last.errors, x.value):
                res.status, a = "confirmed", Assertion(kind="text_visible", value=x.value)
        elif x.kind == "alert":
            if _contains(alerts, x.value):
                res.status, a = "confirmed", Assertion(kind=x.kind, value=x.value)
            elif alerts and not x.observed:
                res.status, res.actual = "corrected", clip(alerts[0], 200)
                a = Assertion(kind=x.kind, value=res.actual)
        if res.status == "not_seen":
            if x.kind == "field_error" and errors:
                res.actual = "; ".join(f"{e.display_name}: {m}" for e, m in errors[:3])
            elif x.kind == "url_contains":
                res.actual = obs.url
            elif last.appeared or last.alerts or last.errors:
                res.actual = clip("; ".join((last.alerts + last.errors + last.appeared)[:4]), 300)
        results.append(res)
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

    def finish(status: CaseStatus, reason: str = "") -> VerifiedCase:
        out.status, out.reason = status, reason
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
