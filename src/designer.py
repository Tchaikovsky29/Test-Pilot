"""Test-case designer: one model call that turns an exploration report into 5-10 structured test cases (FR-05, FR-06),
followed by checks in plain code that tie every case back to what the browser actually showed.

The model proposes; the code decides what is grounded:
  * a step that cites an exploration event must cite one that succeeded, and then carries that event's verified
    locator; a step that does not cite one must name a control that was seen on the page;
  * an expected outcome counts as observed only if its text really appears in the recorded evidence (the changes an
    action caused, or the pages that were seen); otherwise it stays an inference for the verifier to confirm.
Each case ends up "observed", "partial" or "unverified", with the reasons listed, so nothing the model invented can
pass for a fact. Requirement points with no case are reported as gaps, with the model's reason when it gave one.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .actions import Step
from .explorer import Event, ExplorationReport, PageInfo
from .llm import LLMError
from .text import clip, fold

TestType = Literal["Positive", "Negative", "Boundary", "Validation", "Navigation", "E2E"]
Priority = Literal["High", "Medium", "Low"]
ExpectKind = Literal["text_visible", "text_gone", "field_error", "alert", "url_contains", "element_visible"]
AssertKind = Literal["text_visible", "text_gone", "field_error", "alert", "url_contains", "element_visible", "field_value"]
Grounding = Literal["observed", "partial", "unverified"]

MAX_CASES = 10
MAX_EVIDENCE_CHARS = 24000
# Up to ten cases of structured JSON, plus whatever a reasoning model spends thinking first: far more than one
# exploration decision needs. LLM_MAX_TOKENS still wins when it is set higher.
DESIGN_MAX_TOKENS = 16000

_TYPES = {"positive": "Positive", "negative": "Negative", "boundary": "Boundary", "validation": "Validation",
          "navigation": "Navigation", "e2e": "E2E", "end-to-end": "E2E", "end to end": "E2E"}
_POINT = re.compile(r"^\s*p?\s*(\d+)\s*$", re.I)


# ---- the schema ---------------------------------------------------------------------------------------------------

class CaseStep(BaseModel):
    """One ordered step. `event` replays a recorded exploration step; otherwise `target` names the control to use."""
    description: str  # what a tester does, in words
    event: int | None = None  # E<n> of the exploration step this repeats, if any
    target: str = ""  # the control's name as the page shows it (for steps that were not recorded)
    value: str = ""  # data typed or chosen, if any
    locator: str = ""  # filled by the code from the cited event: the verified Playwright locator
    recorded: Step | None = None  # filled by the code: the cited event's step, which a verifier can replay as is

    @field_validator("event", mode="before")
    @classmethod
    def _event_ref(cls, v):
        if isinstance(v, str):  # "E4" or "4"
            m = re.fullmatch(r"\s*e?\s*(\d+)\s*", v, re.I)
            return int(m.group(1)) if m else None
        return v


class Expectation(BaseModel):
    """A checkable claim about the outcome. `evidence` lists the events whose recorded changes show it."""
    kind: ExpectKind
    value: str  # the text, URL fragment or control name to check
    evidence: list[int] = Field(default_factory=list)
    observed: bool = False  # set by the code: the value was found in the recorded evidence

    @field_validator("evidence", mode="before")
    @classmethod
    def _event_refs(cls, v):
        items = v if isinstance(v, list) else [v] if v not in (None, "") else []
        out = []
        for x in items:
            m = re.fullmatch(r"\s*e?\s*(\d+)\s*", str(x), re.I)
            if m:
                out.append(int(m.group(1)))
        return out


class TestCase(BaseModel):
    __test__ = False  # not a pytest class, despite the name

    id: str = ""  # assigned by the code: TC-01, TC-02, ...
    title: str
    module: str  # application module or page
    covers: list[int] = Field(default_factory=list)  # requirement point numbers (P1 = 1)
    scenario: str = ""  # the requirement aspect validated
    type: TestType
    priority: Priority
    priority_reason: str = ""
    preconditions: list[str] = Field(default_factory=list)
    test_data: dict[str, str] = Field(default_factory=dict)
    steps: list[CaseStep]
    expected_result: str  # in words
    expected: list[Expectation] = Field(default_factory=list)  # the same, as checks
    evidence: list[str] = Field(default_factory=list)  # filled by the code: what supports this case
    grounding: Grounding = "unverified"  # filled by the code
    problems: list[str] = Field(default_factory=list)  # filled by the code: why it is not fully observed

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, v):
        return _TYPES.get(str(v).strip().lower(), v)

    @field_validator("priority", mode="before")
    @classmethod
    def _priority(cls, v):
        return str(v).strip().capitalize() if isinstance(v, str) else v

    @field_validator("covers", mode="before")
    @classmethod
    def _points(cls, v):
        items = v if isinstance(v, list) else [v]
        return [int(m.group(1)) for x in items if (m := _POINT.match(str(x)))]

    @field_validator("test_data", mode="before")
    @classmethod
    def _data(cls, v):
        if isinstance(v, list):  # [{"field": .., "value": ..}] or ["City: Pune"]
            out = {}
            for x in v:
                if isinstance(x, dict) and len(x) >= 2:
                    k, val = list(x.values())[:2]
                    out[str(k)] = str(val)
                elif isinstance(x, str) and ":" in x:
                    k, _, val = x.partition(":")
                    out[k.strip()] = val.strip()
            return out
        if isinstance(v, dict):
            return {str(k): "" if x is None else str(x) for k, x in v.items()}
        return v


class Gap(BaseModel):
    point: int
    reason: str

    @field_validator("point", mode="before")
    @classmethod
    def _point(cls, v):
        m = _POINT.match(str(v))
        return int(m.group(1)) if m else v


class DesignReply(BaseModel):
    """What the model returns."""
    cases: list[TestCase]
    unsupported: list[Gap] = Field(default_factory=list)


class DesignResult(BaseModel):
    """What the designer writes: the input to the verifiers, and later the export."""
    url: str
    requirement: str
    requirement_points: list[str]
    exploration_stop: str = ""  # how the exploration ended, so a reader knows how far the evidence goes
    cases: list[TestCase] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)  # requirement points no case covers
    dropped: list[str] = Field(default_factory=list)  # cases the code removed, and why
    problem: str = ""  # set when the model call failed
    usage: str = ""


# ---- the prompt ---------------------------------------------------------------------------------------------------

SYSTEM = """You are a senior QA engineer. You are given a requirement, its numbered requirement points, and the evidence a \
browser explorer recorded on the real application: each step it took (E1, E2, ...) with what that step changed on the \
page, and the pages it saw with their fields and buttons. Design test cases for the requirement from this evidence.

RULES
- Use only what the evidence shows. Never invent pages, controls, validation messages, business rules or outcomes.
- Every step either repeats a recorded step (set "event" to its number; only successful steps can be repeated) or names a \
control exactly as it appears in the evidence ("target"). Never cite a FAILED or BLOCKED step as something to repeat.
- A FAILED step is a failure of the explorer (it could not find or use the control), not behaviour of the application: \
never design a case that expects such a failure.
- Expected outcomes must be checkable. Give each as {"kind", "value", "evidence"}: kind is one of text_visible, \
text_gone, field_error, alert, url_contains, element_visible; value is the exact text, URL fragment or control name as \
recorded (for field_error and alert: the message text itself, never just the field's name); evidence lists the event \
numbers whose changes show it. If you expect something the evidence does not show \
(for example a validation message nobody triggered), still write it, with an empty evidence list: it will be checked.
- Each case runs on its own, in a fresh browser opened at the application URL: list every step needed to get from \
there to what it checks (dismissing a notice, opening a tab, filling the fields before the one under test).
- 5 to 10 cases, most important first: the main end-to-end flow, then validation and negative cases on the same \
screens (empty required field, invalid or out-of-range value, past date, same value where two must differ), then \
boundary and navigation cases. No duplicates or trivial variations. Stay inside the requirement.
- Never enter payment details, card numbers, passwords or one-time codes, and never confirm a payment or final booking: \
the flow ends at the step before.
- A requirement point the evidence never reached (the explorer stopped before it, or the feature is not there) gets no \
case: list it under "unsupported" with the reason.

REPLY with JSON only:
{"cases": [{"title": "...", "module": "page or area", "covers": [1, 2], "scenario": "requirement aspect validated",
  "type": "Positive|Negative|Boundary|Validation|Navigation|E2E", "priority": "High|Medium|Low",
  "priority_reason": "...", "preconditions": ["..."], "test_data": {"field": "value"},
  "steps": [{"description": "...", "event": 3, "target": "", "value": ""}],
  "expected_result": "in words",
  "expected": [{"kind": "field_error", "value": "...", "evidence": [5]}]}],
 "unsupported": [{"point": 4, "reason": "..."}]}"""


def _no_ids(message: str) -> str:
    """Element ids like [171] mean nothing outside the live page."""
    return re.sub(r"\[\d+\]\s*", "", message)


def _event_line(e: Event) -> str:
    msg = _no_ids(e.message)
    head = f"E{e.n} {e.status.upper() if e.status != 'ok' else 'ok'}: {clip(msg, 220)}"
    if e.note:
        head += f"\n    intent: {clip(e.note, 160)}"
    ch = e.changes
    if ch and not ch.empty:
        bits = []
        if ch.navigated:
            bits.append(f"navigated to {ch.url}")
        bits += [f'+ "{clip(t, 140)}"' for t in ch.appeared[:12]]
        bits += [f'! "{clip(t, 140)}"' for t in ch.alerts[:6]]
        bits += [f'error: "{clip(t, 160)}"' for t in ch.errors[:8]]
        bits += [f'- "{clip(t, 100)}"' for t in ch.disappeared[:4]]
        head += "\n    changed: " + "\n             ".join(bits)
    return head


def _page_block(p: PageInfo) -> str:
    out = [f"page {p.url}" + (f'  "{p.title}"' if p.title else "")]
    if p.headings:
        out.append("  headings: " + " | ".join(clip(h, 80) for h in p.headings))
    for f in p.fields:
        bits = [f'"{clip(f.name, 80)}" {f.widget}']
        bits += [s for s, on in (("required", f.required), ("readonly", f.readonly), ("autocomplete", f.autocomplete)) if on]
        if f.input_type and f.input_type != "text":
            bits.append(f"type={f.input_type}")
        if f.constraints:
            bits.append(", ".join(f"{k}={v}" for k, v in f.constraints.items()))
        if f.options:
            bits.append("options: " + ", ".join(clip(o, 40) for o in f.options[:12]))
        if f.values_tried:
            bits.append("tried: " + ", ".join(f'"{clip(v, 40)}"' for v in f.values_tried))
        out.append("  field " + " ".join(bits))
    if p.buttons:
        out.append("  buttons: " + ", ".join(f'"{clip(b, 60)}"' for b in p.buttons))
    return "\n".join(out)


def evidence_text(report: ExplorationReport) -> str:
    """The exploration as the designer model reads it: steps with their effects, then the pages seen."""
    stop = report.stop_reason + (f" ({report.stop_detail})" if report.stop_detail else "")
    parts = [f"EXPLORATION of {report.url} — ended: {stop}"]
    if report.stopped_at:
        parts.append(f"a person was needed at: {report.stopped_at}")
    parts.append("\nSTEPS")
    parts += [_event_line(e) for e in report.events] or ["(no steps were taken)"]
    parts.append("\nPAGES SEEN")
    parts += [_page_block(p) for p in report.pages] or ["(none)"]
    return clip("\n".join(parts), MAX_EVIDENCE_CHARS)


def build_messages(report: ExplorationReport) -> list[dict]:
    points = report.plan.requirement_points or [report.requirement]
    task = (f"Requirement: {report.requirement}\nRequirement points:\n"
            + "\n".join(f"P{i}. {p}" for i, p in enumerate(points, 1)))
    if report.plan.assumptions:
        task += "\nAssumptions:\n" + "\n".join(f"- {a}" for a in report.plan.assumptions)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{task}\n\n{evidence_text(report)}\n\nReply with JSON only."}]


# ---- grounding: the code's own check of what the model proposed ---------------------------------------------------

class _Evidence:
    """Everything the browser recorded, indexed for matching."""

    def __init__(self, report: ExplorationReport):
        self.events = {e.n: e for e in report.events}
        self.pages = report.pages
        names: set[str] = set()
        for p in report.pages:
            names.update(fold(f.name) for f in p.fields)
            names.update(fold(b) for b in p.buttons)
        for e in report.events:
            names.update(fold(n) for n in re.findall(r'"([^"]+)"', e.message))
        self.control_names = {n for n in names if n}

    def texts(self, kind: str, e: Event) -> list[str]:
        ch = e.changes
        if kind == "url_contains":
            return [e.url] + ([ch.url] if ch and ch.url else [])
        if ch is None:
            return [e.message] if kind == "alert" else []
        return {"text_visible": ch.appeared + ch.alerts + ch.errors, "text_gone": ch.disappeared,
                "field_error": ch.errors + ch.alerts, "alert": ch.alerts + ch.errors + [e.message],
                "element_visible": ch.appeared}.get(kind, [])

    def page_texts(self, kind: str) -> list[str]:
        if kind == "url_contains":
            return [p.url for p in self.pages]
        if kind in ("text_visible", "element_visible"):
            return [t for p in self.pages for t in [p.title, *p.headings, *p.buttons, *(f.name for f in p.fields)]]
        return []

    def known_control(self, name: str) -> bool:
        n = fold(name).strip()
        return bool(n) and any(n == c or (len(n) >= 4 and (n in c or c in n)) for c in self.control_names)


def _contains(haystack: list[str], needle: str) -> bool:
    n = fold(needle).strip()
    return bool(n) and any(n in fold(h) for h in haystack if h)


def ground(case: TestCase, ev: _Evidence, n_points: int) -> TestCase:
    """Check one case against the evidence and record what supports it and what does not."""
    problems: list[str] = []
    support: list[str] = []
    steps_ok = True
    for i, step in enumerate(case.steps, 1):
        if step.event is not None:
            e = ev.events.get(step.event)
            if e is None or e.step is None:
                why = ("does not exist" if e is None else f"did not succeed ({e.status})" if e.status != "ok"
                       else "has no recorded locator (a report from an older version: explore again)")
                problems.append(f"step {i} cites E{step.event}, which {why}")
                step.event = None
                if not step.target:  # nothing else to check it by: one problem is enough
                    steps_ok = False
                    continue
            else:
                step.locator = e.step.locator.python() if e.step.locator else ""
                step.recorded = e.step
                step.value = step.value or (e.step.value or "")
                support.append(f"step {i} repeats E{e.n}: {clip(_no_ids(e.message), 120)}")
                continue
        if step.target and ev.known_control(step.target):
            continue
        steps_ok = False
        problems.append(f'step {i} uses "{clip(step.target or step.description, 60)}", which was not seen on the page')

    for x in case.expected:
        cited = [ev.events[n] for n in x.evidence if n in ev.events]
        hit = next((e for e in cited if _contains(ev.texts(x.kind, e), x.value)), None)
        if hit is None:  # the model may cite the wrong step, or none: look through all of them
            hit = next((e for e in ev.events.values() if _contains(ev.texts(x.kind, e), x.value)), None)
        if hit is not None:
            x.evidence, x.observed = [hit.n], True
            support.append(f'{x.kind} "{clip(x.value, 60)}" seen after E{hit.n}')
        elif _contains(ev.page_texts(x.kind), x.value):
            x.evidence, x.observed = [], True
            support.append(f'{x.kind} "{clip(x.value, 60)}" seen on a page')
        else:
            x.evidence, x.observed = [], False
            problems.append(f'expected {x.kind} "{clip(x.value, 60)}" was never observed')
    if not case.expected:
        problems.append("no checkable expected outcome")

    bad = [p for p in case.covers if not 1 <= p <= n_points]
    if bad:
        problems.append("covers unknown requirement point(s) " + ", ".join(f"P{p}" for p in bad))
        case.covers = [p for p in case.covers if 1 <= p <= n_points]

    observed = [x for x in case.expected if x.observed]
    if steps_ok and case.expected and len(observed) == len(case.expected):
        case.grounding = "observed"
    elif observed or any(s.event is not None for s in case.steps):
        case.grounding = "partial"
    else:
        case.grounding = "unverified"
    case.evidence, case.problems = support, problems
    return case


def _key(case: TestCase) -> tuple:
    steps = tuple((s.event, fold(s.target), fold(s.value)) for s in case.steps)
    checks = tuple(sorted((x.kind, fold(x.value)) for x in case.expected))
    return steps, checks


def finalize(reply: DesignReply, report: ExplorationReport) -> DesignResult:
    """Ground every case, drop duplicates and excess, number them, and list the requirement points left uncovered."""
    points = report.plan.requirement_points or [report.requirement]
    ev = _Evidence(report)
    result = DesignResult(url=report.url, requirement=report.requirement, requirement_points=points,
                          exploration_stop=report.stop_reason + (f" ({report.stop_detail})" if report.stop_detail else ""))
    seen: set[tuple] = set()
    for case in reply.cases:
        if not case.steps:
            result.dropped.append(f'"{clip(case.title, 60)}": no steps')
            continue
        key = _key(case)
        if key in seen:
            result.dropped.append(f'"{clip(case.title, 60)}": duplicate of an earlier case')
            continue
        if len(result.cases) >= MAX_CASES:
            result.dropped.append(f'"{clip(case.title, 60)}": over the limit of {MAX_CASES} cases')
            continue
        seen.add(key)
        result.cases.append(ground(case, ev, len(points)))
    for i, case in enumerate(result.cases, 1):
        case.id = f"TC-{i:02d}"

    covered = {p for c in result.cases for p in c.covers}
    reasons = {g.point: g.reason for g in reply.unsupported if 1 <= g.point <= len(points)}
    result.gaps = [Gap(point=p, reason=reasons.get(p, "no test case was designed for it"))
                   for p in range(1, len(points) + 1) if p not in covered]
    return result


def design(llm, report: ExplorationReport) -> DesignResult:
    """One model call, then the code's checks. A failed call gives an empty result with the reason, never an exception."""
    try:
        budget = max(DESIGN_MAX_TOKENS, getattr(getattr(llm, "config", None), "max_tokens", 0) or 0)
        reply = llm.structured(build_messages(report), DesignReply, max_tokens=budget)
    except LLMError as e:
        points = report.plan.requirement_points or [report.requirement]
        return DesignResult(url=report.url, requirement=report.requirement, requirement_points=points,
                            gaps=[Gap(point=p, reason="test design failed") for p in range(1, len(points) + 1)],
                            problem=str(e), usage=llm.usage.summary() if hasattr(llm, "usage") else "")
    result = finalize(reply, report)
    result.usage = llm.usage.summary() if hasattr(llm, "usage") else ""
    return result
