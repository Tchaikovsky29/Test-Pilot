"""test-pilot observe <url>             : print the page exactly as the model will see it (no LLM involved).
test-pilot llm                         : check the LLM configuration with one tiny call.
test-pilot explore <url> "<requirement>": explore the app with the model and write an evidence report.
test-pilot design <report.json>         : design structured test cases from an exploration report.
test-pilot verify <cases.json>          : carry out each case in its own browser, several at once, and keep what held."""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from .browser import BrowserConfig, BrowserSession, NavigationError
from .designer import DesignResult, design
from .verifier import DEFAULT_CASE_SECONDS, VerifiedCase, verify
from .explorer import Event, ExplorationReport, call_budget, explore
from .llm import LLM, PROVIDERS, LLMConfig, LLMError
from .locators import verify_all
from .render import DEFAULT_MAX_CHARS, render
from .text import clip


class _Ping(BaseModel):
    ok: bool
    say: str = ""


def _observe(args) -> int:
    cfg = BrowserConfig(headless=not args.headed)
    if args.cdp_url:
        cfg.cdp_url = args.cdp_url
    try:
        with BrowserSession(cfg) as s:
            s.goto(args.url)
            obs = s.observe()
            print(obs.model_dump_json(indent=2) if args.json else render(obs, args.max_chars, window=not args.full))
            if args.locators:
                print(f"\npage signature: {obs.signature()}\n")
                elems = [e for e in obs.elements if e.widget not in ("link", "option")]
                verify_all(s.page, elems)
                for e in elems:
                    best = e.best_locator()
                    tried = ", ".join(f"{l.kind}:{l.matches}" for l in e.locators)
                    note = "  [fragile: nth]" if best and best.fragile else ""
                    print(f"[{e.id}] {e.widget} {e.name!r}\n    {best.python() if best else 'NO UNIQUE LOCATOR'}{note}   ({tried})")
    except NavigationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _llm(args) -> int:
    try:
        cfg = LLMConfig.from_env(args.provider, args.model)
        llm = LLM(cfg)
        print(f"provider: {cfg.provider.name}   model: {cfg.model}   endpoint: {cfg.base_url}   key: set ({cfg.provider.key_env})")
        if args.models:
            print("\n".join(llm.list_models()))
            return 0
        t0 = time.monotonic()
        reply = llm.structured([{"role": "user", "content": 'Reply with JSON: {"ok": true, "say": "hello"}'}], _Ping)
        print(f"reply: {reply.model_dump_json()}   ({time.monotonic() - t0:.1f}s, {llm.usage.summary()})")
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _print_event(ev: Event) -> None:
    print(f"[{ev.n}] {ev.note or ev.action}")
    print(f"     {ev.status}: {ev.message}")
    ch = ev.changes
    if ch and not ch.empty:  # what the action actually did, which the action's own message cannot know
        for prefix, items in (("+", ch.appeared), ("-", ch.disappeared), ("!", ch.alerts), ("error:", ch.errors)):
            for t in items[:3]:
                print(f"       {prefix} {t}")


def _explore(args) -> int:
    try:
        cfg = LLMConfig.from_env(args.provider, args.model)
        if cfg.max_calls is None:  # a safety net against runaway retries
            cfg.max_calls = call_budget(args.max_steps)
        llm = LLM(cfg)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    bcfg = BrowserConfig(headless=not args.headed)
    if args.cdp_url:
        bcfg.cdp_url = args.cdp_url
    print(f"exploring {args.url} with {cfg.provider.name}/{cfg.model}\nrequirement: {args.requirement}\n")
    with BrowserSession(bcfg) as s:
        report = explore(s, llm, args.url, args.requirement, max_steps=args.max_steps, max_seconds=args.max_seconds,
                         on_event=None if args.quiet else _print_event)
    output = _report_output(args)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path))
    print(f"\nstopped: {report.stop_reason}" + (f" ({report.stop_detail})" if report.stop_detail else ""))
    if report.plan_problem:
        print(f"note: the plan step failed, so the raw requirement was used: {report.plan_problem}")
    if report.summary:
        print(f"model's summary: {report.summary}")
    if report.stopped_at:
        print(f"a person is needed at: {report.stopped_at}")
    print(f"{len(report.events)} action(s), {len(report.pages)} page type(s), {report.usage}, {report.elapsed_s}s")
    print("steps that worked:")
    print("\n".join(f"    {c}" for c in report.code) or "    (none)")
    print(f"report written to {output_path}")
    return exit_code(report)


def exit_code(report) -> int:
    """0 when the exploration ended normally (finished, handed to a person, or used its steps), else 1."""
    return 0 if report.stop_reason in ("done", "needs_person", "max_steps") else 1


def print_design(result: DesignResult) -> None:
    """The test cases as a reviewer reads them: one block per case, then the requirement points left uncovered."""
    for c in result.cases:
        covers = ", ".join(f"P{p}" for p in c.covers) or "-"
        print(f"{c.id} [{c.type}/{c.priority}] {c.title}   covers {covers}   ({c.grounding})")
        for i, s in enumerate(c.steps, 1):
            ref = f"  [E{s.event}]" if s.event is not None else ""
            print(f"     {i}. {s.description}{ref}")
        print(f"     expect: {c.expected_result}")
        for p in c.problems:
            print(f"     ? {p}")
    for g in result.gaps:
        print(f"P{g.point} not covered: {result.requirement_points[g.point - 1]} ({g.reason})")
    for d in result.dropped:
        print(f"dropped {d}")


def _design(args) -> int:
    try:
        report = ExplorationReport.model_validate_json(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"error: cannot read exploration report {args.report}: {str(e).splitlines()[0]}", file=sys.stderr)
        return 1
    try:
        llm = LLM(LLMConfig.from_env(args.provider, args.model))
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    result = design(llm, report)
    out = Path(args.out or str(Path(args.report).with_suffix("")) + ".cases.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    if result.problem:
        print(f"error: test design failed: {result.problem}", file=sys.stderr)
        return 1
    print_design(result)
    counts = {g: sum(c.grounding == g for c in result.cases) for g in ("observed", "partial", "unverified")}
    print(f"\n{len(result.cases)} test case(s): " + ", ".join(f"{n} {g}" for g, n in counts.items())
          + f"; {result.usage}\ntest cases written to {out}")
    return 0


def _print_verified(v: VerifiedCase) -> None:
    print(f"{v.case.id} {v.status.upper():<11} {clip(v.case.title, 70)}  ({v.elapsed_s}s)")
    if v.reason:
        print(f"     {v.reason}")


def _verify(args) -> int:
    try:
        cases = DesignResult.model_validate_json(Path(args.cases).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"error: cannot read test cases {args.cases}: {str(e).splitlines()[0]}", file=sys.stderr)
        return 1
    if args.case:
        wanted = {c.upper() for c in args.case}
        unknown = wanted - {c.id for c in cases.cases}
        if unknown:
            print(f"error: no such test case: {', '.join(sorted(unknown))} (the file has "
                  f"{', '.join(c.id for c in cases.cases) or 'none'})", file=sys.stderr)
            return 1
        cases = cases.model_copy(update={"cases": [c for c in cases.cases if c.id in wanted]})
    if not cases.cases:
        print("error: the file has no test cases to verify", file=sys.stderr)
        return 1
    try:
        cfg = LLMConfig.from_env(args.provider, args.model)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    def llm_for(case) -> LLM:  # one client per case: its own usage and call budget
        per_case = cfg.max_calls if cfg.max_calls is not None else len(case.steps) * 7 + 4
        return LLM(dataclasses.replace(cfg, max_calls=per_case))

    bcfg = BrowserConfig(headless=not args.headed)
    if args.cdp_url:
        bcfg.cdp_url = args.cdp_url
    print(f"verifying {len(cases.cases)} case(s) on {cases.url} with {cfg.provider.name}/{cfg.model}, "
          f"{args.concurrency} at a time\n")
    result = verify(cases, llm_for, bcfg, concurrency=args.concurrency, max_seconds=args.max_seconds,
                    on_case=_print_verified)
    out = Path(args.out or str(Path(args.cases)).removesuffix(".json").removesuffix(".cases") + ".verified.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    counts = result.counts()
    print(f"\n" + ", ".join(f"{n} {s}" for s, n in counts.items()) + f"; {result.usage}; {result.elapsed_s}s")
    print(f"verified cases written to {out}")
    return 0 if counts["verified"] else 1


def _report_output(args) -> str:
    if args.out:
        return args.out
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path("artifacts") / f"{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="test-pilot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ob = sub.add_parser("observe", help="open a URL and print what the model would see")
    ob.add_argument("url")
    ob.add_argument("--headed", action="store_true", help="show the browser window")
    ob.add_argument("--cdp-url", help="connect to a remote Chromium (default: env TEST_PILOT_CDP_URL, else launch locally)")
    ob.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ob.add_argument("--full", action="store_true", help="list the whole page, not just the part near the viewport")
    ob.add_argument("--json", action="store_true", help="print the full Observation as JSON")
    ob.add_argument("--locators", action="store_true", help="also print a verified, unique locator for each control")

    lm = sub.add_parser("llm", help="check the LLM configuration with one tiny call")
    lm.add_argument("--provider", choices=sorted(PROVIDERS), help="default: env LLM_PROVIDER, else gemini")
    lm.add_argument("--model", help="default: env LLM_MODEL, else the provider's default")
    lm.add_argument("--models", action="store_true", help="list the model ids this key can use")

    ex = sub.add_parser("explore", help="explore a web app against a requirement and write an evidence report")
    ex.add_argument("url")
    ex.add_argument("requirement", help='for example "Search for a product under $50"')
    ex.add_argument("--provider", choices=sorted(PROVIDERS), help="default: env LLM_PROVIDER, else gemini")
    ex.add_argument("--model")
    ex.add_argument("--max-steps", type=int, default=25)
    ex.add_argument("--max-seconds", type=float, default=300)
    ex.add_argument("--out", default=None, help="optional report path (default: artifacts/<UTC timestamp>.json)")
    ex.add_argument("--headed", action="store_true", help="show the browser window")
    ex.add_argument("--cdp-url", help="connect to a remote Chromium")
    ex.add_argument("-q", "--quiet", action="store_true", help="do not print each step")

    de = sub.add_parser("design", help="design structured test cases from an exploration report")
    de.add_argument("report", help="the JSON report written by `explore`")
    de.add_argument("--provider", choices=sorted(PROVIDERS), help="default: env LLM_PROVIDER, else gemini")
    de.add_argument("--model")
    de.add_argument("--out", default=None, help="default: <report>.cases.json next to the report")

    ve = sub.add_parser("verify", help="carry out each test case in its own browser and keep what held")
    ve.add_argument("cases", help="the JSON file written by `design`")
    ve.add_argument("--case", action="append", metavar="TC-NN", help="verify only this case (repeatable)")
    ve.add_argument("-c", "--concurrency", type=_positive, default=1, help="cases verified at once (default 1)")
    ve.add_argument("--provider", choices=sorted(PROVIDERS), help="default: env LLM_PROVIDER, else gemini")
    ve.add_argument("--model")
    ve.add_argument("--max-seconds", type=float, default=DEFAULT_CASE_SECONDS, help="time limit per case")
    ve.add_argument("--out", default=None, help="default: <report>.verified.json next to the cases")
    ve.add_argument("--headed", action="store_true", help="show the browser windows (one per running case)")
    ve.add_argument("--cdp-url", help="connect to a remote Chromium (each case gets its own context)")

    load_dotenv()
    args = ap.parse_args(argv)
    return {"observe": _observe, "llm": _llm, "explore": _explore, "design": _design,
            "verify": _verify}[args.cmd](args)


def _positive(raw: str) -> int:
    try:
        n = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {raw!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


if __name__ == "__main__":
    raise SystemExit(main())