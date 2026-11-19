"""test-pilot observe <url>             : print the page exactly as the model will see it (no LLM involved).
test-pilot llm                         : check the LLM configuration with one tiny call.
test-pilot explore <url> "<requirement>": explore the app with the model and write an evidence report."""
from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv
from pydantic import BaseModel

from .browser import BrowserConfig, BrowserSession, NavigationError
from .explorer import Event, explore
from .llm import LLM, PROVIDERS, LLMConfig, LLMError
from .locators import verify_all
from .render import DEFAULT_MAX_CHARS, render


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
            print(obs.model_dump_json(indent=2) if args.json else render(obs, args.max_chars))
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
        if cfg.max_calls is None:  # a safety net: one decision per step, plus room for repairs and the plan
            cfg.max_calls = args.max_steps * 2 + 4
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
    report.save(args.out)
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
    print(f"report written to {args.out}")
    return 0 if report.stop_reason in ("done", "needs_person", "max_steps") else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="test-pilot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ob = sub.add_parser("observe", help="open a URL and print what the model would see")
    ob.add_argument("url")
    ob.add_argument("--headed", action="store_true", help="show the browser window")
    ob.add_argument("--cdp-url", help="connect to a remote Chromium (default: env TEST_PILOT_CDP_URL, else launch locally)")
    ob.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ob.add_argument("--json", action="store_true", help="print the full Observation as JSON")
    ob.add_argument("--locators", action="store_true", help="also print a verified, unique locator for each control")

    lm = sub.add_parser("llm", help="check the LLM configuration with one tiny call")
    lm.add_argument("--provider", choices=sorted(PROVIDERS), help="default: env LLM_PROVIDER, else gemini")
    lm.add_argument("--model", help="default: env LLM_MODEL, else the provider's default")
    lm.add_argument("--models", action="store_true", help="list the model ids this key can use")

    ex = sub.add_parser("explore", help="explore a web app against a requirement and write an evidence report")
    ex.add_argument("url")
    ex.add_argument("requirement", help='for example "Book a flight from Pune to Delhi"')
    ex.add_argument("--provider", choices=sorted(PROVIDERS), help="default: env LLM_PROVIDER, else gemini")
    ex.add_argument("--model")
    ex.add_argument("--max-steps", type=int, default=25)
    ex.add_argument("--max-seconds", type=float, default=300)
    ex.add_argument("--out", default="exploration.json", help="where to write the report (default: exploration.json)")
    ex.add_argument("--headed", action="store_true", help="show the browser window")
    ex.add_argument("--cdp-url", help="connect to a remote Chromium")
    ex.add_argument("-q", "--quiet", action="store_true", help="do not print each step")

    load_dotenv()
    args = ap.parse_args(argv)
    return {"observe": _observe, "llm": _llm, "explore": _explore}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())