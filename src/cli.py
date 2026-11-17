"""python -m test_pilot.cli observe <url>  : print the page exactly as the model will see it (no LLM involved)."""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from .browser import BrowserConfig, BrowserSession, NavigationError
from .locators import verify_all
from .render import DEFAULT_MAX_CHARS, render


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
    load_dotenv()
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
