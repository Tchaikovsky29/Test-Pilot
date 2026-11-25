"""Turn-by-turn browser and model trace.

Run:
  conda run -n tp python -m tests.trace_explorer https://phptravels.net "book a flight" --provider openrouter --headed

No source code is modified by this script. It records every model prompt/reply,
page snapshot, element ID, browser diff, and replayable step under artifacts/traces/<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from test_pilot import BrowserConfig, BrowserSession
from test_pilot.explorer import explore
from test_pilot.llm import LLM, LLMConfig
from test_pilot.render import render


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, default=lambda o: o.model_dump(mode="json") if hasattr(o, "model_dump") else str(o), indent=2), encoding="utf-8")


def text_write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class TraceLLM:
    def __init__(self, llm: LLM, session: BrowserSession, turn_dir: Path):
        self.llm = llm
        self.session = session
        self.turn_dir = turn_dir
        self.usage = llm.usage
        self.calls = 0

    def structured(self, messages, schema):
        self.calls += 1

        # Capture the live browser state before the model answer and link it to this exact request.
        obs = self.session.observe()
        snapshot = {
            "url": obs.url,
            "title": obs.title,
            "rendered_page": render(obs, max_chars=20000, window=False),
            "elements": [
                {
                    "id": e.id,
                    "role": e.role,
                    "widget": e.widget,
                    "name": e.display_name,
                    "raw_name": e.name,
                    "value": e.value,
                    "states": e.states,
                    "locators": [l.model_dump(mode="json") for l in e.locators],
                }
                for e in obs.elements
            ],
        }

        turn_dir = self.turn_dir / f"turn-{self.calls:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        json_write(turn_dir / "messages.json", messages)
        json_write(turn_dir / "live_snapshot.json", snapshot)
        text_write(turn_dir / "page.txt", snapshot["rendered_page"])

        print("\n" + "=" * 100)
        print(f"MODEL TURN {self.calls} — {schema.__name__}")
        print("=" * 100)
        for message in messages:
            print(f"\n[{message['role'].upper()}]\n{message['content']}\n")

        reply = self.llm.structured(messages, schema)
        json_write(turn_dir / "reply.json", reply)
        text_write(turn_dir / "reply.txt", reply.model_dump_json(indent=2))

        print("-" * 100)
        print("MODEL REPLY")
        print("-" * 100)
        print(reply.model_dump_json(indent=2))
        print("=" * 100)
        return reply


class Tee:
    """Mirror terminal text to a complete trace file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.write_text("", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("requirement")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()

    trace_root = Path("artifacts/traces") / utc_stamp()
    trace_root.mkdir(parents=True, exist_ok=True)
    tee = Tee(trace_root / "trace.txt")
    sys.stdout = tee

    config = LLMConfig.from_env(args.provider, args.model)
    llm = LLM(config)

    event_log: list[dict] = []

    def on_event(event):
        item = event.model_dump(mode="json")
        event_log.append(item)
        json_write(trace_root / "events.json", event_log)
        print("\n" + "-" * 100)
        print(f"ACTION {event.n}: {event.note}")
        print(f"status: {event.status}")
        print(f"message: {event.message}")
        if event.code:
            print(f"playwright: {event.code}")
        if event.changes:
            print("changes:")
            for key in ("appeared", "disappeared", "alerts", "errors"):
                values = getattr(event.changes, key)
                if values:
                    print(f"  {key}: {values}")

    with BrowserSession(BrowserConfig(headless=not args.headed)) as session:
        traced = TraceLLM(llm, session, trace_root)
        report = explore(session, traced, args.url, args.requirement,
                         max_steps=args.max_steps, max_seconds=args.max_seconds,
                         on_event=on_event)

    report_path = trace_root / "exploration.json"
    report.save(str(report_path))

    print("\n" + "=" * 100)
    print("TRACE COMPLETE")
    print("=" * 100)
    print(f"stop_reason: {report.stop_reason}")
    print(f"stop_detail: {report.stop_detail}")
    print(f"report: {report_path}")
    print(f"turns: {trace_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
