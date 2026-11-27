"""The designer's own checks: what the model proposes is tied back to the recorded evidence, never taken on trust."""
import json

from test_pilot.actions import Step
from test_pilot.analyzer import Plan
from test_pilot.cli import main as cli_main
from test_pilot.designer import DesignReply, TestCase, build_messages, design, finalize
from test_pilot.evidence import Changes
from test_pilot.explorer import Event, ExplorationReport, FieldInfo, PageInfo
from test_pilot.llm import LLMError
from test_pilot.models import Locator


def _report() -> ExplorationReport:
    city = Step(kind="type", locator=Locator(kind="label", value="City"), value="Pune", element_name="City")
    search = Step(kind="click", locator=Locator(kind="role", role="button", value="Search"), element_name="Search")
    events = [
        Event(n=1, url="https://x.test/", page="home", action={"kind": "click", "id": 9}, status="ok",
              message='clicked [9] button "Search" -> page changed', step=search,
              changes=Changes(errors=["City: Please fill out this field."])),
        Event(n=2, url="https://x.test/", page="home", action={"kind": "type", "id": 7, "text": "Pune"}, status="ok",
              message='typed "Pune" into [7] textbox "City"', step=city),
        Event(n=3, url="https://x.test/", page="home", action={"kind": "click", "id": 9}, status="ok",
              message='clicked [9] button "Search" -> page changed', step=search,
              changes=Changes(navigated=True, url="https://x.test/results?city=Pune", appeared=["3 trips found"])),
        Event(n=4, url="https://x.test/results", page="results", action={"kind": "click", "id": 44}, status="failed",
              message='[44] clickable "Book" is no longer on the page; look again'),
    ]
    pages = [PageInfo(signature="home", url="https://x.test/", title="Trips", headings=["Find a trip"],
                      fields=[FieldInfo(name="City", widget="textbox", required=True, values_tried=["Pune"])],
                      buttons=["Search"])]
    return ExplorationReport(url="https://x.test/", requirement="search for trips from Pune",
                             plan=Plan(requirement_points=["search trips from a city", "see the results",
                                                           "book a trip"]),
                             events=events, pages=pages, stop_reason="done")


def _case(**kw) -> dict:
    base = {"title": "Search with a valid city", "module": "Home", "covers": ["P1", "P2"], "type": "positive",
            "priority": "high", "steps": [{"description": "Type Pune into City", "event": "E2"},
                                          {"description": "Click Search", "event": 3}],
            "expected_result": "Results are listed",
            "expected": [{"kind": "text_visible", "value": "3 trips found", "evidence": ["E3"]},
                         {"kind": "url_contains", "value": "/results"}]}
    return {**base, **kw}


def _finalize(*cases, unsupported=()):
    return finalize(DesignReply.model_validate({"cases": list(cases), "unsupported": list(unsupported)}), _report())


def test_a_case_built_from_recorded_steps_and_seen_outcomes_is_observed():
    r = _finalize(_case())
    c = r.cases[0]
    assert c.id == "TC-01" and c.type == "Positive" and c.priority == "High" and c.covers == [1, 2]
    assert c.grounding == "observed" and c.problems == []
    assert c.steps[0].locator == 'page.get_by_label("City", exact=True)' and c.steps[0].value == "Pune"
    assert [x.evidence for x in c.expected] == [[3], [3]]                    # the URL check found its own source


def test_an_outcome_nobody_saw_is_kept_but_marked_as_an_inference():
    r = _finalize(_case(title="Empty city", type="Validation", covers=[1],
                        steps=[{"description": "Click Search", "event": 1}],
                        expected=[{"kind": "field_error", "value": "Please fill out this field", "evidence": []},
                                  {"kind": "field_error", "value": "City must be at least 3 letters"}]))
    c = r.cases[0]
    assert c.grounding == "partial"
    assert [x.observed for x in c.expected] == [True, False]
    assert c.expected[0].evidence == [1]                                      # cited nothing; the code found E1
    assert any("City must be at least 3 letters" in p and "never observed" in p for p in c.problems)


def test_steps_that_cite_failed_or_missing_events_or_unseen_controls_are_flagged():
    r = _finalize(_case(steps=[{"description": "Book the trip", "event": 4},
                               {"description": "Open filters", "event": 99},
                               {"description": "Click Apply coupon", "target": "Apply coupon"},
                               {"description": "Click Search", "target": "Search"}],
                        expected=[{"kind": "text_visible", "value": "Coupon applied"}]))
    c = r.cases[0]
    assert c.grounding == "partial"                                # only because "Search" is the recorded E1 click
    joined = " | ".join(c.problems)
    assert "E4, which did not succeed (failed)" in joined and "E99, which does not exist" in joined
    assert '"Apply coupon", which was not seen' in joined and '"Search"' not in joined   # Search is a known control
    assert [s.event for s in c.steps] == [None, None, None, 1]    # a target naming a recorded control replays it
    assert len(c.problems) == 4                                    # one per bad step, plus the unseen outcome


def test_duplicates_and_cases_over_the_limit_are_dropped_and_ids_are_sequential():
    many = [_case(title=f"variant {i}", steps=[{"description": "Type", "target": "City", "value": f"Pune{i}"}])
            for i in range(12)]
    r = _finalize(_case(), _case(title="same again"), *many)
    assert len(r.cases) == 10 and [c.id for c in r.cases] == [f"TC-{i:02d}" for i in range(1, 11)]
    assert any("same again" in d and "duplicate" in d for d in r.dropped)
    assert sum("over the limit" in d for d in r.dropped) == 3


def test_uncovered_requirement_points_become_gaps_with_the_models_reason():
    r = _finalize(_case(covers=[1, 7]), unsupported=[{"point": "P3", "reason": "the explorer stopped before booking"}])
    assert [(g.point, g.reason) for g in r.gaps] == [(2, "no test case was designed for it"),
                                                     (3, "the explorer stopped before booking")]
    assert r.cases[0].covers == [1] and any("P7" in p for p in r.cases[0].problems)


def test_the_prompt_carries_the_evidence_without_live_element_ids():
    text = build_messages(_report())[1]["content"]
    assert "P1. search trips from a city" in text and "E3 ok" in text and '+ "3 trips found"' in text
    assert 'error: "City: Please fill out this field."' in text and "E4 FAILED" in text
    assert "[9]" not in text and "[44]" not in text
    assert 'field "City" textbox required tried: "Pune"' in text


class _FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.usage = type("U", (), {"summary": lambda s: "1 call(s)"})()

    def structured(self, messages, schema, max_tokens=None):
        self.calls.append(messages)
        self.max_tokens = max_tokens
        if isinstance(self.reply, Exception):
            raise self.reply
        return schema.model_validate(self.reply)


def test_design_runs_one_call_and_a_failed_call_is_reported_not_raised():
    llm = _FakeLLM({"cases": [_case()]})
    r = design(llm, _report())
    assert len(llm.calls) == 1 and r.cases[0].grounding == "observed" and r.usage == "1 call(s)"
    assert llm.max_tokens == 16000                                    # room for ten cases, not one decision
    failed = design(_FakeLLM(LLMError("quota")), _report())
    assert failed.problem == "quota" and failed.cases == [] and len(failed.gaps) == 3


def test_cli_design_reads_a_report_and_writes_cases(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.json"
    _report().save(str(path))
    monkeypatch.setattr("test_pilot.cli.LLM", lambda cfg: _FakeLLM({"cases": [_case()]}))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert cli_main(["design", str(path), "--provider", "gemini"]) == 0
    out = json.loads((tmp_path / "run.cases.json").read_text())
    assert out["cases"][0]["id"] == "TC-01" and out["gaps"][0]["point"] == 3
    shown = capsys.readouterr().out
    assert "TC-01 [Positive/High] Search with a valid city" in shown and "P3 not covered" in shown


def test_cli_design_reports_an_unreadable_report(tmp_path, capsys):
    assert cli_main(["design", str(tmp_path / "missing.json")]) == 1
    assert "cannot read exploration report" in capsys.readouterr().err


def test_test_case_schema_is_exportable():
    schema = TestCase.model_json_schema()
    assert {"id", "title", "module", "type", "priority", "steps", "expected_result", "grounding"} <= set(schema["properties"])
