"""Verifiers carry each case out in a browser of their own and keep only what held."""
import json
import re
import threading

import pytest

from test_pilot import BrowserConfig
from test_pilot.actions import Step
from test_pilot.cli import main as cli_main
from test_pilot.designer import CaseStep, DesignResult, Expectation, TestCase
from test_pilot.llm import LLMError
from test_pilot.models import Locator
from test_pilot.verifier import StepDecision, verify, verify_case

CITY = Step(kind="type", locator=Locator(kind="role", role="textbox", value="City"), value="Pune",
            element_name="City", widget="textbox")
SEARCH = Step(kind="click", locator=Locator(kind="role", role="button", value="Search"),
              element_name="Search", widget="button")


@pytest.fixture(scope="module", autouse=True)
def _browser_available():
    """Skip when Chromium is not installed. The check's browser is closed again at once: the sync API allows one
    running Playwright per thread, and verify_case starts its own."""
    from test_pilot import BrowserSession
    try:
        BrowserSession(BrowserConfig()).start().close()
    except Exception as e:
        pytest.skip(f"browser unavailable: {e}")


def _case(cid="TC-01", steps=None, expected=None, title="Search for a city") -> TestCase:
    return TestCase(id=cid, title=title, module="Find trips", type="Positive", priority="High",
                    steps=steps or [CaseStep(description="Type Pune into City", event=2, value="Pune", recorded=CITY),
                                    CaseStep(description="Click Search", event=3, recorded=SEARCH)],
                    expected_result="the search starts",
                    expected=expected or [Expectation(kind="text_visible", value="Looking for trips to Pune", observed=True)])


class Model:
    """A fake LLM for the agent steps. Each entry: a dict, an Exception, or fn(page_text) -> dict."""

    def __init__(self, *entries):
        self.entries, self.prompts = list(entries), []
        self.usage = type("U", (), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0})()

    def structured(self, messages, schema):
        self.prompts.append(messages)
        self.usage.calls += 1
        if not self.entries:
            raise AssertionError("the model was asked more than the test expected")
        entry = self.entries.pop(0)
        if isinstance(entry, Exception):
            raise entry
        if callable(entry):
            entry = entry(messages[-1]["content"])
        return schema.model_validate(entry)


def click(name):
    def pick(page):
        m = re.search(r"\[(\d+)\] \w+ " + re.escape(json.dumps(name)), page)
        assert m, f"{name} not on the page"
        return {"note": f"click {name}", "action": {"kind": "click", "id": int(m.group(1))}}
    return pick


def test_recorded_steps_are_replayed_without_the_model(base_url):
    llm = Model()  # any call fails the test
    v = verify_case(_case(), base_url + "/form.html", llm, BrowserConfig())
    assert v.status == "verified", v.reason
    assert [(s.how, s.status) for s in v.steps] == [("replayed", "done"), ("replayed", "done")]
    assert [s.kind for s in v.definition] == ["type", "click"] and v.definition[0].value == "Pune"
    assert [a.kind for a in v.assertions] == ["text_visible"] and llm.prompts == []


def test_a_guessed_validation_message_is_corrected_to_the_browsers_own(base_url):
    case = _case(steps=[CaseStep(description="Click Search with City empty", event=1, recorded=SEARCH)],
                 expected=[Expectation(kind="field_error", value="City is required")])  # a guess: never observed
    v = verify_case(case, base_url + "/form.html", Model(), BrowserConfig())
    assert v.status == "verified", v.reason
    x, a = v.expectations[0], v.assertions[0]
    assert x.status == "corrected" and "fill" in x.actual.lower() and "corrected" in v.reason
    assert a.native and a.locator is not None and "City" in a.locator.value


def test_an_observed_message_that_does_not_appear_is_not_rewritten(base_url):
    case = _case(steps=[CaseStep(description="Click Search with City empty", event=1, recorded=SEARCH)],
                 expected=[Expectation(kind="field_error", value="City is required", observed=True)])
    v = verify_case(case, base_url + "/form.html", Model(), BrowserConfig())
    assert v.status == "unconfirmed" and v.expectations[0].status == "not_seen"
    assert "City" in v.expectations[0].actual                        # what the page showed instead


def test_a_step_without_a_recording_is_done_by_the_model(base_url):
    case = _case(steps=[CaseStep(description="Click Show error", target="Show error")],
                 expected=[Expectation(kind="alert", value="Could not find any trips")])
    llm = Model(click("Show error"), {"done": True, "note": "the error is shown"})
    v = verify_case(case, base_url + "/form.html", llm, BrowserConfig())
    assert v.status == "verified", v.reason
    assert v.steps[0].how == "agent" and v.steps[0].actions == 1 and v.llm_calls == 2
    assert [s.element_name for s in v.definition] == ["Show error"]
    assert "→ 1. Click Show error" in llm.prompts[0][1]["content"]   # the model is told which step it is on


def test_a_step_the_model_cannot_do_fails_the_case_with_the_reason(base_url):
    case = _case(steps=[CaseStep(description="Apply the coupon", target="Apply coupon")])
    v = verify_case(case, base_url + "/form.html", Model({"failed": True, "reason": "there is no coupon field"}),
                    BrowserConfig())
    assert v.status == "failed" and "step 1" in v.reason and "no coupon field" in v.reason
    assert v.expectations == [] and v.assertions == []


def test_a_model_error_fails_the_case_instead_of_raising(base_url):
    case = _case(steps=[CaseStep(description="Click Show error", target="Show error")])
    v = verify_case(case, base_url + "/form.html", Model(LLMError("quota")), BrowserConfig())
    assert v.status == "failed" and "model error: quota" in v.reason


def test_an_outcome_that_never_shows_leaves_the_case_unconfirmed(base_url):
    case = _case(expected=[Expectation(kind="text_visible", value="Booking confirmed")])
    v = verify_case(case, base_url + "/form.html", Model(), BrowserConfig())
    assert v.status == "unconfirmed" and "Booking confirmed" in v.reason and "Searching" in v.reason
    assert v.assertions == [] and len(v.definition) == 2                 # the steps still worked


def test_an_unreachable_site_blocks_the_case():
    v = verify_case(_case(), "http://127.0.0.1:9/", Model(), BrowserConfig(nav_timeout_ms=3000))
    assert v.status == "blocked" and "Could not open" in v.reason


def test_cases_run_in_parallel_each_in_its_own_browser(base_url):
    design = DesignResult(url=base_url + "/form.html", requirement="r", requirement_points=["p"],
                          cases=[_case(f"TC-0{i}") for i in (3, 1, 2)])
    threads = set()

    def factory(case):
        threads.add(threading.get_ident())
        if case.id == "TC-02":
            raise RuntimeError("browser crashed")
        return Model()

    done = []
    r = verify(design, factory, BrowserConfig(), concurrency=2, on_case=done.append)
    assert [c.case.id for c in r.cases] == ["TC-01", "TC-02", "TC-03"] and len(done) == 3
    assert [c.status for c in r.cases] == ["verified", "failed", "verified"]
    assert "verifier crashed: browser crashed" in r.cases[1].reason
    assert len(threads) == 2                                               # two workers, each with its own browser
    assert r.concurrency == 2 and r.counts() == {"verified": 2, "unconfirmed": 0, "failed": 1, "blocked": 0}


def test_step_decision_accepts_the_common_shapes():
    assert StepDecision.model_validate({"kind": "click", "id": 3}).action.id == 3
    assert StepDecision.model_validate({"failed": True, "reason": "x", "action": {"kind": "click", "id": 1}}).action is None
    with pytest.raises(ValueError):
        StepDecision.model_validate({"note": "thinking"})


def test_cli_verify_writes_results_and_rejects_a_bad_concurrency(base_url, tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.cases.json"
    path.write_text(DesignResult(url=base_url + "/form.html", requirement="r", requirement_points=["p"],
                                 cases=[_case()]).model_dump_json())
    monkeypatch.setattr("test_pilot.cli.LLM", lambda cfg: Model())
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert cli_main(["verify", str(path), "--provider", "gemini", "-c", "2"]) == 0
    out = json.loads((tmp_path / "run.verified.json").read_text())
    assert out["cases"][0]["status"] == "verified" and out["concurrency"] == 2
    assert "TC-01 VERIFIED" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli_main(["verify", str(path), "-c", "0"])


SHOW_ERROR = Step(kind="click", locator=Locator(kind="role", role="button", value="Show error"),
                  element_name="Show error", widget="button")


def test_a_guessed_field_error_reported_as_a_page_alert_is_corrected_to_the_alert(base_url):
    case = _case(steps=[CaseStep(description="Click Show error", event=1, recorded=SHOW_ERROR)],
                 expected=[Expectation(kind="field_error", value="Arrival To")])   # a field name, not a message
    v = verify_case(case, base_url + "/form.html", Model(), BrowserConfig())
    assert v.status == "verified", v.reason
    assert v.expectations[0].status == "corrected" and v.expectations[0].actual == "Could not find any trips"
    assert v.assertions[0].kind == "alert" and v.assertions[0].value == "Could not find any trips"


def test_a_value_held_by_a_field_counts_as_seen(base_url):
    case = _case(steps=[CaseStep(description="Type Pune into City", event=2, value="Pune", recorded=CITY)],
                 expected=[Expectation(kind="text_visible", value="Pune")])
    v = verify_case(case, base_url + "/form.html", Model(), BrowserConfig())
    assert v.status == "verified", v.reason
    a = v.assertions[0]
    assert a.kind == "field_value" and a.value == "Pune" and a.locator is not None


def test_cli_verify_runs_only_the_chosen_case_one_at_a_time(base_url, tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.cases.json"
    path.write_text(DesignResult(url=base_url + "/form.html", requirement="r", requirement_points=["p"],
                                 cases=[_case("TC-01"), _case("TC-02")]).model_dump_json())
    monkeypatch.setattr("test_pilot.cli.LLM", lambda cfg: Model())
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert cli_main(["verify", str(path), "--provider", "gemini", "--case", "tc-02"]) == 0
    out = json.loads((tmp_path / "run.verified.json").read_text())
    assert [c["case"]["id"] for c in out["cases"]] == ["TC-02"] and out["concurrency"] == 1
    assert cli_main(["verify", str(path), "--provider", "gemini", "--case", "TC-09"]) == 1
    assert "no such test case: TC-09 (the file has TC-01, TC-02)" in capsys.readouterr().err


def test_guessed_field_errors_shown_as_a_plain_toast_are_corrected_once(base_url):
    go = Step(kind="click", locator=Locator(kind="role", role="button", value="Search Flights"),
              element_name="Search Flights", widget="button")
    case = _case(steps=[CaseStep(description="Click Search Flights with nothing chosen", event=1, recorded=go)],
                 expected=[Expectation(kind="field_error", value="Departure From"),
                           Expectation(kind="field_error", value="Arrival To")])
    v = verify_case(case, base_url + "/toast.html", Model(), BrowserConfig())
    assert v.status == "verified", v.reason
    assert [x.status for x in v.expectations] == ["corrected", "corrected"]
    assert [(a.kind, a.value) for a in v.assertions] == [("text_visible", "Please select both departure and arrival airports!")]
