"""Cases prove business outcomes: saved values under their labels, the right message when nothing is saved, and every
case can run on its own. Checked on a CRM-shaped fixture (a pop-up form over a backdrop, a toast, a record page)."""
import pytest

from test_pilot import BrowserConfig, BrowserSession
from test_pilot.actions import Step
from test_pilot.analyzer import Plan
from test_pilot.designer import CaseStep, DesignReply, Expectation, TestCase, finalize
from test_pilot.explorer import ExplorationReport, PageInfo, summarize_page
from test_pilot.models import Line, Locator, Observation
from test_pilot.text import shows
from test_pilot.verifier import verify_case


# ---- reading a record page ----------------------------------------------------------------------------------------

def test_a_label_followed_by_its_value_is_read_in_any_layout():
    record = ["John Anderson", "Name", "John Anderson", "Company", "TechNova Solutions", "Email", "j@example.com"]
    assert shows(record, "Company", "TechNova Solutions") and shows(record, "email", "j@example.com")
    assert not shows(record, "Company", "j@example.com")                 # a value under another label
    assert shows(["Name John Anderson Company TechNova Solutions"], "Company", "TechNova Solutions")   # joined
    assert shows(["Company: TechNova Solutions"], "Company", "TechNova Solutions")
    assert not shows(record, "", "x") and not shows(record, "Phone", "")


def test_a_page_summary_keeps_what_the_page_said():
    obs = Observation(url="https://x.test/lead/1", lines=[Line(kind="heading", text="John Anderson"),
                                                          Line(kind="text", text="Company"),
                                                          Line(kind="text", text="TechNova Solutions")])
    assert summarize_page(obs).texts == ["John Anderson", "Company", "TechNova Solutions"]


# ---- the designer holds cases to the bar --------------------------------------------------------------------------

def _report():
    return ExplorationReport(url="https://crm.test/", requirement="create a lead",
                             plan=Plan(requirement_points=["create a lead", "check the saved values"]),
                             pages=[PageInfo(signature="lead", url="https://crm.test/lead/1",
                                             texts=["Name", "John Anderson", "Company", "TechNova Solutions"],
                                             buttons=["New", "Save"])])


def _design(**kw):
    case = {"title": "Create a lead", "module": "Leads", "covers": [1, 2], "type": "Positive", "priority": "High",
            "steps": [{"description": "Click New", "target": "New"}, {"description": "Click Save", "target": "Save"}],
            "expected_result": "saved", **kw}
    return finalize(DesignReply.model_validate({"cases": [case]}), _report()).cases[0]


def test_saved_values_seen_under_their_labels_ground_a_case():
    c = _design(expected=[{"kind": "field_shows", "label": "Company", "value": "TechNova Solutions"}])
    assert c.grounding == "observed", c.problems
    assert any("shows" in s for s in c.evidence)
    c = _design(expected=[{"kind": "field_shows", "label": "Company", "value": "Globex"}])
    assert c.grounding != "observed" and any("never observed" in p for p in c.problems)


def test_a_positive_case_that_only_reaches_a_screen_is_not_enough():
    c = _design(expected=[{"kind": "element_visible", "value": "Save"}])
    assert c.grounding != "observed" and any("no business outcome" in p for p in c.problems)
    neg = _design(type="Navigation", expected=[{"kind": "element_visible", "value": "Save"}])
    assert not any("no business outcome" in p for p in neg.problems)       # a navigation case may check a screen


def test_a_case_that_needs_existing_data_must_create_it_itself():
    expected = [{"kind": "field_shows", "label": "Company", "value": "TechNova Solutions"}]
    c = _design(preconditions=["An Account exists in the org"], expected=expected)
    assert any("not created by the case's own setup steps" in p for p in c.problems) and c.grounding != "observed"
    c = _design(preconditions=["An Account exists in the org"], expected=expected,
                steps=[{"description": "Create the Account", "target": "New", "setup": True},
                       {"description": "Click Save", "target": "Save"}])
    assert c.grounding == "observed", c.problems


# ---- the verifier checks the saved record live --------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _browser_available():
    try:
        BrowserSession(BrowserConfig()).start().close()        # closed at once: one Playwright per thread
    except Exception as e:
        pytest.skip(f"browser unavailable: {e}")


def _step(kind, role, name, value=None):
    return Step(kind=kind, locator=Locator(kind="role", role=role, value=name), value=value, element_name=name, widget=role)


NEW, SAVE = _step("click", "button", "New"), _step("click", "button", "Save")


def _create(last="Anderson", company="TechNova Solutions", email="john.anderson.qa@example.com"):
    steps = [CaseStep(description="Click New", recorded=NEW)]
    for name, value in (("* Last Name", last), ("* Company", company), ("Email", email)):
        if value:
            steps.append(CaseStep(description=f"Enter {name}", value=value, recorded=_step("type", "textbox", name, value)))
    steps.append(CaseStep(description="Click Save", recorded=SAVE))
    return steps


class NoModel:
    usage = type("U", (), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0})()

    def structured(self, *a, **kw):
        raise AssertionError("recorded steps should not need the model")


def test_a_created_record_is_verified_by_its_saved_values(base_url):
    case = TestCase(id="TC-01", title="Create a Lead", module="Leads", type="Positive", priority="High",
                    steps=_create(), expected_result="the lead is saved with the entered values",
                    expected=[Expectation(kind="field_shows", label="Company", value="TechNova Solutions"),
                              Expectation(kind="field_shows", label="Email", value="john.anderson.qa@example.com"),
                              Expectation(kind="alert", value="was created", observed=True)])
    v = verify_case(case, base_url + "/crm.html", NoModel(), BrowserConfig())
    assert v.status == "verified", v.reason
    assert [(a.kind, a.label) for a in v.assertions][:2] == [("field_shows", "Company"), ("field_shows", "Email")]
    assert v.notices == []


def test_a_missing_required_field_saves_nothing_and_shows_the_real_message(base_url):
    case = TestCase(id="TC-02", title="Lead needs a Last Name", module="Leads", type="Negative", priority="High",
                    steps=_create(last=""), expected_result="not saved; the missing field is reported",
                    expected=[Expectation(kind="field_error", value="Last Name is mandatory"),   # a guess
                              Expectation(kind="text_visible", value="We hit a snag")])
    v = verify_case(case, base_url + "/crm.html", NoModel(), BrowserConfig())
    assert v.status == "verified", v.reason
    assert v.expectations[0].status == "corrected" and v.expectations[0].actual == "Complete this field."
    assert v.assertions[0].locator is not None and "Last Name" in v.assertions[0].locator.value


def test_the_sites_duplicate_warning_is_a_notice_not_a_failure(base_url):
    again = [CaseStep(description="Back to Leads", recorded=_step("click", "link", "Leads"))]
    case = TestCase(id="TC-03", title="Create the same lead twice", module="Leads", type="Positive", priority="Low",
                    steps=_create() + again + _create(), expected_result="saved",
                    expected=[Expectation(kind="field_shows", label="Company", value="TechNova Solutions")])
    v = verify_case(case, base_url + "/crm.html", NoModel(), BrowserConfig())
    assert v.status == "verified", v.reason
    assert v.notices and "duplicate" in v.notices[0]


def test_what_a_form_holds_is_not_read_as_a_saved_value():
    from test_pilot.models import Element
    box = Element(id=1, role="textbox", widget="textbox", name="First Name", value="John")
    link = Element(id=2, role="link", widget="link", name="john@example.com")
    obs = Observation(url="https://x.test/", elements=[box, link],
                      lines=[Line(kind="control", element_id=1), Line(kind="text", text="Email"),
                             Line(kind="control", element_id=2)])
    rd = obs.readable()
    assert rd == ["Email", "john@example.com"]                      # the edit box is input, the link is display
    assert not shows(rd, "First Name", "John") and shows(rd, "Email", "john@example.com")


def test_a_negative_case_must_assert_the_message_not_just_the_url():
    url_only = [{"kind": "url_contains", "value": "lightning/o/Lead/new"}]     # TC-02 as first designed
    c = _design(type="Negative", expected=url_only)
    assert c.grounding != "observed" and any("no validation message" in p for p in c.problems)
    nav = _design(type="Navigation", expected=[{"kind": "element_visible", "value": "Save"}])
    assert nav.grounding == "observed", nav.problems                          # navigation may check a screen


def test_any_problem_keeps_a_case_from_counting_as_observed():
    c = _design(steps=[{"description": "Click New", "event": 0, "target": "New"},      # TC-06: a step that doesn't exist
                       {"description": "Click Save", "target": "Save"}],
                expected=[{"kind": "field_shows", "label": "Company", "value": "TechNova Solutions"}])
    assert any("E0, which does not exist" in p for p in c.problems)
    assert c.grounding == "partial"                                            # was "observed" before


# ---- what the second Salesforce verify run showed -----------------------------------------------------------------

def test_a_fields_message_drops_hidden_text_repeating_its_label():
    from test_pilot.evidence import _without_name
    assert _without_name("Last Name Complete this field.", "Last Name") == "Complete this field."
    assert _without_name("* Company: Complete this field.", "Company") == "Complete this field."
    assert _without_name("Complete this field.", "Last Name") == "Complete this field."
    assert _without_name("Last Name", "Last Name") == "Last Name"           # nothing else: keep it


def _report_with_events():
    from test_pilot.explorer import Event
    r = _report()
    for n, (step, msg) in enumerate([(NEW, 'clicked "New"'),
                                     (_step("type", "textbox", "* Last Name", "Anderson"), 'typed into "* Last Name"'),
                                     (SAVE, 'clicked "Save"')], 1):
        r.events.append(Event(n=n, url="https://crm.test/", page="lead", action={"kind": step.kind}, status="ok",
                              message=msg, step=step))
    return r


def test_a_step_naming_a_recorded_control_replays_it_with_its_own_value():
    reply = DesignReply.model_validate({"cases": [{
        "title": "Lead needs a Company", "module": "Leads", "covers": [1], "type": "Validation", "priority": "High",
        "steps": [{"description": "Click New", "event": 1},
                  {"description": "Enter Last Name NoCompany", "target": "Last Name", "value": "NoCompany"},
                  {"description": "Click Save", "target": "Save"}],
        "expected_result": "not saved", "expected": [{"kind": "field_error", "label": "Company", "value": ""}]}]})
    c = finalize(reply, _report_with_events()).cases[0]
    assert [s.event for s in c.steps] == [1, 2, 3]
    assert c.steps[1].value == "NoCompany" and c.steps[1].recorded.value == "Anderson"   # replayed with the case's value
    assert not any("not seen on the page" in p for p in c.problems)


def test_an_element_check_finds_a_heading_and_a_labelled_error_finds_its_field(base_url):
    steps = _create(company="")
    case = TestCase(id="TC-03", title="Lead needs a Company", module="Leads", type="Validation", priority="High",
                    steps=steps, expected_result="not saved",
                    expected=[Expectation(kind="field_error", label="Company", value=""),
                              Expectation(kind="element_visible", value="New Lead", observed=True)])
    v = verify_case(case, base_url + "/crm.html", NoModel(), BrowserConfig())
    assert v.status == "verified", v.reason
    assert v.expectations[0].actual == "Complete this field."
    assert "Company" in v.assertions[0].locator.value
    assert v.assertions[1].kind == "text_visible" and v.assertions[1].value == "New Lead"


def _record_report():
    """New -> Save lands on a record page; its Details tab (E3, the explorer's own click) shows Name."""
    from test_pilot.evidence import Changes
    from test_pilot.explorer import Event
    details = _step("click", "tab", "Details")
    r = ExplorationReport(url="https://crm.test/", requirement="create a lead",
                          plan=Plan(requirement_points=["create a lead"]),
                          pages=[PageInfo(signature="form", url="https://crm.test/new", buttons=["Save"]),
                                 PageInfo(signature="record", url="https://crm.test/lead/1",
                                          texts=["Lead", "John Anderson", "Title Company TechNova Solutions"],
                                          buttons=["Details"]),
                                 PageInfo(signature="details", url="https://crm.test/lead/1",   # the tab: its own view
                                          texts=["Name John Anderson", "Company TechNova Solutions"])])
    for n, (page, step, ch) in enumerate([("form", SAVE, Changes(navigated=True)),
                                          ("record", details, Changes(appeared=["Name John Anderson"])),
                                          ("details", _step("click", "button", "Follow"), Changes())], 1):
        r.events.append(Event(n=n + 1, url="u", page=page, action={"kind": "click"}, status="ok",
                              message=f'clicked [{n}] {step.widget} "{step.element_name}" -> page changed',
                              step=step, changes=ch))
    r.events.insert(0, Event(n=1, url="u", page="form", action={"kind": "type"}, status="ok", message="typed",
                             step=_step("type", "textbox", "Company", "TechNova Solutions")))
    return r


def _record_case(steps, expected):
    reply = DesignReply.model_validate({"cases": [{
        "title": "Create a lead", "module": "Leads", "covers": [1], "type": "Positive", "priority": "High",
        "steps": steps, "expected_result": "saved", "expected": expected}]})
    return finalize(reply, _record_report()).cases[0]


def test_a_value_shown_only_after_a_skipped_click_is_flagged_with_that_click():
    typed = {"description": "Enter Company", "event": 1, "value": "Globex"}
    saved = [typed, {"description": "Save", "event": 2}]
    c = _record_case(saved, [{"kind": "field_shows", "label": "Name", "value": "Ann Lee"},
                             {"kind": "field_shows", "label": "Company", "value": "Globex"}])
    assert any('"Name" is only shown after E3 (clicked tab "Details")' in p for p in c.problems), c.problems
    assert not any('"Company"' in p for p in c.problems)      # new data under a label the record page shows
    c = _record_case(saved + [{"description": "Open Details", "event": 3}],
                     [{"kind": "field_shows", "label": "Name", "value": "John Anderson"}])
    assert not any("only shown after" in p for p in c.problems) and c.grounding == "observed", c.problems


def test_the_skipped_click_is_kept_for_the_verifier():
    saved = [{"description": "Enter Company", "event": 1, "value": "Globex"}, {"description": "Save", "event": 2}]
    c = _record_case(saved, [{"kind": "field_shows", "label": "Name", "value": "Ann Lee"}])
    assert c.expected[0].reveal is not None and c.expected[0].reveal.element_name == "Details"


def test_the_verifier_adds_the_click_that_shows_a_checked_value_without_the_model(base_url):
    case = TestCase(id="TC-05", title="Create a Lead", module="Leads", type="Positive", priority="High",
                    steps=_create(), expected_result="saved",
                    expected=[Expectation(kind="field_shows", label="Company", value="TechNova Solutions"),
                              Expectation(kind="field_shows", label="Lead Source", value="Web",
                                          reveal=_step("click", "tab", "Details"))])
    v = verify_case(case, base_url + "/crm.html", NoModel(), BrowserConfig())
    assert v.status == "verified", v.reason
    assert v.steps[-1].description.startswith("(added)") and v.steps[-1].how == "replayed"
    assert v.definition[-1].element_name == "Details" and any("added step" in n for n in v.notices)
    assert {a.label for a in v.assertions} == {"Company", "Lead Source"}


def test_a_negative_case_must_also_show_that_nothing_was_saved():
    msg_only = [{"kind": "field_error", "label": "Company", "value": "Complete this field."}]
    c = _design(type="Validation", expected=msg_only)
    assert any("no check that nothing was saved" in p for p in c.problems)
    c = _design(type="Validation", expected=msg_only + [{"kind": "url_contains", "value": "crm.test/lead"}])
    assert not any("nothing was saved" in p for p in c.problems)
