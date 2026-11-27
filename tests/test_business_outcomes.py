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
