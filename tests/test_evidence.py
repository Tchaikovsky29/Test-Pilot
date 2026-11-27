import pytest

from test_pilot.actions import Actor, Click, Type
from test_pilot.evidence import Changes, diff, render_changes
from test_pilot.models import Element, Line, Observation
from test_pilot.policy import Policy


def find(actor, name):
    return next(e.id for e in actor.obs.elements if name in (e.name, e.display_name))


@pytest.fixture()
def make(session, base_url):
    def _make(path):
        actor = Actor(session, Policy(start_url=base_url + "/"), click_timeout_ms=1500)
        session.goto(base_url + path)
        actor.look()
        return actor
    return _make


# ---- the diff itself (no browser) ---------------------------------------------------------------------------------

def obs(url="http://x/a", *lines, elements=()):
    return Observation(url=url, lines=[Line(kind=k, text=t) for k, t in lines], elements=list(elements))


def test_diff_of_nothing_is_empty():
    assert diff(None, obs()).empty
    same = obs("http://x/a", ("heading", "Title"), ("text", "Body"))
    assert diff(same, same).empty


def test_diff_reports_appeared_and_disappeared_text():
    before = obs("http://x/a", ("heading", "Title"), ("text", "Old line"))
    after = obs("http://x/a", ("heading", "Title"), ("text", "New line"))
    ch = diff(before, after)
    assert ch.appeared == ["New line"] and ch.disappeared == ["Old line"] and not ch.navigated


def test_diff_ignores_fragment_and_case_and_spacing():
    before = obs("http://x/a", ("text", "Hello  World"))
    after = obs("http://x/a#top", ("text", "hello world"))
    assert diff(before, after).empty


def test_diff_after_navigation_keeps_only_headings():
    before = obs("http://x/a", ("heading", "A"), ("text", "old"))
    after = obs("http://x/b", ("heading", "B"), ("text", "lots of new body text"))
    ch = diff(before, after)
    assert ch.navigated and ch.url == "http://x/b" and ch.appeared == ["B"] and ch.disappeared == []


def test_diff_alerts_and_errors():
    el = lambda msg: Element(id=1, role="textbox", widget="textbox", name="City", validation=msg)  # noqa: E731
    before = obs("http://x/a", elements=[el("")])
    after = obs("http://x/a", ("alert", "Could not find trips"), elements=[el("Please fill out this field.")])
    ch = diff(before, after)
    assert ch.alerts == ["Could not find trips"] and ch.errors == ["City: Please fill out this field."]
    assert diff(after, after).empty                              # an error that was already there is not news


def test_render_changes_caps_long_lists():
    ch = Changes(appeared=[f"line {i}" for i in range(12)], errors=["City: required"])
    text = render_changes(ch)
    assert text.count("+ line") == 8 and "+ (… 4 more)" in text and "error: City: required" in text
    assert render_changes(Changes()) == ""


# ---- in the browser -----------------------------------------------------------------------------------------------

def test_first_look_has_no_changes(make):
    assert make("/").changes.empty


def test_search_results_show_up_as_changes(make):
    a = make("/")
    a.act(Click(id=find(a, "Search flights")))
    text = a.look()
    assert "[changes since last look]" in text
    assert "+ Results" in text and any("AI-101" in t for t in a.changes.appeared)
    assert not any("Prices include taxes" in t for t in a.changes.appeared + a.changes.disappeared)   # unchanged text
    assert a.look().count("[changes since last look]") == 0       # nothing new on the next look


def test_navigation_is_a_change(make):
    a = make("/")
    a.act(Click(id=find(a, "Hotels")))
    a.look()
    assert a.changes.navigated and "Find a hotel" in a.changes.appeared and a.changes.url.endswith("/hotels?city=pune")


def test_alert_role_is_flagged_and_diffed(make):
    a = make("/form.html")
    a.act(Click(id=find(a, "Show error")))
    text = a.look()
    assert "! Could not find any trips" in text and a.changes.alerts == ["Could not find any trips"]


def test_untouched_required_field_is_not_an_error(make):
    a = make("/form.html")
    assert all(not e.validation for e in a.obs.elements)           # required and empty, but nobody tried yet
    assert "(error:" not in a.look()


def test_failed_submit_exposes_the_browsers_validation_message(make):
    a = make("/form.html")
    a.act(Click(id=find(a, "Search")))                              # real submit button, empty required field
    text = a.look()
    city = next(e for e in a.obs.elements if e.name == "City")
    assert city.validation and "invalid" in city.states
    assert f'(error: "{city.validation}")' in text
    assert a.changes.errors == [f"City: {city.validation}"]
    assert "Looking for trips" not in text                          # the browser blocked the submit


def test_valid_submit_clears_the_error_and_shows_results(make):
    a = make("/form.html")
    a.act(Click(id=find(a, "Search")))
    a.look()
    a.act(Type(id=find(a, "City"), text="Pune"))
    a.act(Click(id=find(a, "Search")))
    a.look()
    assert all(not e.validation for e in a.obs.elements)
    assert "Searching" in a.changes.appeared and "Looking for trips to Pune" in a.changes.appeared


def test_bad_email_pattern_is_reported(make):
    a = make("/form.html")
    a.act(Type(id=find(a, "City"), text="Pune"))
    a.act(Type(id=find(a, "Email"), text="not-an-email"))
    a.act(Click(id=find(a, "Search")))
    a.look()
    email = next(e for e in a.obs.elements if e.name == "Email")
    assert email.validation and any(s.startswith("Email:") for s in a.changes.errors)


def test_constraints_are_shown_for_test_design(make):
    a = make("/form.html")
    by = {e.name: e for e in a.obs.elements}
    assert by["City"].constraints == {"maxlength": "20"} and by["City"].required
    assert by["Age"].constraints == {"min": "18", "max": "99"}
    assert by["Email"].constraints["pattern"].startswith("[^@]+@")
    text = a.look()
    assert 'textbox "City" required [maxlength=20]' in text and "[min=18 max=99]" in text