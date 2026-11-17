import pytest

from test_pilot.actions import Actor, Click, GoBack, Press, Select, Type, parse_action
from test_pilot.policy import Policy


def find(actor, name, widget=None):
    """The id of the first element on the last look with this raw or display name."""
    for e in actor.obs.elements:
        if name in (e.name, e.display_name) and (widget is None or e.widget == widget):
            return e.id
    raise AssertionError(f"{name!r} not on the page: {[(e.widget, e.name) for e in actor.obs.elements]}")


@pytest.fixture()
def make(session, base_url):
    def _make(path="/", **kw):
        actor = Actor(session, Policy(start_url=base_url + "/", **kw), click_timeout_ms=1500)
        session.goto(base_url + path)
        actor.look()
        return actor
    return _make


# ---- ordinary use -----------------------------------------------------------------------------------------------

def test_click_waits_for_the_page_and_reports_the_change(make):
    a = make()
    r = a.act(Click(id=find(a, "Search flights")))
    assert r.ok and "page changed" in r.message
    assert "AI-101" in a.look()                      # results were rendered before we returned


def test_type_select_and_checkbox(make):
    a = make()
    assert a.act(Type(id=find(a, "From"), text="DEL")).ok
    assert a.session.page.input_value("#from") == "DEL"
    assert a.act(Select(id=find(a, "Cabin"), option="business")).ok          # case-insensitive
    assert a.session.page.input_value("#cabin") == "Business"
    assert a.act(Click(id=find(a, "Nonstop only"))).ok
    a.look()
    assert "checked" in next(e for e in a.obs.elements if e.name == "Nonstop only").states


def test_select_with_unknown_option_lists_the_choices(make):
    a = make()
    r = a.act(Select(id=find(a, "Cabin"), option="Luxury"))
    assert r.status == "failed" and "Economy | Premium economy | Business" in r.message


def test_native_date_wants_iso(make):
    a = make()
    r = a.act(Type(id=find(a, "Departure date"), text="19 Dec 2026"))
    assert r.status == "failed" and "YYYY-MM-DD" in r.message
    assert a.act(Type(id=find(a, "Departure date"), text="2026-12-19")).ok
    assert a.session.page.input_value("#depart") == "2026-12-19"


def test_typing_into_a_button_is_explained(make):
    a = make()
    r = a.act(Type(id=find(a, "Search flights"), text="x"))
    assert r.status == "failed" and "use click" in r.message


def test_readonly_field_points_to_the_picker(make):
    a = make("/portal.html")
    a.act(Click(id=find(a, "I Understand & Continue")))
    a.look()
    r = a.act(Type(id=find(a, "Check-in"), text="2026-12-19"))
    assert r.status == "failed" and "read-only" in r.message and "click it" in r.message


def test_custom_dropdown_via_clickables(make):
    a = make("/checkout.html")
    assert a.act(Click(id=find(a, "Select destination"))).ok
    text = a.look()
    assert '*[' in text and 'clickable "Delhi (DEL)"' in text                # the options are new elements
    assert a.act(Click(id=find(a, "Delhi (DEL)"))).ok
    assert a.session.page.inner_text("#destval") == "Delhi (DEL)"


def test_identical_buttons_are_told_apart(make):
    a = make("/checkout.html")
    adds = [e.id for e in a.obs.elements if e.name == "Add"]
    assert len(adds) == 2
    assert a.act(Click(id=adds[1])).ok
    assert a.session.page.inner_text("#counts") == "0,1"                      # the second one, not the first


# ---- safety -----------------------------------------------------------------------------------------------------

def test_password_is_never_typed(make):
    a = make()
    r = a.act(Type(id=find(a, "Account password"), text="zx-typed-9431"))
    assert r.status == "stopped" and "a person must enter" in r.message and "zx-typed-9431" not in r.message
    assert a.session.page.input_value("#pw") == "hunter2"                     # untouched


def test_card_field_is_never_typed(make):
    a = make("/checkout.html")
    r = a.act(Type(id=find(a, "Card number"), text="4111111111111111"))
    assert r.status == "stopped" and a.session.page.input_value("#cardno") == ""
    assert a.act(Type(id=find(a, "Passenger name"), text="Test User")).ok     # ordinary test data is fine


def test_final_and_destructive_buttons_are_not_clicked(make):
    a = make("/checkout.html")
    for name in ("Pay now", "Delete booking"):
        r = a.act(Click(id=find(a, name)))
        assert r.status == "stopped", name
    assert a.session.page.evaluate("() => [window.paid, window.deleted]") == [None, None]
    assert a.steps == []                                                       # nothing was recorded as done


def test_external_link_is_blocked_before_clicking(make):
    a = make()
    url = a.session.page.url
    r = a.act(Click(id=find(a, "Partner deals")))
    assert r.status == "blocked" and "partner.example.com" in r.message
    assert a.session.page.url == url


def test_internal_link_navigates(make):
    a = make()
    r = a.act(Click(id=find(a, "Hotels")))
    assert r.ok and "navigated to" in r.message and "/hotels" in r.message
    assert a.act(GoBack()).ok and a.session.page.url.endswith("/")


def test_leaving_the_site_by_script_is_undone(make):
    a = make("/checkout.html")
    r = a.act(Click(id=find(a, "Visit partner site")))
    assert r.status == "blocked" and "localhost" in r.message and "went back" in r.message
    assert "checkout.html" in a.session.page.url


def test_dialog_blocking_the_page_is_explained(make):
    a = make("/portal.html")
    assert a.obs.blocked
    r = a.act(Click(id=find(a, "Flights", "clickable")))
    assert r.status == "failed" and "dialog is blocking" in r.message
    assert a.act(Click(id=find(a, "I Understand & Continue"))).ok
    a.look()
    assert not a.obs.blocked and a.act(Click(id=find(a, "Flights", "clickable"))).ok


def test_action_budget(make):
    a = make(max_actions=2)
    assert a.act(Press(key="Escape")).ok and a.act(Press(key="Escape")).ok
    r = a.act(Press(key="Escape"))
    assert r.status == "failed" and "budget" in r.message


# ---- stale ids and look-first rule ------------------------------------------------------------------------------

def test_unknown_and_stale_ids(make):
    a = make()
    assert "not on the page" in a.act(Click(id=999999)).message
    go = find(a, "Search flights")
    a.session.page.evaluate("document.getElementById('go').remove()")
    r = a.act(Click(id=go))
    assert r.status == "failed" and "no longer on the page" in r.message


def test_acting_before_looking(session, base_url):
    a = Actor(session, Policy(start_url=base_url))
    session.goto(base_url + "/")
    assert "look at the page first" in a.act(Click(id=1)).message


# ---- recorded steps become runnable Playwright --------------------------------------------------------------------

def test_recorded_steps_replay_as_code(make):
    a = make()
    a.act(Type(id=find(a, "From"), text="DEL"))
    a.act(Select(id=find(a, "Cabin"), option="Business"))
    a.act(Click(id=find(a, "Search flights")))
    code = [s.code() for s in a.steps]
    assert code[0] == 'page.get_by_role("textbox", name="From", exact=True).fill("DEL")'
    assert code[1] == 'page.get_by_role("combobox", name="Cabin", exact=True).select_option(label="Business")'
    assert code[2] == 'page.get_by_role("button", name="Search flights", exact=True).click()'
    page = a.session.page
    page.reload()
    for line in code:                       # the same lines, run fresh, reproduce the result
        exec(line, {"page": page})          # noqa: S102
    page.get_by_text("AI-101").wait_for(timeout=3000)
    assert page.input_value("#from") == "DEL"


# ---- parsing what the model sends ---------------------------------------------------------------------------------

def test_parse_action():
    assert parse_action({"kind": "click", "id": 3}) == Click(id=3)
    assert parse_action('{"kind": "type", "id": 3, "text": "x", "keys": true}').keys is True
    with pytest.raises(Exception):
        parse_action({"kind": "format_disk"})
    with pytest.raises(Exception):
        parse_action({"kind": "click"})                                       # id is required