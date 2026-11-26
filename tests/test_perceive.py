import pytest

from test_pilot import NavigationError, validate_url
from test_pilot.locators import is_dynamic_id, verify_locators
from test_pilot.render import render
from test_pilot.text import clean_name, is_icon_text  # noqa: E402
from test_pilot.watch import describe_change

# ---- URL handling (no browser) -------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "phptravels.net", "ftp://x.com", "javascript:alert(1)", "http://"])
def test_validate_url_rejects(bad):
    with pytest.raises(NavigationError):
        validate_url(bad)


def test_validate_url_accepts():
    assert validate_url(" https://phptravels.net/ ") == "https://phptravels.net/"


@pytest.mark.parametrize("v,dyn", [("promo-8f3a9c21d4e5", True), ("ui-id-3", True), (":r1:", True),
                                    ("hotel_location", False), ("check_in_time", False), ("from", False)])
def test_dynamic_ids(v, dyn):
    assert is_dynamic_id(v) is dyn


# ---- navigation errors ----------------------------------------------------------------------------------------

def test_unreachable_host_is_a_clear_error(session):
    import socket
    with socket.socket() as s:                 # a free port with nothing listening on it
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(NavigationError, match="ERR_CONNECTION_REFUSED"):
        session.goto(f"http://127.0.0.1:{port}/")


def test_http_error_status_is_reported(session, base_url):
    with pytest.raises(NavigationError, match="404"):
        session.goto(base_url + "/nope.html")


# ---- what the model sees --------------------------------------------------------------------------------------

@pytest.fixture()
def flight(session, base_url):
    session.goto(base_url + "/")
    return session


def test_invisible_text_is_dropped(flight):
    text = flight.look()
    assert "Booking confirmed" not in text      # opacity:0 pre-rendered banner
    assert "Hidden promo text" not in text      # display:none
    assert "Prices include taxes." in text      # visible text stays


def test_controls_and_widgets(flight):
    obs = flight.observe()
    by_name = {e.name: e for e in obs.elements}
    assert by_name["From"].required and "required" in by_name["From"].states
    assert by_name["Contact email"].input_type == "email"
    cabin = by_name["Cabin"]
    assert cabin.widget == "select"
    assert cabin.options == ["Economy", "Premium economy", "Business"]   # disabled option dropped
    assert by_name["Departure date"].widget == "date"
    assert by_name["Nonstop only"].states == ["unchecked"]
    assert "disabled" in by_name["Book now"].states
    assert "invalid" not in by_name["Departure date"].states            # untouched field is not "invalid"


def test_hint_is_attached_not_repeated(flight):
    text = flight.look()
    assert text.count("Enter an IATA code or a city") == 1


def test_password_never_reaches_the_model(flight):
    obs = flight.observe()
    assert "hunter2" not in render(obs)
    assert "hunter2" not in obs.model_dump_json()


def test_links_show_destination(flight):
    links = {e.name: e.href for e in flight.observe().elements if e.role == "link"}
    assert links["Flights"] == "/flights"
    assert links["Partner deals"].startswith("partner.example.com")
    assert links["Back to top"] == ""          # in-page anchor


def test_iframe_content_is_included(flight):
    text = flight.look()
    assert "--- frame" in text and 'textbox "Offer code"' in text
    inner = next(e for e in flight.observe().elements if e.name == "Offer code")
    assert inner.frame_url.endswith("/frame.html")


def test_truncation_keeps_head_and_tail(flight):
    text = flight.look(max_chars=300)
    assert "lines cut here" in text
    assert text.splitlines()[0].startswith("[state]") and text.rstrip().endswith("[/state]")
    assert 'button "Apply offer"' in text      # the last line survives


# ---- waiting and change detection -----------------------------------------------------------------------------

def test_observation_waits_for_slow_request(flight):
    flight.observe()
    flight.page.click("#go")                  # triggers a 0.4 s fetch, then renders results
    obs = flight.observe()
    assert "AI-101" in render(obs), "perception returned before the page finished updating"
    assert any(e.name == "Select" for e in obs.elements)


def test_new_elements_are_marked(flight):
    flight.observe()
    flight.page.click("#go")
    text = flight.look()
    assert '*[' in text and 'button "Select"' in text
    again = flight.look()
    assert "*[" not in again                  # nothing new on a second look


def test_change_detection(flight):
    before = flight.snapshot()
    flight.page.click("#go")
    flight.observe()
    assert describe_change(before, flight.snapshot()) == "page changed"
    before = flight.snapshot()
    assert describe_change(before, flight.snapshot()) == "no change"
    flight.page.click("text=Hotels")
    flight.observe()
    assert describe_change(before, flight.snapshot()).startswith("navigated to")


def test_dialogs_are_handled_and_reported(flight):
    flight.page.evaluate("setTimeout(() => alert('Session expiring'), 50)")
    flight.page.wait_for_timeout(200)
    assert "Session expiring" in flight.look()


# ---- locators -------------------------------------------------------------------------------------------------

def test_locators_are_unique_and_user_facing(flight):
    obs = flight.observe()
    for e in obs.elements:
        if e.widget in ("link", "option"):
            continue
        best = verify_locators(flight.page, e)
        assert best is not None, f"no unique locator for {e.widget} {e.name!r}"
        assert best.kind in ("role", "label", "placeholder", "testid", "text")   # CSS only as a last resort


def test_dynamic_ids_are_never_offered(flight):
    promo = next(e for e in flight.observe().elements if e.name == "Promo code")
    assert not any("8f3a9c21d4e5" in l.value for l in promo.locators)


def test_locator_python_is_executable(flight):
    e = next(e for e in flight.observe().elements if e.name == "From")
    loc = verify_locators(flight.page, e)
    page = flight.page
    target = eval(loc.python("page"))        # the same string will be emitted into generated tests
    target.fill("DEL")
    assert page.input_value("#from") == "DEL"


def test_iframe_locator_resolves_inside_the_frame(flight):
    e = next(e for e in flight.observe().elements if e.name == "Offer code")
    assert verify_locators(flight.page, e) is not None


# ---- page-type signature --------------------------------------------------------------------------------------

def test_signature_ignores_values_and_differs_across_pages(session, base_url):
    session.goto(base_url + "/")
    a = session.observe().signature()
    session.page.fill("#from", "Delhi")
    assert session.observe().signature() == a
    session.goto(base_url + "/hotels")
    assert session.observe().signature() != a


@pytest.mark.parametrize("raw,clean", [
    ("hotel Stays", "Stays"), ("auto_awesome AI Trip Planner", "AI Trip Planner"),
    ("flag Nationality Select Nationality", "Nationality Select Nationality"),
    ("Check-in Date", "Check-in Date"), ("iPhone Case", "iPhone Case"), ("search", "search"), ("hotel", "hotel"),
    ("Book now", "Book now"), ("flight 2 stops", "2 stops"),
])
def test_clean_name(raw, clean):
    assert clean_name(raw) == clean


@pytest.mark.parametrize("text,icon", [
    ("vpn_key", True), ("currency_exchange", True), ("help", True), ("storage", True),
    ("IDR", False), ("EN", False), ("or", False), ("Search", False), ("Check-in", False), ("01", False),
    ("two words", False), ("Prices include taxes.", False),
])
def test_is_icon_text(text, icon):
    assert is_icon_text(text) is icon


def test_autocomplete_fields_are_marked(flight):
    flight.page.evaluate("document.querySelector('#from').setAttribute('aria-autocomplete', 'list')")
    from_element = next(e for e in flight.observe().elements if e.name == "From")
    assert "autocomplete" in from_element.states


def test_city_and_readonly_date_fields_are_marked(flight):
    flight.page.evaluate("document.querySelector('#from').setAttribute('placeholder', 'City or airport')")
    obs = flight.observe()
    by_name = {e.name: e for e in obs.elements}
    assert "autocomplete" in by_name["From"].states
    assert "date-picker" in by_name["Departure date"].states