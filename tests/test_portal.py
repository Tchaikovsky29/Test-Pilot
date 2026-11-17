"""Regressions found on the real phptravels.net page: div widgets, a modal last in the DOM, repeated cards."""
import pytest

from test_pilot.locators import verify_all
from test_pilot.render import render


@pytest.fixture()
def portal(session, base_url):
    session.goto(base_url + "/portal.html")
    return session


def names(obs, widget):
    return [e.name for e in obs.elements if e.widget == widget]


def test_div_widgets_become_clickables(portal):
    obs = portal.observe()
    clickable = names(obs, "clickable")
    assert "flight Flights" in clickable and "hotel Stays" in clickable      # cursor:pointer tabs
    assert "Select destination" in clickable                                   # custom dropdown
    assert "Tours" in clickable                                                # click listener only, no pointer cursor


def test_plain_text_is_not_clickable(portal):
    clickable = names(portal.observe(), "clickable")
    assert not any("2 Guests" in c for c in clickable)
    assert not any("Prices include" in c for c in clickable)
    assert len(clickable) == 4                                                 # nothing from the body-level handler


def test_clickable_is_rendered_and_locatable(portal):
    obs = portal.observe()
    assert 'clickable "hotel Stays"' in render(obs)
    stays = next(e for e in obs.elements if e.name == "hotel Stays")
    verify_all(portal.page, [stays])
    best = stays.best_locator()
    assert best and best.kind == "text"
    assert eval(best.python("portal.page")).count() == 1                       # noqa: S307


def test_blocking_dialog_comes_first(portal):
    obs = portal.observe()
    assert obs.blocked
    text = render(obs)
    lines = text.splitlines()
    assert lines[1].startswith("--- dialog blocking the page")
    assert "I Understand" in lines[3] or "I Understand" in lines[4]
    assert text.index("I Understand") < text.index("Travel The Way You Love")  # although it is last in the document
    assert text.count("--- end of dialog ---") == 1


def test_dialog_gone_after_dismissal(portal):
    portal.observe()
    portal.page.click("#ok")
    obs = portal.observe()
    assert not obs.blocked and "dialog blocking" not in render(obs)


def test_card_text_is_not_repeated(portal):
    text = render(portal.observe(), 20000)
    assert text.count("How to plan trip number 3") == 1        # in the link line only, not again as a heading
    lines = text.splitlines()
    assert not any(l.startswith("Travel broadens") or l == "Read More" for l in lines)   # no repeated body text


def test_description_equal_to_name_is_dropped(portal):
    btn = next(e for e in portal.observe().elements if e.name == "Search Hotels")
    assert btn.description == ""
    assert "hint" not in render(portal.observe())


def test_budget_drops_prose_before_controls(portal):
    full = render(portal.observe(), 20000)
    small = render(portal.observe(), 1700)
    assert len(small) < len(full)
    for must in ('clickable "flight Flights"', 'clickable "Select destination"', 'button "Search Hotels"',
                 'button "I Understand & Continue"', "## Travel The Way You Love"):
        assert must in small, must
    assert "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt" not in small


def test_identical_siblings_get_distinct_nth_locators(portal):
    obs = portal.observe()
    services = [e for e in obs.elements if e.name == "Services"]
    assert len(services) == 2
    verify_all(portal.page, services)
    a, b = (e.best_locator() for e in services)
    assert a and b and a.fragile and b.fragile and (a.nth, b.nth) == (0, 1)
    assert eval(a.python("portal.page")).count() == 1                           # noqa: S307
    assert eval(a.python("portal.page")).bounding_box() != eval(b.python("portal.page")).bounding_box()
