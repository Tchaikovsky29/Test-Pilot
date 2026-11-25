from src.locators import candidates
from src.models import Element


def test_long_named_clickable_gets_a_prefix_text_locator():
    # An autocomplete option with no attributes and a name past MAX_TEXT_NAME used to get no locator at all,
    # so the actor reported it as "no longer on the page" while it was on screen.
    name = "New Delhi, National Capital Territory of India, India Indira Gandhi International Airport DEL"
    locs = candidates(Element(id=1, role="generic", widget="clickable", name=name), {})
    assert len(locs) == 1
    loc = locs[0]
    assert loc.kind == "text" and not loc.exact
    assert name.startswith(loc.value) and len(loc.value) <= 60


def test_short_named_clickable_keeps_the_exact_text_locator():
    locs = candidates(Element(id=1, role="generic", widget="clickable", name="Pune International Airport PNQ"), {})
    assert any(l.kind == "text" and l.exact and l.value == "Pune International Airport PNQ" for l in locs)
