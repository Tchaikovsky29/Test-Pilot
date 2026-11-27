"""Dates are set through the field's own calendar, whatever it looks like, and checked in the site's own format."""
from datetime import date

import pytest

from test_pilot.actions import Actor, Type
from test_pilot.datepick import in_format, means, pick_date
from test_pilot.policy import Policy


@pytest.mark.parametrize("value", ["03-12-2026", "3/12/2026", "Dec 3, 2026", "3 December 2026", "2026-12-03", "03.12.26"])
def test_values_in_any_format_are_read_as_the_date(value):
    assert means(value, date(2026, 12, 3))


@pytest.mark.parametrize("value", ["18-05-1932", "", "Dec 4, 2026", "03-11-2026", "Nov 3, 2026"])
def test_other_dates_are_not(value):
    assert not means(value, date(2026, 12, 3))


def test_a_placeholder_format_is_followed():
    d = date(2026, 12, 3)
    assert in_format(d, "dd/mm/yyyy") == "03/12/2026" and in_format(d, "MM-DD-YY") == "12-03-26"
    assert in_format(d, "Departure Date") is None and in_format(d, "yyyy/yyyy/dd") is None


def _pick(session, base_url, query, iso):
    session.goto(f"{base_url}/datepicker.html?{query}")
    field = session.page.locator("#when")
    return pick_date(field, iso), field


@pytest.mark.parametrize("query, iso, shown", [
    ("format=dmy&nav=class", "2026-11-28", "28-11-2026"),            # this month, a day also shown in the next month's row
    ("format=dmy&nav=class", "2026-12-03", "03-12-2026"),            # next month; 3 also appears as a grey day before
    ("format=name&nav=aria", "2027-03-15", "Mar 15, 2027"),          # several months on, controls named by aria-label only
    ("format=dmy&nav=bare&mark=no", "2027-01-02", "02-01-2027"),     # icon-only controls, other-month days unmarked
    ("format=dmy&nav=bare&mark=no", "2026-12-30", "30-12-2026"),     # a late day: the unmarked grey 30 comes first
    ("format=name&nav=class", "2026-08-09", "Aug 9, 2026"),          # backwards
    ("format=dmy&nav=class&months=2", "2027-01-20", "20-01-2027"),   # two months side by side
])
def test_the_calendar_is_used_whatever_it_looks_like(session, base_url, query, iso, shown):
    picked, field = _pick(session, base_url, query, iso)
    assert picked.ok, picked.message
    assert picked.how == "calendar" and field.input_value() == shown


def test_a_disabled_day_is_reported_not_forced(session, base_url):
    picked, field = _pick(session, base_url, "min=2026-11-20", "2026-11-10")
    assert not picked.ok and "disabled" in picked.message and field.input_value() == ""


def test_without_a_calendar_the_placeholder_format_is_written(session, base_url):
    picked, field = _pick(session, base_url, "mode=plain", "2026-12-03")
    assert picked.ok and picked.how == "typed" and field.input_value() == "03/12/2026"


def test_the_type_action_sets_a_picker_field_through_its_calendar_and_records_it(session, base_url):
    session.goto(f"{base_url}/datepicker.html?format=dmy&nav=bare&mark=no")
    a = Actor(session, Policy(start_url=base_url))
    a.look()
    field = next(e for e in a.obs.elements if e.name == "Departure Date")
    assert "date-picker" in field.states
    r = a.act(Type(id=field.id, text="2026-12-30"))
    assert r.ok, r.message
    assert session.page.locator("#when").input_value() == "30-12-2026"
    step = a.steps[-1]
    assert step.date_picker and step.value == "2026-12-30" and step.code().startswith("pick_date(")
    bad = a.act(Type(id=field.id, text="30-12-2026"))                 # the model must still give ISO
    assert not bad.ok and "YYYY-MM-DD" in bad.message


def test_a_calendar_already_open_under_a_shade_is_used_as_it_is(session, base_url):
    session.goto(f"{base_url}/datepicker.html?format=dmy&nav=class&overlay=1")
    field = session.page.locator("#when")
    field.click()                                                      # opened before, as the model does
    picked = pick_date(field, "2026-12-03")
    assert picked.ok, picked.message
    assert field.input_value() == "03-12-2026"


def test_a_field_covered_by_a_shade_is_still_opened(session, base_url):
    session.goto(f"{base_url}/datepicker.html?format=dmy&nav=class&overlay=1")
    page = session.page
    page.evaluate("() => { const s = document.getElementById('shade'); s.classList.add('open'); s.onclick = null; }")
    picked = pick_date(page.locator("#when"), "2026-11-28")             # the click cannot land: focus + click event
    assert picked.ok, picked.message


def test_an_example_date_in_a_hint_gives_the_format():
    d = date(2026, 12, 3)
    assert in_format(d, "Format: 12/31/2024") == "12/3/2026"            # 31 is no month: month/day/year
    assert in_format(d, "e.g. 31.01.2024") == "03.12.2026"              # day first, with the example's zeros
    assert in_format(d, "Format: 2024-12-31") == "2026-12-03"
    assert in_format(d, "05/06/2024") is None                           # ambiguous: no guess
