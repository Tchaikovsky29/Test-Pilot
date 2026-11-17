"""Stable, test-worthy locators for an element, and a live check that each one is unique.

The assignment asks for user-facing locators (getByRole / getByLabel / getByText) first, with CSS only when justified.
Candidates are ordered that way; verify_locators() counts live matches so a generated test never uses an ambiguous one.
"""
from __future__ import annotations

import re

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Frame, Page
from .text import clean_name
from .models import Element, Locator
from .roles import CONTROLS, LABELLED

# Ids that look machine-generated: long digit runs, hex blobs, React :r1: ids, ember123, ui-id-3.
_DYNAMIC_ID = re.compile(r"\d{4,}|[0-9a-f]{8,}|^:r|ember\d+|ui-id-\d+", re.I)
_TESTID_ATTRS = ("data-testid", "data-test-id")
_OTHER_TEST_ATTRS = ("data-test", "data-qa", "data-cy")
MAX_TEXT_NAME = 80  # very long names (whole result cards) make brittle locators


def is_dynamic_id(value: str) -> bool:
    return bool(_DYNAMIC_ID.search(value))


def _css_attr(attr: str, value: str) -> str:
    return f'[{attr}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"]'


def candidates(e: Element, attrs: dict[str, str]) -> list[Locator]:
    out: list[Locator] = []
    name = e.name
    clean = clean_name(name)  # "hotel Stays" -> "Stays": the icon word is not part of what a user sees
    if name and e.role in CONTROLS:
        if clean != name:
            out.append(Locator(kind="role", role=e.role, value=clean, exact=False))
        out.append(Locator(kind="role", role=e.role, value=name))
    if name and e.widget in LABELLED:
        out.append(Locator(kind="label", value=name))
    if attrs.get("placeholder"):
        out.append(Locator(kind="placeholder", value=attrs["placeholder"]))
    for a in _TESTID_ATTRS:
        if attrs.get(a):
            out.append(Locator(kind="testid", value=attrs[a]))
    for a in _OTHER_TEST_ATTRS:
        if attrs.get(a):
            out.append(Locator(kind="css", value=_css_attr(a, attrs[a])))
    if name and (e.role in ("button", "link") or e.widget == "clickable") and len(name) <= MAX_TEXT_NAME:
        if clean != name:
            out.append(Locator(kind="text", value=clean, exact=False))
        out.append(Locator(kind="text", value=name))
    if attrs.get("id") and not is_dynamic_id(attrs["id"]):
        out.append(Locator(kind="css", value=_css_attr("id", attrs["id"])))
    if attrs.get("name") and e.tag:
        out.append(Locator(kind="css", value=e.tag.lower() + _css_attr("name", attrs["name"])))
    seen: set[tuple] = set()
    unique = []
    for loc in out:
        key = (loc.kind, loc.role, loc.value, loc.exact)
        if key not in seen:
            seen.add(key)
            unique.append(loc)
    return unique


def resolve(root: Page | Frame, loc: Locator):
    if loc.kind == "role":
        r = root.get_by_role(loc.role, name=loc.value, exact=loc.exact)
    elif loc.kind == "label":
        r = root.get_by_label(loc.value, exact=loc.exact)
    elif loc.kind == "placeholder":
        r = root.get_by_placeholder(loc.value, exact=loc.exact)
    elif loc.kind == "testid":
        r = root.get_by_test_id(loc.value)
    elif loc.kind == "text":
        r = root.get_by_text(loc.value, exact=loc.exact)
    else:
        r = root.locator(loc.value)
    if loc.visible:
        r = r.filter(visible=True)
    if loc.nth is not None:
        r = r.nth(loc.nth)
    return r


def root_for(page: Page, frame_url: str) -> Page | Frame:
    if not frame_url:
        return page
    return next((f for f in page.frames if f.url == frame_url), page)


def verify_locators(page: Page, e: Element) -> Locator | None:
    """Fill `matches` on each candidate with its live count and return the first unique one (or None).
    A count of 1 predicts Playwright's strict mode will accept it; counts include hidden elements on purpose."""
    root = root_for(page, e.frame_url)
    for loc in e.locators:
        try:
            loc.matches = resolve(root, loc).count()
        except PlaywrightError:
            loc.matches = None
    if e.best_locator():
        return e.best_locator()
    # Several matches: maybe only one is visible (a hidden mobile copy of the menu, say).
    for loc in list(e.locators):
        if (loc.matches or 0) > 1 and not loc.visible and loc.nth is None:
            narrowed = loc.model_copy(update={"visible": True, "matches": None})
            try:
                narrowed.matches = resolve(root, narrowed).count()
            except PlaywrightError:
                continue
            if narrowed.matches == 1:
                e.locators.append(narrowed)
                break
    return e.best_locator()


def _key(x: Element) -> tuple:
    f = x.locators[0]
    return (x.frame_url, f.kind, f.role, f.value, f.exact)


def ensure_locator(page: Page, e: Element, elements: list[Element]) -> Locator | None:
    """The best unique locator for one element, with a last resort for identical siblings: .nth(i).
    Playwright returns matches in DOM order, as does the accessibility tree, so if the live count of an ambiguous
    locator equals the number of elements we saw sharing it, the i-th one is the i-th match. Such locators are
    marked fragile: they break if the page order changes."""
    best = verify_locators(page, e)
    if best or not e.locators:
        return best
    same = [x for x in elements if x.locators and _key(x) == _key(e)]
    if e.locators[0].matches == len(same):
        e.locators.append(e.locators[0].model_copy(update={"nth": same.index(e), "matches": 1}))
    return e.best_locator()


def verify_all(page: Page, elements: list[Element]) -> None:
    for e in elements:
        ensure_locator(page, e, elements)