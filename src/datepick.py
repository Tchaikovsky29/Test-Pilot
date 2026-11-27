"""Set a date in a date-picker field the way a person does: open the calendar, step to the month, click the day, and
check the field now holds that date. No knowledge of any date-picker library or site.

Why not just write the value: a picker field is usually read-only and the site parses it in its own format
("26-11-2026", "Nov 26, 2026", ...). An ISO string written into it gets misread (2026-11-26 read as day-month-year
becomes a date in 1932), and the picker's own state, which the site submits, never changes. Clicking the calendar
always produces the site's format, whatever it is.

How the calendar is found, generically:
  * a month panel is the smallest element around a "<Month> <Year>" heading that holds at least 28 day numbers;
  * previous/next controls are recognised by label, title, class or glyph (prev, ‹, «, chevron-left, ...), or, when
    they carry no hint at all (an icon-only button), by sitting left/right of the heading;
  * days from neighbouring months and disabled days are skipped by their markings (class old/new/outside/disabled,
    aria-disabled), and when a grid does not mark them, by position: a small number is taken from the top of the grid,
    a large one from the bottom.
When no calendar opens (or none it can read), the date is written in the format the placeholder or hint shows
("dd/mm/yyyy", or an example like "Format: 12/31/2024"), else as ISO: typed if the field can be typed into, and
checked the same way. The result says which way worked and what the field holds.

This module is also what generated tests call (`pick_date(page, locator, "2026-12-03")`), so a test sets dates
exactly as the verifier did.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator

MAX_MONTH_STEPS = 36  # three years either way is plenty for a test date
_MARK = "data-tp-pick"  # where the finder marks the element to click

# Runs in the field's frame. Finds the open calendar, compares its month with the target, and marks one element to
# click: the target day, or the previous/next control. Returns what it marked (or why nothing).
_FIND = r"""
(field, target) => {
  const MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"];
  const monthOf = w => { w = w.toLowerCase().replace(/\.$/, ""); const i = MONTHS.findIndex(m => w.length >= 3 && m.startsWith(w)); return i < 0 ? null : i + 1; };
  const HEAD = /\b([A-Za-z]{3,9})\.?,?\s+(\d{4})\b|\b(\d{4})\s+([A-Za-z]{3,9})\b/;
  const vis = e => { if (!e || !e.getClientRects().length) return false; const s = getComputedStyle(e); return s.visibility !== "hidden" && s.display !== "none" && +s.opacity !== 0; };
  const own = e => [...e.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join(" ").replace(/\s+/g, " ").trim();
  const text = e => (e.innerText || "").replace(/\s+/g, " ").trim();
  const leafDay = e => /^\d{1,2}$/.test(text(e)) && ![...e.children].some(c => text(c) === text(e));
  document.querySelectorAll("[" + MARK + "]").forEach(e => e.removeAttribute(MARK));

  // 1. month headings, each with the panel (smallest ancestor holding a month of day numbers) it titles
  const panels = [];
  for (const h of document.querySelectorAll("body *")) {
    if (h === field || !vis(h)) continue;
    // the heading's own text, or a short split one ("<b>Nov</b> <b>2026</b>") when no child holds the whole of it:
    // a row wrapping the heading and its arrows is not the heading
    const t = own(h) || (h.children.length <= 3 && text(h).length <= 25 && ![...h.children].some(c => HEAD.test(text(c))) ? text(h) : "");
    const m = t.length <= 40 && HEAD.exec(t);
    if (!m) continue;
    const month = monthOf(m[1] || m[4]), year = +(m[2] || m[3]);
    if (!month) continue;
    let p = h.parentElement;
    for (let up = 0; p && up < 6; up++, p = p.parentElement) {
      const days = [...p.querySelectorAll("*")].filter(e => vis(e) && leafDay(e));
      if (days.length >= 28) { if (!panels.some(x => x.p === p)) panels.push({h, p, month, year, days}); break; }
    }
  }
  if (!panels.length) return {what: "none"};
  panels.sort((a, b) => (a.year * 12 + a.month) - (b.year * 12 + b.month));
  const want = target.year * 12 + target.month;

  // 2. the target month is shown: mark its day
  const panel = panels.find(x => x.year * 12 + x.month === want);
  if (panel) {
    const OTHER = /\b(old|new|other|outside|adjacent|off|disabled|unavailable|prev-month|next-month|not-current|is-other)\b/i;
    const marked = e => { for (let x = e; x && x !== panel.p.parentElement; x = x.parentElement) {
      if (OTHER.test(typeof x.className === "string" ? x.className : "") || x.getAttribute("aria-disabled") === "true" || x.hasAttribute("disabled")) return true; } return false; };
    const all = panel.days.filter(e => +text(e) === target.day);
    const clean = all.filter(e => !marked(e));
    const pool = clean.length ? clean : all;
    if (!pool.length) return {what: "no-day", shown: panel.month + "/" + panel.year};
    const pick = pool.length === 1 ? pool[0] : (target.day <= 14 ? pool[0] : pool[pool.length - 1]);
    if (!clean.length && all.length && marked(pick)) return {what: "disabled", shown: panel.month + "/" + panel.year};
    pick.setAttribute(MARK, "day");
    return {what: "day", shown: panel.month + "/" + panel.year};
  }

  // 3. step towards it: a control named or drawn as next/previous, else the one beside the heading
  const dir = want > panels[panels.length - 1].year * 12 + panels[panels.length - 1].month ? "next" : "prev";
  const head = dir === "next" ? panels[panels.length - 1] : panels[0];
  let scope = head.p;
  for (let up = 0; up < 3 && scope.parentElement && scope.parentElement !== document.body; up++) scope = scope.parentElement;
  const HINT = dir === "next" ? /\bnext\b|›|»|→|chevron[-_]?right|arrow[-_]?right|forward/i : /\bprev(ious)?\b|‹|«|←|chevron[-_]?left|arrow[-_]?left|\bback\b/i;
  const hints = e => [e.getAttribute("aria-label"), e.getAttribute("title"), typeof e.className === "string" ? e.className : (e.className && e.className.baseVal), e.id, text(e).length <= 3 ? text(e) : ""].filter(Boolean).join(" ");
  const cands = [...scope.querySelectorAll("*")].filter(e => vis(e) && e !== head.h && !leafDay(e) && !e.contains(head.h));
  let nav = cands.filter(e => HINT.test(hints(e)));
  nav = nav.filter(e => !nav.some(o => o !== e && e.contains(o)));  // the innermost of nested matches
  if (!nav.length) {
    const hr = head.h.getBoundingClientRect();
    const clickable = e => e.matches("button,a,th,[role=button]") || getComputedStyle(e).cursor === "pointer";
    nav = cands.filter(e => { const r = e.getBoundingClientRect();
      const level = r.bottom > hr.top - 4 && r.top < hr.bottom + 4;
      return level && clickable(e) && (dir === "next" ? r.left >= hr.right - 2 : r.right <= hr.left + 2); });
    nav.sort((a, b) => { const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
      return dir === "next" ? ra.left - rb.left : rb.right - ra.right; });
  }
  if (!nav.length) return {what: "no-nav", dir, shown: head.month + "/" + head.year};
  nav[0].setAttribute(MARK, "nav");
  return {what: "nav", dir, shown: head.month + "/" + head.year};
}
""".replace("MARK", repr(_MARK))

_SET_VALUE = """(el, value) => {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  setter ? setter.call(el, value) : (el.value = value);
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
}"""


@dataclass
class Picked:
    ok: bool
    value: str  # what the field holds afterwards
    how: str  # "calendar", "typed", or "" when it failed
    message: str


def means(value: str, d: date) -> bool:
    """Does a field value, in whatever format the site uses, denote date d? ("26-11-2026", "Nov 26, 2026", ...)"""
    v = value.lower()
    nums = [n.lstrip("0") or "0" for n in re.findall(r"\d+", v)]
    if not nums:
        return False
    year = str(d.year) in nums or f"{d.year % 100}" in nums
    names = (calendar.month_name[d.month].lower(), calendar.month_abbr[d.month].lower())
    named = any(re.search(rf"\b{n}", v) for n in names)
    day = str(d.day) in nums
    month = named or nums.count(str(d.month)) >= (2 if d.month == d.day and not named else 1)
    return year and day and month


def in_format(d: date, hint: str) -> str | None:
    """The date in the format a hint shows: a pattern ("dd/mm/yyyy", "MM-DD-YY") or an example date
    ("Format: 12/31/2024": 31 cannot be a month, so month/day/year). None when the hint shows neither."""
    h = hint.lower()
    m = re.search(r"\b(dd|mm|yyyy|yy)([-/. ])(dd|mm|yyyy|yy)\2(dd|mm|yyyy|yy)\b", h)
    if m and len({m.group(1), m.group(3), m.group(4)} & {"dd", "mm"}) == 2:
        parts = {"dd": f"{d.day:02d}", "mm": f"{d.month:02d}", "yyyy": f"{d.year}", "yy": f"{d.year % 100:02d}"}
        return m.group(2).join(parts[g] for g in (m.group(1), m.group(3), m.group(4)))
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", h)
    if m:
        return d.isoformat()
    m = re.search(r"\b(\d{1,2})([/.-])(\d{1,2})\2(\d{4}|\d{2})\b", h)
    if not m:
        return None
    a, sep, b, y = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    if a > 12 and b <= 12:
        first, second = d.day, d.month  # day first
    elif b > 12 and a <= 12:
        first, second = d.month, d.day  # month first
    else:
        return None  # 05/06/2024 does not say which is which
    zeros = m.group(1).startswith("0") or m.group(3).startswith("0")  # "31.01.2024": the site pads to two digits
    year = str(d.year) if len(y) == 4 else f"{d.year % 100:02d}"
    return sep.join([f"{first:02d}" if zeros else str(first), f"{second:02d}" if zeros else str(second), year])


def _value(field: Locator) -> str:
    try:
        return field.input_value(timeout=2000)
    except PlaywrightError:
        return field.inner_text(timeout=2000)


def _open(field: Locator) -> None:
    """Click the field. When something lies over it (a shade from a calendar opened before, a sticky header), a
    click cannot land: focus it and send the click event directly, which is what opens most pickers."""
    try:
        field.click(timeout=2000)
    except PlaywrightError:
        try:
            field.focus(timeout=2000)
            field.dispatch_event("click")
        except PlaywrightError:
            pass


def pick_date(field: Locator, iso: str, timeout_ms: int = 4000, hint: str = "") -> Picked:
    """Set `field` to the date `iso` (YYYY-MM-DD). Never raises for a picker it does not understand: returns why."""
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return Picked(False, "", "", f"{iso!r} is not a YYYY-MM-DD date")
    page = field.page
    want = {"year": d.year, "month": d.month, "day": d.day}
    try:
        handle = field.element_handle(timeout=timeout_ms)
        frame = handle.owner_frame() if handle else None
        root = frame or page
        found = handle.evaluate(_FIND, want)
        if found.get("what") == "none":  # not open yet: open it the way a person does
            _open(field)
            page.wait_for_timeout(400)
            found = handle.evaluate(_FIND, want)
            if found.get("what") == "none":  # some pickers open a moment later
                page.wait_for_timeout(500)
                found = handle.evaluate(_FIND, want)
        for step in range(MAX_MONTH_STEPS + 2):
            if step:
                found = handle.evaluate(_FIND, want)
            what = found.get("what")
            if what == "day":
                root.locator(f"[{_MARK}=day]").first.click(timeout=timeout_ms)
                page.wait_for_timeout(300)
                value = _value(field)
                if means(value, d):
                    return Picked(True, value, "calendar", f"picked {iso} in the calendar: the field shows {value!r}")
                return Picked(False, value, "", f"clicked day {d.day} in the calendar but the field shows {value!r}")
            if what == "nav":
                root.locator(f"[{_MARK}=nav]").first.click(timeout=timeout_ms)
                page.wait_for_timeout(250)
                continue
            if what in ("no-day", "disabled"):
                return Picked(False, _value(field), "", f"{iso} cannot be chosen in the calendar (shown: {found.get('shown')}; "
                                                         f"{'the day is disabled' if what == 'disabled' else 'no such day'})")
            if what == "no-nav":
                return Picked(False, _value(field), "", f"the calendar shows {found.get('shown')} and has no "
                                                         f"{found.get('dir')} control to reach {iso}")
            break  # no calendar: write it instead
        else:
            return Picked(False, _value(field), "", f"could not reach {iso} in the calendar within {MAX_MONTH_STEPS} months")

        placeholder = field.get_attribute("placeholder", timeout=timeout_ms) or ""
        text = in_format(d, f"{placeholder} {hint}") or iso
        if field.is_editable(timeout=timeout_ms):  # type it as a person would, then leave the field so it is checked
            field.fill(text, timeout=timeout_ms)
            field.press("Tab", timeout=timeout_ms)
        else:
            handle.evaluate(_SET_VALUE, text)
        page.wait_for_timeout(300)
        value = _value(field)
        if means(value, d):
            return Picked(True, value, "typed", f"no calendar opened; wrote {text!r} and the field shows {value!r}")
        return Picked(False, value, "", f"no calendar opened, and writing {text!r} left the field showing {value!r}")
    except PlaywrightError as e:
        return Picked(False, "", "", f"could not set the date: {str(e).splitlines()[0][:160]}")
