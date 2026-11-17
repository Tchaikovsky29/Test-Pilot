# Test-Pilot

AI test-case generator for web apps. You give it a URL and a requirement (for example, "book a flight from Pune to Delhi"). It explores the app in a real browser, designs valid and invalid test cases, and turns them into runnable Playwright tests with a Pass/Fail result.

> **Status: proof of concept.** The browser layer (perception, actions, safety policy) is built and tested. The LLM explorer loop, test designer, validator and report are next. See the [roadmap](#roadmap).

## How it works

```
requirement + URL
      │
      ▼
 [analyzer] ──► requirement points, exploration plan                       (planned)
      │
      ▼
 [explorer]  perceive ─► LLM picks one action ─► policy check ─► act       (perceive, act, policy: built)
      │                                                                    (LLM loop: planned)
      ▼
 [designer]  4–5 test cases, valid and invalid, each with an assertion     (planned)
      │
      ▼
 [validator] every locator and assertion checked against what was seen     (locator check: built)
      │
      ▼
 [compiler] ─► Playwright tests ─► [runner] ─► Pass / Fail + evidence      (planned; Step.code() built)
```

Design choices:

- **Text only, no vision.** The model reads the accessibility tree plus DOM attributes, which is cheap and works with free-tier models.
- **The model acts on ids, tests replay locators.** Every action resolves to a unique Playwright locator on the live page, and that same locator is what a generated test will use. A step that worked during exploration will replay.
- **Safety is plain code.** A policy module decides what may run. It never asks the model.
- **Remote browsers are supported.** Connect to any Chromium over CDP.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
test-pilot observe https://phptravels.net/ --locators
```

## Configuration

Copy `.env.example` to `.env` (it must stay out of git).

| Variable | Purpose |
|---|---|
| `TEST_PILOT_CDP_URL` | CDP endpoint of a remote Chromium. Leave empty to launch one locally. |
| `LLM_PROVIDER`, `LLM_MODEL`, `GEMINI_API_KEY`, `GROQ_API_KEY` | Reserved for the explorer loop (commented out for now). |

## Command line

```bash
test-pilot observe <url> [--headed] [--cdp-url URL] [--max-chars N] [--json] [--locators]
```

Prints the page exactly as the model will see it. `--locators` also prints a verified, unique Playwright locator for each control and marks `.nth()` fallbacks as fragile. `--json` prints the full `Observation`.

## Using it from Python

```python
from test_pilot import BrowserSession
from test_pilot.actions import Actor, Click, Type
from test_pilot.policy import Policy

url = "https://phptravels.net/"
with BrowserSession() as s:
    s.goto(url)
    actor = Actor(s, Policy(start_url=url))
    print(actor.look())                          # the page as the model sees it
    result = actor.act(Click(id=89))             # ids come from the last look
    print(result.status, result.message)         # ok | failed | blocked | stopped
    print([step.code() for step in actor.steps]) # Playwright code for what was done
```
As given in tests/try.py

## What the model sees

One line per control, in page order. `[id]` is Chrome's backend node id and is valid for the current page only. `*` marks controls that are new since the last look.

```
[state] PHPTRAVELS | https://phptravels.net/
--- dialog blocking the page: handle it first ---
## Testing Environment Only - NO Real Payments
This is a testing platform for development purposes. Do not use real payment methods…
[89] button "I Understand & Continue"
--- end of dialog ---
## Travel The Way You Love!
[627] tab "Flights" selected
[633] tab "Stays"
[696] button "Economy"
[57] checkbox "Flexible Dates +/- 7 Days" unchecked
[58] clickable "Departure From"
[59] clickable "Arrival To"
[61] textbox "Departure Date" readonly
[801] clickable "Passengers 1 Passenger"
[859] button "Search Flights"
```

What the perception layer does:

- **Sources.** Accessibility tree (main frame and same-process iframes), DOM attributes (including `<select>` options and shadow DOM), and a visible-text probe.
- **Hidden text is dropped.** `opacity:0` and `display:none` text, such as a pre-rendered "Booking confirmed!" banner, never reaches the model.
- **Widgets.** Native `select` shows its options, native date inputs show as `date`, and `required`, `readonly`, `disabled`, `checked` and `invalid` states are shown.
- **Div widgets without accessibility roles** (custom tabs, dropdowns, date-picker triggers) appear as `clickable`. They are found from click listeners and `cursor:pointer`, using Chrome's DOM snapshot.
- **Blocking dialogs come first.** A modal is detected from the element at the centre of the viewport and is shown first, even if it is last in the document. `Observation.blocked` is set.
- **Secrets are masked.** Password values show as `(hidden)` and never enter the observation, the JSON or the logs.
- **Icon-font noise is cleaned.** `hotel Stays` is shown as `Stays` and `Economy expand_more` as `Economy`. Locators keep the raw name and add an icon-free, non-exact variant.
- **Waiting without sleeps.** The page is settled when the DOM has been quiet and no recent request is pending.
- **Token budget.** Long prose is clipped or dropped first, so controls, headings and short labels survive. A blocking dialog's text is never dropped.

## Actions

| Action | Notes |
|---|---|
| `click(id)` | Works on links, buttons, checkboxes and `clickable` widgets |
| `type(id, text, keys=False, submit=False)` | `keys=True` types real key presses for autocomplete boxes. Native date fields need `YYYY-MM-DD`. |
| `select(id, option)` | Native `<select>`, case-insensitive. A bad option returns the list of valid ones. |
| `press(key)`, `scroll(direction)`, `wait(ms)`, `go_back()` | |

Each action returns an `ActionResult` with one of four statuses:

- `ok`: it ran. The message says `page changed`, `navigated to <url>` or `no change`.
- `failed`: it did not run, with a plain-sentence reason, for example `it is covered by <div id="modal">; a dialog is blocking the page, so use the dialog's own controls first`.
- `blocked`: it would leave the allowed site.
- `stopped`: it needs a person (see below).

Successful actions are recorded as `Step`s. `Step.code()` returns the Playwright line for each one.

## Safety policy

Decisions are made by `policy.py` before anything runs:

- **Allowed host.** Links to other hosts are refused before clicking. If a script navigates away, the action is undone with a go-back.
- **Stop for a person.** The explorer does not click final payment or booking buttons (`Pay now`, `Confirm booking`, `Place order`) or destructive ones (`Delete`, `Cancel booking`), and it does not type into password, card, CVV, expiry, IBAN, OTP or PIN fields. "Book Now" on a result list is allowed, because it opens a form.
- **Budget.** Maximum actions per run (default 40).
- **Dialogs.** Browser alerts and confirms are auto-accepted and reported in the next observation.

These are regex and structure rules, deliberately simple and readable. Edit `FINAL_STEP`, `DESTRUCTIVE` and `SENSITIVE_FIELD` in `policy.py` to fit your target app.

## Locators

For each element the candidates are tried in this order: role + name, label, placeholder, test id, text, then CSS (`id`, `name`, `data-test*`) as a last resort. Machine-generated ids (`promo-8f3a9c21d4e5`, `:r1:`, `ui-id-3`) are never used. Each candidate is counted on the live page, and the first unique one wins. For duplicates the order is: narrow with `.filter(visible=True)`, then `.nth(i)` marked fragile.

## Project layout

Files live directly in `src/` and are installed as the package `test_pilot` (see `[tool.setuptools.package-dir]` in `pyproject.toml`).

| File | Job |
|---|---|
| `perceive.py` | Accessibility tree + DOM + visible text → `Observation`; clickables; dialog hoisting |
| `watch.py` | DOM and network settling, and "navigated / page changed / no change" |
| `locators.py` | Candidates, live uniqueness check, visible and `.nth` fallbacks (`ensure_locator`) |
| `render.py` | Observation → prompt text, budget tiers, head+tail truncation |
| `browser.py` | Local or remote (CDP) Chromium, URL validation, dialog handling |
| `actions.py` | Action models, `Actor`, `Step`, `ActionResult` |
| `policy.py` | Allowed host, final-step and sensitive-field rules, action budget |
| `models.py` | Pydantic models (`Observation`, `Element`, `Locator`) and the page-type `signature()` |
| `roles.py`, `text.py` | Role tables and text helpers (icon cleaning, quoting) |
| `cli.py` | `test-pilot observe` |

## Tests

```bash
pytest
```

The suite (84 tests at this milestone) runs against local fixture sites and a standalone Chromium over CDP. No internet is needed. Fixtures live in `tests/fixtures/`. `flight.html` and `hotels.html` are simple forms, `portal.html` reproduces what was found on a real travel site (div widgets, a modal last in the DOM, duplicate buttons, repeated cards), and `checkout.html` covers payment, delete and off-site cases.

## Known limitations

- **Cross-origin iframes** are not read.
- **Custom calendars and pickers** are operated through clicks (open it, look again, click the day). Not yet verified end to end on a real site.
- **`.nth()` locators** depend on page order and are marked fragile.
- **Heuristics** can misfire. Examples: a lone lowercase word of four or more letters is treated as an icon name, clickables come from listeners and `cursor:pointer`, and a blocking dialog needs a z-index or `role=dialog` to be detected.
- **Login and CAPTCHA** are not handled. The policy stops at password fields.
- **No LLM yet.** The explorer loop, designer, validator and runner are not built.

## Development log

1. **Perception core.** Accessibility-tree text with ids, DOM attributes, a visible-text filter, native select and date handling, password masking, remote Chromium over CDP, settle and change detection, unique-locator check, 31 tests.
2. **First real-site fixes** (phptravels): `clickable` detection for div widgets, blocking-dialog hoisting, no repeated card text, budget tiers that drop prose before controls, `.filter(visible=True)` and `.nth()` locator fallbacks.
3. **Second real-site fixes:** icon-font name cleaning (leading and trailing) with icon-free locators, dialog prose never dropped, one-line `ensure_locator` for single elements.
4. **Actions and safety:** `Actor` with click, type, select, press, scroll, wait and go-back; `Policy` with host scope, final-step and sensitive-field stops, budget; `Step.code()` that replays; 84 tests.

## Roadmap

1. **Evidence extraction:** an observation diff (new headings and text after an action), native `validationMessage` capture, and form schemas built from observations.
2. **Explorer loop:** LLM client, requirement analyzer, bounded loop with stuck detection and budgets.
3. **Test designer:** 4–5 valid and invalid cases with assertions, in a constrained JSON format.
4. **Validator and compiler:** every locator and assertion checked against what was seen, then rendered as Playwright tests with relative dates.
5. **Runner and report:** Passed, Failed or Blocked with expected and actual results, plus screenshots and traces.
6. **Optional:** a per-site graph cache for faster repeat runs, and Kubernetes deployment.

## Credits

The perception approach follows ideas from Browser Use's `ax.py` (backend node ids as element refs, visible-text filtering, native select and date handling, new-element marks). The code is a separate implementation on Playwright's CDP session.