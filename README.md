# Test-Pilot

AI test-case generator for web apps. Give it a URL and a requirement (for example, "book a flight from Pune to Delhi"), and it explores the app in a real browser, records replayable Playwright steps and browser evidence, and produces a JSON exploration report for later test design and execution.

## Status

Test-Pilot is an early-stage working implementation, not just a proof of concept. It can:

- perceive and summarize a live page;
- act safely through Playwright;
- call Gemini, Groq, OpenAI or OpenRouter models for structured decisions;
- turn a requirement into a short exploration plan;
- run a bounded browser exploration and collect replayable steps and evidence;
- write a timestamped JSON exploration report;
- compare observations to capture navigation, text changes, alerts and field validation errors.

The test designer, compiled Playwright test suite and Pass/Fail runner are still in progress.

## How it works

```
requirement + URL
      │
      ▼
 [analyzer] ──► requirement points, assumptions, areas to probe
      │
      ▼
 [explorer]  perceive ─► LLM action ─► policy check ─► act ─► evidence diff
      │                           ▲                         │
      └──── history / stuck checks ────────────────────────┘
      │
      ▼
 ExplorationReport ──► JSON report + replayable Playwright steps
      │
      ▼
 [designer / compiler / runner]  (planned)
```

Design choices:

- **Text only, no vision.** The model receives the accessibility tree and DOM-derived controls, headings and visible text. It does not receive screenshots.
- **Backend-node ids are short-lived refs.** The model chooses an id from the latest observation. The browser verifies that id against a live locator before acting.
- **The model acts on ids, while evidence stays replayable.** Successful actions are recorded as `Step`s and rendered as Playwright Python code.
- **Safety is plain code.** Policy checks precede actions; the model cannot override payment, password, card or host restrictions.
- **Evidence is observation-derived.** Reports and summaries are based on recorded browser changes, not model claims.
- **Remote browsers are supported.** Local Chromium is the default; CDP can connect to an existing Chromium instance.

## Quick start

Requires Python 3.11+ and a Playwright Chromium installation.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
# Add a key for Gemini, Groq, OpenAI or OpenRouter, then:
test-pilot llm
test-pilot observe https://phptravels.net/ --locators
test-pilot explore https://phptravels.net/ "Search for a flight from Pune to Delhi"
```

## Configuration

Copy `.env.example` to `.env`. Keep secrets out of source control.

| Variable | Purpose |
|---|---|
| `TEST_PILOT_CDP_URL` | CDP endpoint of an existing Chromium. Leave empty to launch Chromium locally. |
| `LLM_PROVIDER` | `gemini` (default), `groq`, `openai`, or `openrouter`. |
| `LLM_MODEL` | Optional provider model override. |
| `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY` | API key for the selected provider. |
| `LLM_MAX_TOKENS` | Maximum response tokens; default `4096`. |
| `LLM_REASONING_EFFORT` | Optional `none`, `minimal`, `low`, `medium`, or `high`. |
| `LLM_TEMPERATURE` | Sampling temperature; ignored by OpenAI reasoning models. |
| `LLM_MAX_CALLS` | Optional model-call budget for a run. |
| `LLM_BASE_URL` | Optional OpenAI-compatible endpoint override. |

## Command line

```bash
test-pilot observe <url> [--headed] [--cdp-url URL] [--max-chars N] [--full] [--json] [--locators]
test-pilot llm [--provider gemini|groq|openai|openrouter] [--model MODEL] [--models]
test-pilot explore <url> "<requirement>" [--provider PROVIDER] [--model MODEL] [--max-steps N] [--max-seconds N] [--out PATH] [--headed] [--cdp-url URL] [--quiet]
test-pilot design <report.json> [--provider PROVIDER] [--model MODEL] [--out PATH]
test-pilot verify <cases.json> [--case TC-NN] [-c N] [--provider PROVIDER] [--model MODEL] [--max-seconds N] [--out PATH] [--headed] [--cdp-url URL]
```

- `observe` prints the text the model would receive. `--locators` verifies one Playwright locator per control.
- `llm` validates the provider configuration with a tiny structured call. `--models` lists models accessible to the configured key.
- `explore` runs the bounded LLM/browser loop and writes an `ExplorationReport` JSON file under `artifacts/` by default.
- `design` turns an exploration report into 5–10 structured test cases (`<report>.cases.json`) with one model call, then checks each case against the recorded evidence in code. A step must repeat a successful recorded step or name a control that was seen. An expected outcome counts as observed only if its text appears in what the browser recorded. Each case is marked `observed`, `partial` or `unverified` with the reasons, and requirement points no case covers are listed as gaps.
- `verify` runs each test case in its own fresh browser context, one at a time by default. `-c N` runs N cases at once (watch free-tier rate limits), and `--case TC-NN` runs just that case. It writes `<report>.verified.json`:
  - **Recorded steps** are replayed without the model.
  - **Other steps** are carried out by the model through the same safety policy as the explorer, in at most 6 actions per step.
  - **Expected outcomes** are checked on the live page, including the values fields hold. A validation message the designer only guessed is replaced by the page's real message, whether that shows on the field, in an alert or as a toast.
  - **Each case** ends `verified`, `unconfirmed`, `failed` or `blocked`, with the steps that worked and the checks that held. Those are the input for the test compiler. The model never writes code.

## Using it from Python

```python
from test_pilot import BrowserSession
from test_pilot.actions import Actor, Click, Type
from test_pilot.policy import Policy

url = "https://phptravels.net/"
with BrowserSession() as session:
    session.goto(url)
    actor = Actor(session, Policy(start_url=url))
    print(actor.look())
    result = actor.act(Click(id=89))
    print(result.status, result.message)
    print([step.code() for step in actor.steps])
```

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

## Tests

Use the project environment if packages are not on your shell's default `PATH`:

```bash
pytest -q
# or
conda run -n tp pytest -q
```

The suite runs against local fixture sites and a standalone Chromium over CDP. No external internet is needed by the fixtures. Fixtures live in `tests/fixtures/`; `flight.html` and `hotels.html` cover form behavior, `portal.html` covers div widgets and modal/duplicate-card cases, and `checkout.html` covers payment, destructive and off-site cases.

The suite currently contains 198 tests. The latest local run has 195 passing and 3 failing perception/action tests that are being investigated; see `pytest -q` for details.

## Known limitations

- **No test compiler or runner yet.** Exploration reports contain replayable steps and evidence, but Playwright test suites are not generated or executed here yet.
- **Cross-origin iframes** are not read.
- **Custom calendars and pickers** rely on the model opening the widget and choosing individual controls. They are most reliable for native date inputs.
- **`.nth()` locators** depend on page order and are marked fragile.
- **Heuristics** can misfire. Blocking dialogs, icon text and custom clickable widgets depend on accessible roles, z-indexes, listeners and DOM structure.
- **Login and CAPTCHA** are not handled. The policy stops at passwords and human-only steps.
- **Model summaries are diagnostic only.** The report's primary summary is rebuilt from browser events and observed validation messages.
- **LLM usage** depends on provider availability and API credentials; browser fixture tests do not require an LLM.

## Development log

1. **Perception core:** accessibility-tree text with ids, DOM attributes, visible-text filtering, native select/date handling, password masking, settle/change detection and unique locators.
2. **Real-site resilience:** custom `clickable` widgets, blocking-dialog hoisting, repeated-card filtering, token-budget tiers and view-only prompts.
3. **Locator improvements:** icon-font name cleaning, icon-free locator variants, visible filtering and marked `.nth()` fallbacks.
4. **Actions and safety:** `Actor` with click/type/select/press/scroll/wait/go-back; allowed-host and sensitive-action policy; replayable `Step.code()`.
5. **Evidence extraction:** observation diffs for navigation, appeared/disappeared text, alerts and native validation messages.
6. **LLM support:** one OpenAI-compatible client for Gemini, Groq, OpenAI and OpenRouter, with structured JSON repair, retries, call budgets and usage accounting.
7. **Requirement analysis and exploration:** bounded end-to-end browser loops, valid-first behavior, validation probing, stuck/failure/time/action budgets, deterministic reports and JSON artifacts.

## Roadmap

1. **Test designer:** 4–5 valid and invalid cases with assertions in a constrained schema.
2. **Validator and compiler:** verify locators and assertions against collected evidence, then render Playwright tests with relative dates.
3. **Runner and report:** Passed, Failed or Blocked outcomes with expected/actual results, screenshots and traces.
4. **Optional:** per-site graph caching for faster repeat runs and Kubernetes deployment.

## Credits

The perception approach follows ideas from Browser Use's `ax.py` (backend-node ids as element refs, visible-text filtering, native select/date handling and new-element marks). The implementation is separate and uses Playwright's CDP session.