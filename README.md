# Test-Pilot

Turns a live page into the text an LLM reads, plus stable locators for later test generation. No LLM involved yet.

## Run

    pip install -e ".[dev]" && python -m playwright install chromium
    python -m test-pilot.cli observe https://example.com --locators
    TEST_PILOT_CDP_URL=http://host:9222 python -m test-pilot.cli observe <url>     # remote browser
    pytest

## What the model sees

    [state] SkyBook - Flights | http://.../
    ## Book a flight
    [25] textbox "From" required (hint: "Enter an IATA code or a city")
    [36] date "Departure date" required
    [47] select "Cabin" = "Economy"  options: Economy | Premium economy | Business
    [69] textbox "Account password" type=password = "(hidden)"
    [74] button "Search flights"
    [75] button "Book now" disabled
    --- frame http://.../frame.html ---
    [115] textbox "Offer code"
    [/state]

`[id]` is Chrome's backendNodeId (valid for this page only). `*` marks controls new since the last look.

## Design

| Module | Job |
|---|---|
| `perceive.py` | AX tree + DOM attributes + visible-text probe -> `Observation` (text only, no screenshots) |
| `watch.py` | event-based settle (DOM quiet + network idle, no sleeps) and "navigated / page changed / no change" |
| `locators.py` | ordered candidates (role, label, placeholder, testid, text, css), live uniqueness check, dynamic-id filter |
| `render.py` | Observation -> prompt text, head+tail truncation |
| `browser.py` | local Chromium or remote CDP, URL validation, dialog handling |
| `models.py` | pydantic models, page-type `signature()` for the site graph |

Ideas adapted from Browser Use's `ax.py` (backendNodeId refs, visible-text filtering, native select/date handling, new-element marks);
the code is a fresh implementation on Playwright's CDP session.

## Known limits (deliberate, for now)

- Cross-origin iframes are not read (their controls are missing from the AX tree we can reach).
- Hidden file inputs and custom (non-native) date pickers need handling in the action step.
- `DOM.getDocument(pierce)` is one call per look; very large DOMs will be slower.
- Dialogs are auto-accepted (`dialog_policy`); the policy layer should veto destructive confirms.
- `signature()` is coarse; refine it when the site graph is built.

## Next: actions (`click`, `type`, `select`) + safety policy, reusing the same ids and the settle/change helpers.