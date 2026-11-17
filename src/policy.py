"""Deterministic safety rules. The LLM proposes; this module decides, and it never asks the model.

Three outcomes for a proposed action:
  allowed   -> run it
  blocked   -> never run it (it would leave the allowed site)
  critical  -> do not run it either, but it is not an error: the exploration has reached a point that needs a
               human (a payment, a card field, a password, a destructive button). The explorer records it as
               evidence ("reached the payment step") and ends that branch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .models import Element

# Final, money-moving or irreversible steps. Deliberately narrow: "Book Now" on a result list opens a booking form
# and is a normal step, while "Pay now" / "Confirm booking" is the end of the line for an automated explorer.
FINAL_STEP = re.compile(
    r"\b(pay now|pay (and|&) (book|confirm)|pay\s*[$€£₹]\s?\d|make (a )?payment|submit payment|"
    r"confirm (and pay|booking|order|payment|purchase|reservation)|place (your )?order|"
    r"complete (your |the )?(booking|purchase|payment|order|reservation)|buy now|purchase now|authori[sz]e payment)\b",
    re.I,
)
DESTRUCTIVE = re.compile(
    r"\b(delete|cancel (my |the |this )?(booking|order|reservation|subscription|account)|close (my )?account|deactivate)\b",
    re.I,
)
# Fields a person must fill in themselves. Test data (names, emails, dates) is fine; money and secrets are not.
SENSITIVE_FIELD = re.compile(
    r"(card.?(number|no\b)|credit.?card|debit.?card|\bcvv\b|\bcvc\b|security.?code|expir|\biban\b|"
    r"passw(or)?d|\botp\b|one.?time|\bpin\b)",
    re.I,
)
ACTIONABLE_FOR_FINAL = {"button", "link", "clickable", "menuitem", "tab"}


@dataclass
class Verdict:
    ok: bool = True  # False: blocked, do not run
    critical: bool = False  # True: needs a human, do not run
    reason: str = ""


def _host(url_or_host: str) -> str:
    h = urlparse(url_or_host if "//" in url_or_host else "//" + url_or_host).hostname or ""
    return h.lower().removeprefix("www.")


@dataclass
class Policy:
    start_url: str
    max_actions: int = 40
    allowed_hosts: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.allowed_hosts:
            self.allowed_hosts = {_host(self.start_url)}

    def host_allowed(self, url: str) -> bool:
        if url.startswith(("about:", "chrome-error:", "data:")):
            return True
        h = _host(url)
        return any(h == a or h.endswith("." + a) for a in self.allowed_hosts)

    def check_url(self, url: str) -> Verdict:
        if self.host_allowed(url):
            return Verdict()
        return Verdict(ok=False, reason=f"{_host(url)} is outside the allowed site ({', '.join(sorted(self.allowed_hosts))})")

    def check_click(self, e: Element) -> Verdict:
        if e.role == "link" and e.href and not e.href.startswith("/"):
            # short hrefs carry a host only when it differs from the page's own: "partner.example.com/deals"
            host = _host(e.href.split("/", 1)[0])
            if host and not self.host_allowed(host):
                return Verdict(ok=False, reason=f'the link "{e.display_name}" leads to {host}, outside the allowed site')
        label = f"{e.display_name} {e.description}"
        if e.widget in ACTIONABLE_FOR_FINAL and FINAL_STEP.search(label):
            return Verdict(critical=True, reason=f'"{e.display_name}" is a final payment/booking step: a person must do this')
        if e.widget in ACTIONABLE_FOR_FINAL and DESTRUCTIVE.search(label):
            return Verdict(critical=True, reason=f'"{e.display_name}" is destructive: a person must do this')
        return Verdict()

    def check_type(self, e: Element) -> Verdict:
        hints = " ".join([e.name, e.description, *(l.value for l in e.locators if l.kind in ("css", "placeholder", "label"))])
        if e.input_type == "password" or SENSITIVE_FIELD.search(hints):
            return Verdict(critical=True, reason=f'"{e.display_name}" takes a password, card or other secret: a person must enter it')
        return Verdict()