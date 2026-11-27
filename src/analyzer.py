"""Requirement analyzer: one model call that turns "Book a flight from Pune to Delhi" into atomic, testable points,
the assumptions the explorer will make, and the validation behaviour worth probing. It is advisory: if the call
fails, exploration still runs on the raw requirement."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .llm import LLMError


class Plan(BaseModel):
    requirement_points: list[str] = Field(default_factory=list)  # atomic things the requirement asks for
    outcomes: list[str] = Field(default_factory=list)  # what must be true afterwards: the business result to verify
    assumptions: list[str] = Field(default_factory=list)  # defaults chosen where the requirement is silent
    areas_to_probe: list[str] = Field(default_factory=list)  # validation and edge behaviour worth observing


SYSTEM = """You are a senior QA analyst. Given a web application URL and a requirement, break the requirement into a short \
plan for exploring the application. Reply with JSON only, in this shape:
{"requirement_points": ["..."], "outcomes": ["..."], "assumptions": ["..."], "areas_to_probe": ["..."]}

- requirement_points: 2 to 6 atomic, testable statements of what the requirement asks for (for example "search for a product \
under $50", "open the matching result and view its details").
- outcomes: 1 to 4 business results that must be true afterwards and can be checked on the screens, not steps \
(for example "a record exists with the entered name, company and email", "the new item appears in the list", \
"the form is not submitted and the missing field is reported"). Test outcomes, not clicks.
- assumptions: defaults you will use where the requirement is silent (for example "some default value", "another default value").
- areas_to_probe: 3 to 6 validation or edge behaviours a tester would check on the screens this requirement involves \
(for example "submit an empty required field", "enter a value outside a listed range", "submit a duplicate value").
Be concrete and short. Do not invent features the requirement does not suggest, and do not assume which fields are \
required: that is for the application to show.

Scope: the explorer works in a browser and never enters payment details, passwords or one-time codes, never completes a \
payment or final booking, and cannot read email. So plan only up to the step just before payment (for example "reach the \
passenger details or payment page"), and list only behaviour that can be seen on the screens without paying: no payment \
failures, no confirmation emails, no accounts that must already exist."""


def analyze(llm, requirement: str, url: str) -> tuple[Plan, str]:
    """Returns (plan, problem). `problem` is empty on success; on failure the plan is just the raw requirement."""
    try:
        plan = llm.structured(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"Application: {url}\nRequirement: {requirement}"}],
            Plan,
        )
        if not plan.requirement_points:
            plan.requirement_points = [requirement]
        return plan, ""
    except LLMError as e:
        return Plan(requirement_points=[requirement]), str(e)