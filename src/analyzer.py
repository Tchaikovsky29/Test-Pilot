"""Requirement analyzer: one model call that turns "Book a flight from Pune to Delhi" into atomic, testable points,
the assumptions the explorer will make, and the validation behaviour worth probing. It is advisory: if the call
fails, exploration still runs on the raw requirement."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .llm import LLMError


class Plan(BaseModel):
    requirement_points: list[str] = Field(default_factory=list)  # atomic things the requirement asks for
    assumptions: list[str] = Field(default_factory=list)  # defaults chosen where the requirement is silent
    areas_to_probe: list[str] = Field(default_factory=list)  # validation and edge behaviour worth observing


SYSTEM = """You are a senior QA analyst. Given a web application URL and a requirement, break the requirement into a short \
plan for exploring the application. Reply with JSON only, in this shape:
{"requirement_points": ["..."], "assumptions": ["..."], "areas_to_probe": ["..."]}

- requirement_points: 2 to 6 atomic, testable statements of what the requirement asks for (for example "search for flights \
from Pune to Delhi", "see a list of matching flights").
- assumptions: defaults you will use where the requirement is silent (for example "one adult, economy, one way", "a date \
about two weeks from today").
- areas_to_probe: 3 to 6 validation or edge behaviours a tester would check on the screens this requirement involves \
(for example "search with an empty destination", "return date before departure date", "same origin and destination").
Be concrete and short. Do not invent features the requirement does not suggest.

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