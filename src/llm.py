"""One LLM client for three OpenAI-compatible providers, chosen by name: gemini (default), groq, openai.

Switching is a name, not a code change:   LLM_PROVIDER=groq   or   test-pilot llm --provider groq
Each provider is a small record (endpoint, key variable, default model, and the two request details where the
providers still differ). Everything else, JSON handling, repair of bad output, retries, a call budget and usage
counting, is shared.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

import openai
from dotenv import find_dotenv, load_dotenv
from pydantic import TypeAdapter, ValidationError

from .text import clip

T = TypeVar("T")


class LLMError(Exception):
    """A problem with the model call that is safe to show the user. `transient` errors (a flaky generation, a timeout,
    a dropped connection) are worth one more try; the others (bad key, unknown model, budget) are not."""
    transient = False


class LLMTransientError(LLMError):
    transient = True


class LLMJsonModeError(LLMTransientError):
    """The provider's own JSON check rejected the reply (Groq: json_validate_failed). `failed_generation` is what the
    model actually produced, which is often usable."""

    def __init__(self, message: str, failed_generation: str = ""):
        super().__init__(message)
        self.failed_generation = failed_generation


class LLMConfigError(LLMError):
    """Missing or invalid configuration."""


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    default_model: str
    token_param: str = "max_tokens"  # OpenAI's reasoning models only accept max_completion_tokens
    temperature: bool = True  # ...and reject any temperature but their default


PROVIDERS: dict[str, Provider] = {
    # Free tier on Google AI Studio (free-tier prompts may be used by Google to improve its products).
    "gemini": Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
                       "gemini-3.8-flash"),
    # Production models on Groq; the Llama 3.x models were retired in 2026.
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b",
                     token_param="max_completion_tokens"),
    "openai": Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-6-luna",
                       token_param="max_completion_tokens", temperature=False),
}
DEFAULT_PROVIDER = "gemini"


def _number(env, name: str, default, cast):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise LLMConfigError(f"{name} must be a number, got {raw!r}") from None


@dataclass
class LLMConfig:
    provider: Provider
    model: str
    api_key: str = field(repr=False)  # never printed
    base_url: str = ""
    max_tokens: int = 4096  # reasoning models spend part of this thinking, so keep it generous
    temperature: float = 0.2
    reasoning_effort: str | None = None  # low | medium | high, sent only when set
    timeout: float = 90.0
    max_retries: int = 4  # rate limits (429) and transient errors, with the provider's retry-after honoured
    max_calls: int | None = None  # call budget for one run

    @classmethod
    def from_env(cls, name: str | None = None, model: str | None = None, env=None) -> "LLMConfig":
        """name > LLM_PROVIDER > gemini.  model > LLM_MODEL > the provider's default."""
        if env is None:
            load_dotenv(find_dotenv(usecwd=True))
            env = os.environ
        pname = (name or env.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        prov = PROVIDERS.get(pname)
        if prov is None:
            raise LLMConfigError(f"Unknown LLM provider {pname!r}. Choose one of: {', '.join(PROVIDERS)} "
                                 "(set LLM_PROVIDER in .env or pass --provider).")
        key = (env.get(prov.key_env) or "").strip()
        if not key:
            raise LLMConfigError(f"No API key for {pname}: set {prov.key_env} in your environment or .env file.")
        effort = (env.get("LLM_REASONING_EFFORT") or "").strip().lower() or None
        if effort not in (None, "none", "minimal", "low", "medium", "high"):
            raise LLMConfigError(f"LLM_REASONING_EFFORT must be none, minimal, low, medium or high, got {effort!r}")
        return cls(
            provider=prov,
            model=(model or env.get("LLM_MODEL") or prov.default_model).strip(),
            api_key=key,
            base_url=(env.get("LLM_BASE_URL") or prov.base_url).strip(),
            max_tokens=_number(env, "LLM_MAX_TOKENS", 4096, int),
            temperature=_number(env, "LLM_TEMPERATURE", 0.2, float),
            reasoning_effort=effort,
            max_calls=_number(env, "LLM_MAX_CALLS", None, int),
        )


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def summary(self) -> str:
        return f"{self.calls} call(s), {self.prompt_tokens} prompt + {self.completion_tokens} completion tokens"


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON out of a model reply that may carry <think> blocks, a code fence or prose around it."""
    t = _FENCE.sub("", _THINK.sub("", text or "").strip()).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = t.find(open_), t.rfind(close)
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except ValueError:
                continue
    raise ValueError("no JSON object found in the reply")


class LLM:
    def __init__(self, config: LLMConfig, client: openai.OpenAI | None = None):
        self.config = config
        self.usage = Usage()
        self._client = client or openai.OpenAI(
            api_key=config.api_key, base_url=config.base_url, timeout=config.timeout, max_retries=config.max_retries)

    # -- plain and JSON chat ---------------------------------------------------------------------------------------
    def chat(self, messages: list[dict], *, json_mode: bool = False, max_tokens: int | None = None) -> str:
        c, p = self.config, self.config.provider
        if c.max_calls is not None and self.usage.calls >= c.max_calls:
            raise LLMError(f"LLM call budget used up ({c.max_calls}); stop and report what was found")
        msgs = list(messages)
        if json_mode and not any("json" in str(m.get("content", "")).lower() for m in msgs):
            msgs.insert(0, {"role": "system", "content": "Reply with a single JSON object and nothing else."})
        kwargs: dict[str, Any] = {"model": c.model, "messages": msgs, p.token_param: max_tokens or c.max_tokens}
        if p.temperature:
            kwargs["temperature"] = c.temperature
        if c.reasoning_effort:
            kwargs["reasoning_effort"] = c.reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self.usage.calls += 1  # counted when attempted: a failing call still spends quota
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as e:
            raise self._wrap(e) from None
        if resp.usage:
            self.usage.prompt_tokens += resp.usage.prompt_tokens or 0
            self.usage.completion_tokens += resp.usage.completion_tokens or 0
        choice = resp.choices[0]
        text = _THINK.sub("", choice.message.content or "").strip()
        if not text and choice.finish_reason == "length":
            raise LLMError(f"{p.name} ran out of tokens before answering (reasoning models think first): "
                           "raise LLM_MAX_TOKENS or set LLM_REASONING_EFFORT=low")
        if not text:
            raise LLMTransientError(f"{p.name} returned an empty reply (finish_reason={choice.finish_reason})")
        return text

    def structured(self, messages: list[dict], schema: type[T] | TypeAdapter, *, max_repairs: int = 1) -> T:
        """Ask for JSON and validate it against a pydantic model.

        Two kinds of slip are handled here so callers do not have to:
          * the provider's own JSON check rejects the reply (Groq's json_validate_failed): the text it generated is
            used if it parses, otherwise the request is repeated once without the strict mode;
          * the reply parses but does not fit the schema: it is sent back once with the problems listed.
        After that it raises LLMError."""
        adapter = schema if isinstance(schema, TypeAdapter) else TypeAdapter(schema)
        msgs = list(messages)
        repairs, json_mode = 0, True
        while True:
            try:
                text = self.chat(msgs, json_mode=json_mode)
            except LLMJsonModeError as e:
                if e.failed_generation:
                    try:
                        return adapter.validate_python(extract_json(e.failed_generation))
                    except (ValueError, ValidationError):
                        pass
                if json_mode:
                    json_mode = False
                    continue
                raise
            try:
                return adapter.validate_python(extract_json(text))
            except (ValueError, ValidationError) as e:
                problem = clip(str(e).replace("\n", " "), 400)
                if repairs >= max_repairs:
                    raise LLMError(f"{self.config.provider.name} did not return valid JSON for this request: {problem}") from None
                repairs += 1
                msgs = [*msgs, {"role": "assistant", "content": text},
                        {"role": "user", "content": f"That reply was not valid: {problem}. "
                                                    "Reply again with only the corrected JSON object."}]

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except openai.OpenAIError as e:
            raise self._wrap(e) from None

    # -- errors in plain words -------------------------------------------------------------------------------------
    @staticmethod
    def _provider_message(e: openai.OpenAIError) -> str:
        """The provider's own sentence. Bodies come as {"error": {"message": ...}}, {"message": ...} or, from Gemini,
        a list of those; the SDK's own string prefixes "Error code: 400 - ", which is noise to the reader."""
        body = getattr(e, "body", None)
        if isinstance(body, list) and body:
            body = body[0]
        if isinstance(body, dict):
            inner = body.get("error")
            msg = body.get("message") or (inner.get("message") if isinstance(inner, dict) else None)
            if msg:
                return str(msg)
        return re.sub(r"^Error code: \d+ - ", "", str(getattr(e, "message", e)))

    @staticmethod
    def _error_object(e: openai.OpenAIError) -> dict:
        body = getattr(e, "body", None)
        if isinstance(body, list) and body:
            body = body[0]
        if not isinstance(body, dict):
            return {}
        return body["error"] if isinstance(body.get("error"), dict) else body

    def _wrap(self, e: openai.OpenAIError) -> LLMError:
        message = self._explain(e)
        obj = self._error_object(e)
        if isinstance(e, openai.BadRequestError) and (
                obj.get("code") == "json_validate_failed" or "failed to validate json" in message.lower()):
            return LLMJsonModeError(message, str(obj.get("failed_generation") or ""))
        if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)):
            return LLMTransientError(message)
        return LLMError(message)

    def _explain(self, e: openai.OpenAIError) -> str:
        c, p = self.config, self.config.provider
        if isinstance(e, openai.AuthenticationError):
            return f"{p.name} rejected the API key: check {p.key_env}"
        if isinstance(e, openai.PermissionDeniedError):
            return f"{p.name} refused this request (permission denied): check the key's access to {c.model}"
        if isinstance(e, openai.NotFoundError):
            return (f"{p.name} has no model {c.model!r}: set LLM_MODEL, or run "
                    f"`test-pilot llm --provider {p.name} --models` to see what your key can use")
        if isinstance(e, openai.RateLimitError):
            return (f"{p.name} is rate-limiting this key (still after {c.max_retries} retries): wait a minute, "
                    "lower the request rate, or switch provider with LLM_PROVIDER")
        if isinstance(e, openai.APITimeoutError):
            return f"{p.name} did not answer within {c.timeout:.0f}s"
        if isinstance(e, openai.APIConnectionError):
            return f"could not reach {p.name} at {c.base_url}: check the network"
        if isinstance(e, openai.BadRequestError):
            return f"{p.name} rejected the request: {clip(self._provider_message(e), 300)}"
        return f"{p.name} error: {clip(self._provider_message(e), 300)}"