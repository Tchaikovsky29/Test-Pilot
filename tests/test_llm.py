"""The LLM client against a fake OpenAI-compatible server: what is sent, what is made of the reply, how failures read."""
import http.server
import json
import threading

import pytest
from pydantic import BaseModel

from test_pilot.cli import main as cli_main
from test_pilot.llm import LLM, PROVIDERS, LLMConfig, LLMConfigError, LLMError, LLMTransientError, extract_json


# ---- a fake server ----------------------------------------------------------------------------------------------

class Fake:
    def __init__(self):
        self.script: list[dict] = []   # one entry per expected call: {"content": ...} or {"status": 429, ...}
        self.requests: list[dict] = []
        self.models = ["alpha", "beta"]


@pytest.fixture()
def fake():
    state = Fake()

    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, status, body, headers=()):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.endswith("/models"):
                return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in state.models]})
            self._send(404, {"error": {"message": "nope"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            step = state.script.pop(0) if state.script else {"content": "{}"}
            status = step.get("status", 200)
            if status != 200:
                err = step.get("error") or {"message": step.get("message", "boom"), "type": "x"}
                return self._send(status, {"error": err}, [("retry-after-ms", "20")])
            self._send(200, {
                "id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": step.get("finish", "stop"),
                             "message": {"role": "assistant", "content": step.get("content")}}],
                "usage": {"prompt_tokens": step.get("pt", 10), "completion_tokens": step.get("ct", 5), "total_tokens": 15},
            })

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    yield state
    srv.shutdown()


def make(fake, provider="gemini", **over):
    env = {"GEMINI_API_KEY": "gk", "GROQ_API_KEY": "qk", "OPENAI_API_KEY": "ok", "LLM_BASE_URL": fake.url,
           "LLM_PROVIDER": provider, "LLM_MAX_TOKENS": "321", **over}
    cfg = LLMConfig.from_env(env=env)
    cfg.max_retries = 1
    return LLM(cfg)


class Plan(BaseModel):
    goal: str
    steps: list[str]


MSG = [{"role": "user", "content": "plan as json"}]


# ---- choosing a provider by name -------------------------------------------------------------------------------

def test_gemini_is_the_default():
    c = LLMConfig.from_env(env={"GEMINI_API_KEY": "k"})
    assert c.provider.name == "gemini" and c.model == "gemini-3.8-flash"
    assert c.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_switching_is_a_name():
    env = {"GEMINI_API_KEY": "g", "GROQ_API_KEY": "q", "OPENAI_API_KEY": "o"}
    assert LLMConfig.from_env(env={**env, "LLM_PROVIDER": "groq"}).model == "openai/gpt-oss-120b"
    assert LLMConfig.from_env(env={**env, "LLM_PROVIDER": "openai"}).model == "gpt-6-luna"
    assert LLMConfig.from_env("openai", env={**env, "LLM_PROVIDER": "groq"}).provider.name == "openai"   # name wins
    assert LLMConfig.from_env("groq", model="m", env={**env, "LLM_MODEL": "other"}).model == "m"        # arg > env
    assert LLMConfig.from_env(env={**env, "LLM_MODEL": "other"}).model == "other"
    assert LLMConfig.from_env(env={**env, "LLM_PROVIDER": " GROQ "}).provider.name == "groq"             # forgiving


def test_unknown_provider_and_missing_key_are_explained():
    with pytest.raises(LLMConfigError, match="gemini, groq, openai"):
        LLMConfig.from_env("claude", env={})
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        LLMConfig.from_env("groq", env={"GEMINI_API_KEY": "k"})
    with pytest.raises(LLMConfigError, match="LLM_MAX_TOKENS"):
        LLMConfig.from_env(env={"GEMINI_API_KEY": "k", "LLM_MAX_TOKENS": "lots"})
    with pytest.raises(LLMConfigError, match="LLM_REASONING_EFFORT"):
        LLMConfig.from_env(env={"GEMINI_API_KEY": "k", "LLM_REASONING_EFFORT": "extreme"})


def test_the_key_is_never_printed():
    c = LLMConfig.from_env(env={"GEMINI_API_KEY": "sk-very-secret"})
    assert "sk-very-secret" not in repr(c) and "sk-very-secret" not in str(c)


# ---- what is sent -----------------------------------------------------------------------------------------------

def sent(fake):
    return fake.requests[-1]["body"]


def test_gemini_request_shape(fake):
    fake.script = [{"content": "hi"}]
    make(fake, "gemini").chat(MSG)
    b = sent(fake)
    assert fake.requests[-1]["auth"] == "Bearer gk" and fake.requests[-1]["path"].endswith("/chat/completions")
    assert b["model"] == "gemini-3.8-flash" and b["max_tokens"] == 321 and b["temperature"] == 0.2
    assert "max_completion_tokens" not in b and "response_format" not in b and "reasoning_effort" not in b


def test_groq_uses_max_completion_tokens_and_keeps_temperature(fake):
    fake.script = [{"content": "hi"}]
    make(fake, "groq").chat(MSG)
    b = sent(fake)
    assert fake.requests[-1]["auth"] == "Bearer qk"
    assert b["model"] == "openai/gpt-oss-120b" and b["max_completion_tokens"] == 321 and b["temperature"] == 0.2
    assert "max_tokens" not in b


def test_openai_reasoning_model_gets_no_temperature(fake):
    fake.script = [{"content": "hi"}]
    make(fake, "openai").chat(MSG)
    b = sent(fake)
    assert fake.requests[-1]["auth"] == "Bearer ok"
    assert b["model"] == "gpt-6-luna" and b["max_completion_tokens"] == 321
    assert "temperature" not in b and "max_tokens" not in b


def test_reasoning_effort_only_when_asked(fake):
    fake.script = [{"content": "hi"}]
    make(fake, "openai", LLM_REASONING_EFFORT="low").chat(MSG)
    assert sent(fake)["reasoning_effort"] == "low"


def test_json_mode_sets_response_format_and_adds_the_word_json_if_missing(fake):
    fake.script = [{"content": "{}"}, {"content": "{}"}]
    llm = make(fake)
    llm.chat([{"role": "user", "content": "plan the trip"}], json_mode=True)
    b = sent(fake)
    assert b["response_format"] == {"type": "json_object"} and "json" in b["messages"][0]["content"].lower()
    llm.chat(MSG, json_mode=True)
    assert len(sent(fake)["messages"]) == 1                        # already mentions json: nothing added


# ---- making sense of replies ----------------------------------------------------------------------------------

@pytest.mark.parametrize("reply", [
    '{"goal": "g", "steps": ["a"]}',
    '```json\n{"goal": "g", "steps": ["a"]}\n```',
    '<think>let me plan {not json}</think>{"goal": "g", "steps": ["a"]}',
    'Sure! Here you go:\n{"goal": "g", "steps": ["a"]}\nHope that helps.',
])
def test_structured_accepts_the_usual_wrappers(fake, reply):
    fake.script = [{"content": reply}]
    assert make(fake).structured(MSG, Plan) == Plan(goal="g", steps=["a"])


def test_a_bad_reply_is_sent_back_once_with_the_problem(fake):
    fake.script = [{"content": '{"goal": "g"}'}, {"content": '{"goal": "g", "steps": ["a"]}'}]
    assert make(fake).structured(MSG, Plan).steps == ["a"]
    assert len(fake.requests) == 2
    retry = sent(fake)["messages"]
    assert retry[-2]["role"] == "assistant" and "steps" in retry[-1]["content"]       # told what was wrong


def test_it_gives_up_after_the_repair(fake):
    fake.script = [{"content": "not json"}, {"content": "still not json"}]
    with pytest.raises(LLMError, match="did not return valid JSON"):
        make(fake).structured(MSG, Plan)
    assert len(fake.requests) == 2


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}), ('[1, 2]', [1, 2]), ('```\n{"a": 1}\n```', {"a": 1}),
    ('<think>x</think> {"a": {"b": 2}} trailing', {"a": {"b": 2}}),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no braces here")


def test_think_blocks_are_removed_from_plain_chat(fake):
    fake.script = [{"content": "<think>hmm</think>The answer"}]
    assert make(fake).chat(MSG) == "The answer"


# ---- failures in plain words ------------------------------------------------------------------------------------

def test_bad_key(fake):
    fake.script = [{"status": 401}]
    with pytest.raises(LLMError, match="rejected the API key: check GEMINI_API_KEY"):
        make(fake).chat(MSG)


def test_unknown_model_points_to_the_model_list(fake):
    fake.script = [{"status": 404}]
    with pytest.raises(LLMError, match=r"no model 'nope'.*--models"):
        make(fake, LLM_MODEL="nope").chat(MSG)


def test_rate_limit_is_retried_then_succeeds(fake):
    fake.script = [{"status": 429}, {"content": "ok"}]
    assert make(fake).chat(MSG) == "ok" and len(fake.requests) == 2


def test_rate_limit_that_never_clears(fake):
    fake.script = [{"status": 429}, {"status": 429}]
    with pytest.raises(LLMError, match="rate-limiting this key.*switch provider"):
        make(fake).chat(MSG)


def test_unsupported_parameter_surfaces_the_providers_words(fake):
    fake.script = [{"status": 400, "message": "Unsupported value: temperature"}]
    with pytest.raises(LLMError, match="rejected the request: Unsupported value: temperature"):
        make(fake).chat(MSG)


def test_unreachable_endpoint():
    cfg = LLMConfig.from_env(env={"GEMINI_API_KEY": "k", "LLM_BASE_URL": "http://127.0.0.1:9/v1"})
    cfg.max_retries = 0
    with pytest.raises(LLMError, match="could not reach gemini"):
        LLM(cfg).chat(MSG)


def test_reasoning_model_that_runs_out_of_tokens(fake):
    fake.script = [{"content": None, "finish": "length"}]
    with pytest.raises(LLMError, match="ran out of tokens.*LLM_MAX_TOKENS"):
        make(fake, "openai").chat(MSG)


def test_empty_reply(fake):
    fake.script = [{"content": ""}]
    with pytest.raises(LLMError, match="empty reply"):
        make(fake).chat(MSG)


# ---- budget and usage ---------------------------------------------------------------------------------------------

def test_call_budget(fake):
    fake.script = [{"content": "a"}, {"content": "b"}]
    llm = make(fake, LLM_MAX_CALLS="2")
    llm.chat(MSG)
    llm.chat(MSG)
    with pytest.raises(LLMError, match="budget used up"):
        llm.chat(MSG)
    assert len(fake.requests) == 2                                  # the third call was never made


def test_usage_is_counted(fake):
    fake.script = [{"content": "a", "pt": 100, "ct": 20}, {"content": "b", "pt": 50, "ct": 10}]
    llm = make(fake)
    llm.chat(MSG)
    llm.chat(MSG)
    assert (llm.usage.calls, llm.usage.prompt_tokens, llm.usage.completion_tokens) == (2, 150, 30)
    assert "2 call(s)" in llm.usage.summary()


def test_list_models(fake):
    assert make(fake).list_models() == ["alpha", "beta"]


@pytest.mark.parametrize("body,expected", [
    ({"error": {"message": "a"}}, "a"),                      # OpenAI, Groq
    ({"message": "b"}, "b"),
    ([{"error": {"code": 400, "message": "c"}}], "c"),       # Gemini returns a list
    (None, "Error code: 400 - d"),
])
def test_provider_message_shapes(body, expected):
    class E(Exception):
        message = "Error code: 400 - d"
    e = E()
    e.body = body
    assert LLM._provider_message(e) == (expected if body else "d")


# ---- the check command --------------------------------------------------------------------------------------------

def test_cli_llm_check(fake, monkeypatch, capsys):
    monkeypatch.setenv("GROQ_API_KEY", "qk")
    monkeypatch.setenv("LLM_BASE_URL", fake.url)
    fake.script = [{"content": '{"ok": true, "say": "hello"}'}]
    assert cli_main(["llm", "--provider", "groq"]) == 0
    out = capsys.readouterr().out
    assert "provider: groq" in out and "openai/gpt-oss-120b" in out and '"ok":true' in out and "qk" not in out


def test_cli_llm_check_without_a_key(monkeypatch, capsys):
    for k in PROVIDERS.values():
        monkeypatch.delenv(k.key_env, raising=False)
    monkeypatch.chdir("/tmp")                                       # no .env to pick up
    assert cli_main(["llm", "--provider", "openai"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err

# ---- the provider's own JSON check (Groq's json_validate_failed) ---------------------------------------------------

def groq_json_failure(generated):
    return {"status": 400, "error": {"message": "Failed to validate JSON. Please adjust your prompt. See 'failed_generation' "
                                                "for more details.", "type": "invalid_request_error",
                                     "code": "json_validate_failed", "failed_generation": generated}}


def test_a_usable_failed_generation_is_salvaged_without_another_call(fake):
    fake.script = [groq_json_failure('Here is the plan: {"goal": "g", "steps": ["a"]}')]
    assert make(fake, "groq").structured(MSG, Plan) == Plan(goal="g", steps=["a"])
    assert len(fake.requests) == 1


def test_an_unusable_failed_generation_is_retried_without_strict_json_mode(fake):
    fake.script = [groq_json_failure("{broken"), {"content": '{"goal": "g", "steps": ["a"]}'}]
    assert make(fake, "groq").structured(MSG, Plan).steps == ["a"]
    assert "response_format" in fake.requests[0]["body"] and "response_format" not in fake.requests[1]["body"]


def test_a_missing_failed_generation_is_also_retried(fake):
    fake.script = [groq_json_failure(""), {"content": '{"goal": "g", "steps": ["a"]}'}]
    assert make(fake, "groq").structured(MSG, Plan).goal == "g"


def test_a_second_provider_json_failure_is_a_transient_error(fake):
    fake.script = [groq_json_failure("{broken"), groq_json_failure("{broken")]
    with pytest.raises(LLMTransientError, match="Failed to validate JSON"):
        make(fake, "groq").structured(MSG, Plan)
    assert len(fake.requests) == 2                                   # not an endless loop


def test_a_failed_attempt_still_counts_against_the_budget(fake):
    fake.script = [groq_json_failure("{broken"), {"content": '{"goal": "g", "steps": []}'}]
    llm = make(fake, "groq", LLM_MAX_CALLS="5")
    llm.structured(MSG, Plan)
    assert llm.usage.calls == 2


def test_which_errors_are_worth_another_try(fake):
    fake.script = [{"status": 401}]
    with pytest.raises(LLMError) as auth:
        make(fake).chat(MSG)
    assert auth.value.transient is False
    fake.script = [{"status": 500, "message": "server hiccup"}, {"status": 500, "message": "server hiccup"}]
    with pytest.raises(LLMError) as server:
        make(fake).chat(MSG)
    assert server.value.transient is True
    cfg = LLMConfig.from_env(env={"GEMINI_API_KEY": "k", "LLM_BASE_URL": "http://127.0.0.1:9/v1"})
    cfg.max_retries = 0
    with pytest.raises(LLMError) as conn:
        LLM(cfg).chat(MSG)
    assert conn.value.transient is True
    fake.script = [{"status": 429}, {"status": 429}]
    with pytest.raises(LLMError) as rate:
        make(fake).chat(MSG)
    assert rate.value.transient is False                             # already waited; retrying at once would not help