"""The model interface: unit tests mock the subprocess and the HTTP calls, nothing else."""

import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import pytest

from askmystuff import llm
from askmystuff.llm import claude_cli, gemini

# --- fakes: the subprocess and the network, the only things replaced -------------------------


def cli_json(
    result: str = "Press the power button.",
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_creation: int = 0,
    cache_read: int = 0,
    cost: float = 0.0123,
    is_error: bool = False,
) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "error_during_execution" if is_error else "success",
            "is_error": is_error,
            "result": result,
            "total_cost_usd": cost,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            },
        }
    )


class FakeClaude:
    """Stands in for the `claude` subprocess: records the call, prints a canned JSON."""

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, args: list[str], *, input: str) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, input))
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, "boom")


def gemini_reply(
    text: str = "Hold the button 3 seconds.", prompt: int = 100, out: int = 20, thoughts: int = 0
) -> dict[str, Any]:
    usage: dict[str, int] = {"promptTokenCount": prompt, "candidatesTokenCount": out}
    if thoughts:
        usage["thoughtsTokenCount"] = thoughts
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": usage,
    }


class FakeHttp:
    """Stands in for the network: plays queued replies (or errors) and records the requests."""

    def __init__(self, *replies: dict[str, Any] | Exception) -> None:
        self.replies: list[Any] = list(replies)
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((url, headers, body))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    def skip_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr(gemini, "_sleep", skip_sleep)


@pytest.fixture
def claude(monkeypatch: pytest.MonkeyPatch) -> FakeClaude:
    fake = FakeClaude(cli_json())
    monkeypatch.setattr(claude_cli, "_run", fake)
    return fake


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> FakeHttp:
    fake = FakeHttp(gemini_reply())
    monkeypatch.setattr(gemini, "_post", fake)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    return fake


@pytest.fixture(autouse=True)
def default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASKMYSTUFF_LLM", raising=False)


def flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


# --- complete(): the Completion --------------------------------------------------------------


def test_complete_returns_text_tokens_cost_latency_and_model(
    claude: FakeClaude, monkeypatch: pytest.MonkeyPatch
):
    ticks: Iterator[float] = iter([100.0, 102.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks, 102.5))
    completion = llm.complete("Be brief.", "How do I power on?")
    assert completion == llm.Completion(
        text="Press the power button.",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0123,
        latency_s=2.5,
        model="claude-sonnet-5-5",
    )
    assert len(claude.calls) == 1


# --- backend claude_cli ----------------------------------------------------------------------


def test_claude_cli_runs_print_mode_json_with_system_prompt_and_no_tools(claude: FakeClaude):
    llm.complete("Answer from the manual only.", "How do I reset it?")
    [(args, stdin)] = claude.calls
    assert args[0] == "claude"
    assert "-p" in args
    assert flag_value(args, "--output-format") == "json"
    assert flag_value(args, "--system-prompt") == "Answer from the manual only."
    assert flag_value(args, "--tools") == ""
    # The prompt can be a whole manual: it goes through stdin, not the argument list.
    assert stdin == "How do I reset it?"
    assert "How do I reset it?" not in args


def test_claude_cli_default_model_is_sonnet_and_model_can_be_chosen(claude: FakeClaude):
    llm.complete("s", "p")
    assert flag_value(claude.calls[0][0], "--model") == "claude-sonnet-5-5"
    completion = llm.complete("s", "p", model="claude-opus-5-5")
    assert flag_value(claude.calls[1][0], "--model") == "claude-opus-5-5"
    assert completion.model == "claude-opus-5-5"


def test_claude_cli_input_tokens_count_the_whole_prompt_including_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = FakeClaude(cli_json(input_tokens=3, cache_creation=1000, cache_read=200))
    monkeypatch.setattr(claude_cli, "_run", fake)
    assert llm.complete("s", "p").input_tokens == 1203


def test_claude_cli_failure_raises_with_the_reason(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(claude_cli, "_run", FakeClaude("", returncode=1))
    with pytest.raises(llm.LLMError, match="boom"):
        llm.complete("s", "p")
    monkeypatch.setattr(claude_cli, "_run", FakeClaude(cli_json("Not logged in", is_error=True)))
    with pytest.raises(llm.LLMError, match="Not logged in"):
        llm.complete("s", "p")


# --- backend gemini --------------------------------------------------------------------------


def test_gemini_posts_system_and_prompt_with_the_key_and_default_model(
    http: FakeHttp, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    completion = llm.complete("Answer from the manual only.", "How do I pair it?")
    [(url, headers, body)] = http.calls
    assert url.endswith("/models/gemini-3.8-flash:generateContent")
    assert headers["x-goog-api-key"] == "test-key-not-real"
    assert body["systemInstruction"] == {"parts": [{"text": "Answer from the manual only."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "How do I pair it?"}]}]
    assert completion.text == "Hold the button 3 seconds."
    assert completion.model == "gemini-3.8-flash"
    assert completion.input_tokens == 100
    assert completion.output_tokens == 20
    assert completion.latency_s >= 0


def test_gemini_cost_uses_the_standard_prices_per_million_tokens(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttp
):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    http.replies = [gemini_reply(prompt=2_000_000, out=1_000_000)]
    # $1.50 per 1M input + $7.50 per 1M output: not the introductory rate, which ends 2026-12-31.
    assert llm.complete("s", "p").cost_usd == pytest.approx(2 * 1.50 + 7.50)


def test_gemini_thinking_tokens_are_billed_as_output(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttp
):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    http.replies = [gemini_reply(prompt=0, out=400_000, thoughts=600_000)]
    completion = llm.complete("s", "p")
    assert completion.output_tokens == 1_000_000
    assert completion.cost_usd == pytest.approx(7.50)


def test_gemini_unknown_model_has_no_price_and_raises(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttp
):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    with pytest.raises(llm.LLMError, match="no price"):
        llm.complete("s", "p", model="gemini-unpriced")
    assert http.calls == []


def test_gemini_without_api_key_raises_before_any_call(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttp
):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY")
    with pytest.raises(llm.LLMError, match="GOOGLE_API_KEY"):
        llm.complete("s", "p")
    assert http.calls == []


@pytest.mark.parametrize("status", [429, 500, 503])
def test_gemini_retries_once_on_429_and_5xx(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttp, status: int
):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    http.replies = [gemini.HttpError(status, "busy"), gemini_reply(text="second try")]
    assert llm.complete("s", "p").text == "second try"
    assert len(http.calls) == 2


def test_gemini_gives_up_after_one_retry(monkeypatch: pytest.MonkeyPatch, http: FakeHttp):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    http.replies = [gemini.HttpError(503, "busy")] * 3
    with pytest.raises(llm.LLMError, match="503"):
        llm.complete("s", "p")
    assert len(http.calls) == 2


def test_gemini_does_not_retry_a_client_error(monkeypatch: pytest.MonkeyPatch, http: FakeHttp):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    http.replies = [gemini.HttpError(400, "bad request")]
    with pytest.raises(llm.LLMError, match="400"):
        llm.complete("s", "p")
    assert len(http.calls) == 1


# --- backend choice --------------------------------------------------------------------------


def test_backend_defaults_to_claude_cli(claude: FakeClaude, http: FakeHttp):
    llm.complete("s", "p")
    assert len(claude.calls) == 1
    assert http.calls == []


def test_asking_for_gemini_uses_the_api_and_not_the_cli(
    claude: FakeClaude, http: FakeHttp, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    llm.complete("s", "p")
    assert len(http.calls) == 1
    assert claude.calls == []


def test_asking_for_claude_cli_explicitly_uses_the_cli(
    claude: FakeClaude, http: FakeHttp, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "claude_cli")
    llm.complete("s", "p")
    assert len(claude.calls) == 1


def test_unknown_backend_raises_naming_the_choices(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "openai")
    with pytest.raises(llm.LLMError, match=r"claude_cli.*gemini"):
        llm.complete("s", "p")


# --- embed() ---------------------------------------------------------------------------------


def embed_reply(n: int, start: int = 0) -> dict[str, Any]:
    return {"embeddings": [{"values": [float(start + i), 0.5]} for i in range(n)]}


def test_embed_uses_gemini_embedding_2_with_768_dimensions(http: FakeHttp):
    http.replies = [embed_reply(2)]
    vectors = llm.embed(["reset the printer", "pair the speaker"], "document")
    [(url, headers, body)] = http.calls
    assert url.endswith("/models/gemini-embedding-2:batchEmbedContents")
    assert headers["x-goog-api-key"] == "test-key-not-real"
    assert [r["content"] for r in body["requests"]] == [
        {"parts": [{"text": "reset the printer"}]},
        {"parts": [{"text": "pair the speaker"}]},
    ]
    assert {r["outputDimensionality"] for r in body["requests"]} == {768}
    assert {r["model"] for r in body["requests"]} == {"models/gemini-embedding-2"}
    assert vectors == [[0.0, 0.5], [1.0, 0.5]]


@pytest.mark.parametrize(
    ("task", "task_type"), [("document", "RETRIEVAL_DOCUMENT"), ("query", "RETRIEVAL_QUERY")]
)
def test_embed_maps_the_task_to_the_gemini_task_type(http: FakeHttp, task: str, task_type: str):
    http.replies = [embed_reply(1)]
    llm.embed(["x"], task)
    assert {r["taskType"] for r in http.calls[0][2]["requests"]} == {task_type}


def test_embed_unknown_task_raises_before_any_call(http: FakeHttp):
    with pytest.raises(llm.LLMError, match="task"):
        llm.embed(["x"], "classification")
    assert http.calls == []


def test_embed_batches_and_keeps_the_order(http: FakeHttp):
    texts = [f"page {i}" for i in range(250)]
    http.replies = [embed_reply(100, 0), embed_reply(100, 100), embed_reply(50, 200)]
    vectors = llm.embed(texts, "document")
    sizes = [len(call[2]["requests"]) for call in http.calls]
    assert sizes == [100, 100, 50]
    assert [v[0] for v in vectors] == [float(i) for i in range(250)]


def test_embed_nothing_calls_nothing(http: FakeHttp):
    assert llm.embed([], "query") == []
    assert http.calls == []


def test_embed_retries_once_on_a_server_error(http: FakeHttp):
    http.replies = [gemini.HttpError(500, "oops"), embed_reply(1)]
    assert len(llm.embed(["x"], "query")) == 1
    assert len(http.calls) == 2


def test_embed_checks_the_number_of_vectors_returned(http: FakeHttp):
    http.replies = [embed_reply(1)]
    with pytest.raises(llm.LLMError, match="2 texts"):
        llm.embed(["a", "b"], "query")


# --- the real services: skipped when not available -------------------------------------------


def claude_logged_in() -> bool:
    if shutil.which("claude") is None:
        return False
    try:
        status = subprocess.run(["claude", "auth", "status"], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return status.returncode == 0


@pytest.mark.skipif(not claude_logged_in(), reason="claude is not logged in")
def test_real_claude_cli_answers_with_usage_and_cost():
    completion = llm.complete("Reply with the single word: pong.", "ping")
    assert "pong" in completion.text.lower()
    assert completion.input_tokens > 0
    assert completion.output_tokens > 0
    assert completion.cost_usd > 0
    assert completion.model == "claude-sonnet-5-5"


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="no GOOGLE_API_KEY")
def test_real_gemini_answers_and_embeds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    completion = llm.complete("Reply with the single word: pong.", "ping")
    assert "pong" in completion.text.lower()
    assert completion.cost_usd > 0
    vectors = llm.embed(["bonjour", "au revoir"], "document")
    assert len(vectors) == 2
    assert {len(v) for v in vectors} == {768}
