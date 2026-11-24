"""llm.stream(): the subprocess and the HTTP call are replaced by recorded streams, nothing else."""

import dataclasses
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import pytest

from askmystuff import llm
from askmystuff.llm import claude_cli, gemini

# --- recorded streams ------------------------------------------------------------------------


def cli_stream(
    deltas: list[str],
    *,
    result: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int = 0,
    cost: float = 0.0123,
    is_error: bool = False,
) -> list[str]:
    """What `claude -p --output-format stream-json --verbose --include-partial-messages` prints."""
    text = "".join(deltas) if result is None else result

    def event(payload: dict[str, Any]) -> dict[str, Any]:
        return {"type": "stream_event", "session_id": "recorded", "event": payload}

    lines: list[dict[str, Any]] = [
        {"type": "system", "subtype": "init", "model": "claude-sonnet-5-5"},
        event({"type": "message_start", "message": {"role": "assistant"}}),
        event({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
    ]
    for delta in deltas:
        payload = {"type": "text_delta", "text": delta}
        lines.append(event({"type": "content_block_delta", "index": 0, "delta": payload}))
    lines += [
        event({"type": "content_block_stop", "index": 0}),
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
        {
            "type": "result",
            "subtype": "error_during_execution" if is_error else "success",
            "is_error": is_error,
            "result": text,
            "total_cost_usd": cost,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": cache_read,
            },
        },
    ]
    return [json.dumps(line) for line in lines]


class FakeClaudeStream:
    """Stands in for the `claude` subprocess: records the call, prints a recorded stream."""

    def __init__(self, lines: list[str], returncode: int = 0) -> None:
        self.lines = lines
        self.returncode = returncode
        self.calls: list[tuple[list[str], str]] = []
        self.printed = 0  # lines the caller has read so far

    def __call__(self, args: list[str], *, input: str) -> Iterator[str]:
        self.calls.append((args, input))
        return self._lines()

    def _lines(self) -> Iterator[str]:
        for line in self.lines:
            self.printed += 1
            yield line
        if self.returncode != 0:
            raise llm.LLMError(f"claude CLI exited with {self.returncode}: boom")


def gemini_chunks(
    deltas: list[str], *, prompt: int = 100, out: int = 20, thoughts: int = 0
) -> list[dict[str, Any]]:
    """What the `streamGenerateContent?alt=sse` endpoint sends: usage on every chunk, growing."""
    chunks: list[dict[str, Any]] = []
    for i, delta in enumerate(deltas):
        usage: dict[str, int] = {"promptTokenCount": prompt}
        if i == len(deltas) - 1:
            usage["candidatesTokenCount"] = out
            if thoughts:
                usage["thoughtsTokenCount"] = thoughts
        chunks.append(
            {"candidates": [{"content": {"parts": [{"text": delta}]}}], "usageMetadata": usage}
        )
    return chunks


class FakeHttpStream:
    """Stands in for the network: plays queued streams (or errors) and records the requests."""

    def __init__(self, *replies: list[dict[str, Any]] | Exception) -> None:
        self.replies: list[Any] = list(replies)
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        self.calls.append((url, headers, body))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return self._chunks(reply)

    @staticmethod
    def _chunks(chunks: list[Any]) -> Iterator[dict[str, Any]]:
        for chunk in chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


@pytest.fixture(autouse=True)
def default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASKMYSTUFF_LLM", raising=False)

    def skip_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr(gemini, "_sleep", skip_sleep)


@pytest.fixture
def claude(monkeypatch: pytest.MonkeyPatch) -> FakeClaudeStream:
    fake = FakeClaudeStream(cli_stream(["Press ", "the power ", "button."]))
    monkeypatch.setattr(claude_cli, "_stream", fake)
    return fake


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> FakeHttpStream:
    fake = FakeHttpStream(gemini_chunks(["Hold the ", "button ", "3 seconds."]))
    monkeypatch.setattr(gemini, "_post_stream", fake)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    return fake


@pytest.fixture
def gemini_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")


def flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def split(items: Iterator[str | llm.Completion]) -> tuple[list[str], llm.Completion]:
    """The deltas, then the one Completion that ends the stream."""
    all_items = list(items)
    *deltas, completion = all_items
    assert all(isinstance(delta, str) for delta in deltas)
    assert isinstance(completion, llm.Completion)
    return [d for d in deltas if isinstance(d, str)], completion


# --- the contract: deltas, then the Completion -----------------------------------------------


def test_stream_yields_the_deltas_then_ends_with_the_completion(claude: FakeClaudeStream):
    deltas, completion = split(llm.stream("Be brief.", "How do I power on?"))
    assert deltas == ["Press ", "the power ", "button."]
    assert completion.text == "Press the power button."
    assert completion.input_tokens == 10
    assert completion.output_tokens == 5
    assert completion.cost_usd == 0.0123
    assert completion.model == "claude-sonnet-5-5"
    assert len(claude.calls) == 1


def test_joining_the_deltas_gives_exactly_the_completion_text_with_claude_cli(
    monkeypatch: pytest.MonkeyPatch,
):
    pieces = ["Étape 1 : ", "appuyez\n", "", " sur « OK » ", "[manuel, p. 12]"]
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(cli_stream(pieces)))
    deltas, completion = split(llm.stream("s", "p"))
    assert "".join(deltas) == completion.text == "".join(pieces)


def test_joining_the_deltas_gives_exactly_the_completion_text_with_gemini(
    http: FakeHttpStream, gemini_backend: None
):
    deltas, completion = split(llm.stream("s", "p"))
    assert len(deltas) == 3
    assert "".join(deltas) == completion.text == "Hold the button 3 seconds."


def test_claude_cli_without_partial_messages_still_joins_to_the_text(
    monkeypatch: pytest.MonkeyPatch,
):
    # No text_delta event at all: the final result is yielded as the one delta.
    monkeypatch.setattr(
        claude_cli, "_stream", FakeClaudeStream(cli_stream([], result="All at once."))
    )
    deltas, completion = split(llm.stream("s", "p"))
    assert deltas == ["All at once."]
    assert completion.text == "All at once."


def test_the_completion_is_the_one_complete_returns_plus_first_token_s(
    monkeypatch: pytest.MonkeyPatch,
):
    pieces = ["Press ", "the power ", "button."]
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(cli_stream(pieces)))
    _, streamed = split(llm.stream("s", "p"))
    final = json.loads(cli_stream(pieces)[-1])  # the same result line, as `--output-format json`

    def run(args: list[str], *, input: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, json.dumps(final), "")

    monkeypatch.setattr(claude_cli, "_run", run)
    completed = llm.complete("s", "p")
    assert completed.first_token_s is None
    assert streamed.first_token_s is not None
    same = dataclasses.replace(streamed, latency_s=completed.latency_s, first_token_s=None)
    assert same == completed


# --- the time to first token -----------------------------------------------------------------


def ticking(monkeypatch: pytest.MonkeyPatch, *ticks: float) -> None:
    clock = iter(ticks)
    last = ticks[-1]
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock, last))


def test_claude_cli_measures_first_token_at_the_first_delta(
    claude: FakeClaudeStream, monkeypatch: pytest.MonkeyPatch
):
    # start, first delta, end: the later deltas do not move first_token_s.
    ticking(monkeypatch, 100.0, 100.8, 104.0)
    _, completion = split(llm.stream("s", "p"))
    assert completion.first_token_s == pytest.approx(0.8)
    assert completion.latency_s == pytest.approx(4.0)


def test_gemini_measures_first_token_at_the_first_delta(
    http: FakeHttpStream, gemini_backend: None, monkeypatch: pytest.MonkeyPatch
):
    ticking(monkeypatch, 10.0, 10.4, 12.0)
    _, completion = split(llm.stream("s", "p"))
    assert completion.first_token_s == pytest.approx(0.4)
    assert completion.latency_s == pytest.approx(2.0)


def test_first_token_s_is_none_when_the_model_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(cli_stream([])))
    deltas, completion = split(llm.stream("s", "p"))
    assert deltas == []
    assert completion.text == ""
    assert completion.first_token_s is None


def test_deltas_arrive_before_the_model_is_done(claude: FakeClaudeStream):
    stream = llm.stream("s", "p")
    assert next(stream) == "Press "
    # The result line, the last one printed, has not been read yet.
    assert claude.printed < len(claude.lines)
    assert list(stream)[-1] != "Press "


# --- backend claude_cli ----------------------------------------------------------------------


def test_claude_cli_runs_stream_json_with_partial_messages(claude: FakeClaudeStream):
    list(llm.stream("Answer from the manual only.", "How do I reset it?"))
    [(args, stdin)] = claude.calls
    assert args[0] == "claude"
    assert "-p" in args
    assert flag_value(args, "--output-format") == "stream-json"
    assert "--verbose" in args  # required by the CLI with stream-json
    assert "--include-partial-messages" in args
    assert flag_value(args, "--system-prompt") == "Answer from the manual only."
    assert flag_value(args, "--tools") == ""
    # The prompt can be a whole manual: it goes through stdin, not the argument list.
    assert stdin == "How do I reset it?"
    assert "How do I reset it?" not in args


def test_claude_cli_stream_default_model_and_chosen_model(claude: FakeClaudeStream):
    list(llm.stream("s", "p"))
    assert flag_value(claude.calls[0][0], "--model") == "claude-sonnet-5-5"
    claude.printed = 0
    _, completion = split(llm.stream("s", "p", model="claude-opus-5-5"))
    assert flag_value(claude.calls[1][0], "--model") == "claude-opus-5-5"
    assert completion.model == "claude-opus-5-5"


def test_claude_cli_stream_input_tokens_count_the_whole_prompt(monkeypatch: pytest.MonkeyPatch):
    lines = cli_stream(["x"], input_tokens=3, cache_read=1200)
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(lines))
    assert split(llm.stream("s", "p"))[1].input_tokens == 1203


def test_claude_cli_stream_exit_code_raises_with_the_reason(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        claude_cli, "_stream", FakeClaudeStream(cli_stream(["Press "])[:4], returncode=1)
    )
    stream = llm.stream("s", "p")
    assert next(stream) == "Press "
    with pytest.raises(llm.LLMError, match="boom"):
        list(stream)


def test_claude_cli_stream_error_result_raises_with_the_reason(monkeypatch: pytest.MonkeyPatch):
    lines = cli_stream([], result="Not logged in", is_error=True)
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(lines))
    with pytest.raises(llm.LLMError, match="Not logged in"):
        list(llm.stream("s", "p"))


def test_claude_cli_stream_without_a_result_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(cli_stream(["a"])[:4]))
    with pytest.raises(llm.LLMError, match="result"):
        list(llm.stream("s", "p"))


def test_claude_cli_stream_garbage_line_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(["not json"]))
    with pytest.raises(llm.LLMError, match="unexpected"):
        list(llm.stream("s", "p"))


CHILD = "import sys; [print(w, flush=True) for w in sys.stdin.read().split()]; {tail}"


def test_the_real_subprocess_reader_yields_lines_as_printed_and_raises_on_failure():
    # Not a model call: a child process stands in for `claude`, to test the reader itself.
    ok = [sys.executable, "-c", CHILD.format(tail="pass")]
    assert list(claude_cli._stream(ok, input="a b c")) == ["a\n", "b\n", "c\n"]  # pyright: ignore[reportPrivateUsage]
    fail = [sys.executable, "-c", CHILD.format(tail="sys.stderr.write('no login'); sys.exit(3)")]
    stream = claude_cli._stream(fail, input="a")  # pyright: ignore[reportPrivateUsage]
    assert next(stream) == "a\n"
    with pytest.raises(llm.LLMError, match="exited with 3: no login"):
        next(stream)
    with pytest.raises(llm.LLMError, match="could not run"):
        list(claude_cli._stream(["/nonexistent/claude"], input=""))  # pyright: ignore[reportPrivateUsage]


# --- backend gemini --------------------------------------------------------------------------


def test_gemini_streams_from_the_sse_endpoint_with_the_key_and_the_same_body(
    http: FakeHttpStream, gemini_backend: None
):
    split(llm.stream("Answer from the manual only.", "How do I pair it?"))
    [(url, headers, body)] = http.calls
    assert url.endswith("/models/gemini-3.8-flash:streamGenerateContent?alt=sse")
    assert headers["x-goog-api-key"] == "test-key-not-real"
    assert body["systemInstruction"] == {"parts": [{"text": "Answer from the manual only."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "How do I pair it?"}]}]


def test_gemini_stream_completion_has_model_tokens_and_cost(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttpStream, gemini_backend: None
):
    http.replies = [gemini_chunks(["a", "b"], prompt=2_000_000, out=400_000, thoughts=600_000)]
    _, completion = split(llm.stream("s", "p"))
    assert completion.model == "gemini-3.8-flash"
    assert completion.input_tokens == 2_000_000
    # Thinking tokens are billed as output, as with complete().
    assert completion.output_tokens == 1_000_000
    assert completion.cost_usd == pytest.approx(2 * 1.50 + 7.50)


def test_gemini_stream_unknown_model_raises_before_any_call(
    http: FakeHttpStream, gemini_backend: None
):
    with pytest.raises(llm.LLMError, match="no price"):
        list(llm.stream("s", "p", model="gemini-unpriced"))
    assert http.calls == []


def test_gemini_stream_without_api_key_raises_before_any_call(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttpStream, gemini_backend: None
):
    monkeypatch.delenv("GOOGLE_API_KEY")
    with pytest.raises(llm.LLMError, match="GOOGLE_API_KEY"):
        list(llm.stream("s", "p"))
    assert http.calls == []


@pytest.mark.parametrize("status", [429, 503])
def test_gemini_stream_retries_once_when_the_call_is_refused(
    http: FakeHttpStream, gemini_backend: None, status: int
):
    http.replies = [gemini.HttpError(status, "busy"), gemini_chunks(["second ", "try"])]
    deltas, completion = split(llm.stream("s", "p"))
    assert completion.text == "second try"
    assert "".join(deltas) == "second try"
    assert len(http.calls) == 2


def test_gemini_stream_gives_up_after_one_retry(http: FakeHttpStream, gemini_backend: None):
    http.replies = [gemini.HttpError(503, "busy")] * 3
    with pytest.raises(llm.LLMError, match="503"):
        list(llm.stream("s", "p"))
    assert len(http.calls) == 2


def test_gemini_stream_does_not_retry_a_client_error(http: FakeHttpStream, gemini_backend: None):
    http.replies = [gemini.HttpError(400, "bad request")]
    with pytest.raises(llm.LLMError, match="400"):
        list(llm.stream("s", "p"))
    assert len(http.calls) == 1


def test_gemini_stream_cut_after_the_first_delta_is_not_replayed(
    http: FakeHttpStream, gemini_backend: None
):
    # Retrying now would repeat words the caller has already received.
    http.replies = [[*gemini_chunks(["Hold "])[:1], gemini.HttpError(503, "cut")]]
    stream = llm.stream("s", "p")
    assert next(stream) == "Hold "
    with pytest.raises(llm.LLMError, match="503"):
        list(stream)
    assert len(http.calls) == 1


def test_gemini_stream_without_usage_raises(http: FakeHttpStream, gemini_backend: None):
    chunk = {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}
    http.replies = [[chunk]]
    with pytest.raises(llm.LLMError, match="usage"):
        list(llm.stream("s", "p"))


# --- backend choice --------------------------------------------------------------------------


def test_stream_defaults_to_claude_cli(claude: FakeClaudeStream, http: FakeHttpStream):
    list(llm.stream("s", "p"))
    assert len(claude.calls) == 1
    assert http.calls == []


def test_stream_asking_for_gemini_uses_the_api_and_not_the_cli(
    claude: FakeClaudeStream, http: FakeHttpStream, gemini_backend: None
):
    list(llm.stream("s", "p"))
    assert len(http.calls) == 1
    assert claude.calls == []


def test_stream_unknown_backend_raises_naming_the_choices(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "openai")
    with pytest.raises(llm.LLMError, match=r"claude_cli.*gemini"):
        list(llm.stream("s", "p"))
