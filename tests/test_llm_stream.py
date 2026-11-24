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


def cli_stream(deltas: list[str], *, result: str | None = None, error: bool = False) -> list[str]:
    """What `claude -p --output-format stream-json --verbose --include-partial-messages` prints."""
    text = "".join(deltas) if result is None else result
    usage = {"input_tokens": 3, "output_tokens": 5, "cache_read_input_tokens": 7}
    events: list[dict[str, Any]] = [{"type": "system", "subtype": "init"}]
    for delta in deltas:
        inner = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": delta}}
        events.append({"type": "stream_event", "event": inner})
    events.append({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})
    result_event = {"type": "result", "is_error": error, "result": text, "usage": usage}
    events.append({**result_event, "total_cost_usd": 0.0123})
    return [json.dumps(event) for event in events]


class FakeClaudeStream:
    """Stands in for the `claude` subprocess: records the call, prints a recorded stream."""

    def __init__(self, lines: list[str], failure: str = "") -> None:
        self.lines, self.failure = lines, failure
        self.calls: list[tuple[list[str], str]] = []
        self.printed = 0  # lines the caller has read so far

    def __call__(self, args: list[str], *, input: str) -> Iterator[str]:
        self.calls.append((args, input))
        for line in self.lines:
            self.printed += 1
            yield line
        if self.failure:
            raise llm.LLMError(self.failure)


def gemini_chunks(deltas: list[str], *, prompt: int = 100, out: int = 20, thoughts: int = 0):
    """What `streamGenerateContent?alt=sse` sends; the last event has the totals."""
    usage = {
        "promptTokenCount": prompt,
        "candidatesTokenCount": out,
        "thoughtsTokenCount": thoughts,
    }
    return [
        {
            "candidates": [{"content": {"parts": [{"text": delta}]}}],
            "usageMetadata": usage if i == len(deltas) - 1 else {"promptTokenCount": prompt},
        }
        for i, delta in enumerate(deltas)
    ]


class FakeHttpStream:
    """Stands in for the network: plays queued streams (or errors) and records the requests."""

    def __init__(self, *replies: list[Any] | Exception) -> None:
        self.replies: list[Any] = list(replies)
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(self, url: str, headers: dict[str, str], body: dict[str, Any]):
        self.calls.append((url, headers, body))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        for chunk in reply:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


@pytest.fixture(autouse=True)
def default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASKMYSTUFF_LLM", raising=False)
    monkeypatch.setattr(gemini, "_sleep", float)  # no waiting: float(seconds) returns at once


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
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    return fake


def split(items: Iterator[str | llm.Completion]) -> tuple[list[str], llm.Completion]:
    """The deltas, then the one Completion that ends the stream."""
    *deltas, completion = items
    assert all(isinstance(delta, str) for delta in deltas)
    assert isinstance(completion, llm.Completion)
    return [d for d in deltas if isinstance(d, str)], completion


def ticking(monkeypatch: pytest.MonkeyPatch, *ticks: float) -> None:
    clock = iter(ticks)
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock, ticks[-1]))


# --- the contract: deltas, then the Completion -----------------------------------------------


def test_claude_cli_yields_the_deltas_then_the_completion_with_first_token_s(
    claude: FakeClaudeStream, monkeypatch: pytest.MonkeyPatch
):
    ticking(monkeypatch, 100.0, 100.8, 104.0)  # start, first delta, end
    deltas, completion = split(llm.stream("s", "p"))
    assert deltas == ["Press ", "the power ", "button."]
    assert completion == llm.Completion(
        text="Press the power button.",
        input_tokens=3 + 7,  # the whole prompt counts, cached or not
        output_tokens=5,
        cost_usd=0.0123,
        latency_s=pytest.approx(4.0),  # pyright: ignore[reportArgumentType]
        model="claude-sonnet-5-5",
        first_token_s=pytest.approx(0.8),  # pyright: ignore[reportArgumentType]
    )


def test_gemini_yields_the_deltas_then_the_completion_with_first_token_s(
    http: FakeHttpStream, monkeypatch: pytest.MonkeyPatch
):
    http.replies = [
        gemini_chunks(["Hold ", "it."], prompt=2_000_000, out=400_000, thoughts=600_000)
    ]
    ticking(monkeypatch, 10.0, 10.4, 12.0)
    deltas, completion = split(llm.stream("s", "p"))
    assert deltas == ["Hold ", "it."]
    assert completion.text == "Hold it."
    assert completion.model == "gemini-3.8-flash"
    assert completion.input_tokens == 2_000_000
    assert completion.output_tokens == 1_000_000  # thinking tokens are billed as output
    assert completion.cost_usd == pytest.approx(2 * 1.50 + 7.50)
    assert completion.first_token_s == pytest.approx(0.4)
    assert completion.latency_s == pytest.approx(2.0)


def test_joining_the_deltas_gives_exactly_the_text_with_both_backends(
    http: FakeHttpStream, monkeypatch: pytest.MonkeyPatch
):
    pieces = ["Étape 1 : ", "appuyez\n", "", " sur « OK » ", "[manuel, p. 12]"]
    http.replies = [gemini_chunks(pieces)]
    deltas, completion = split(llm.stream("s", "p"))
    assert "".join(deltas) == completion.text == "".join(pieces)
    monkeypatch.setenv("ASKMYSTUFF_LLM", "claude_cli")
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(cli_stream(pieces)))
    deltas, completion = split(llm.stream("s", "p"))
    assert "".join(deltas) == completion.text == "".join(pieces)


def test_claude_cli_without_partial_messages_yields_the_result_as_one_delta(
    monkeypatch: pytest.MonkeyPatch,
):
    lines = cli_stream([], result="All at once.")
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(lines))
    deltas, completion = split(llm.stream("s", "p"))
    assert deltas == ["All at once."] == [completion.text]


def test_the_completion_is_the_one_complete_returns_plus_first_token_s(
    claude: FakeClaudeStream, monkeypatch: pytest.MonkeyPatch
):
    _, streamed = split(llm.stream("s", "p"))
    final = cli_stream(["Press ", "the power ", "button."])[-1]

    def run(args: list[str], *, input: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, final, "")

    monkeypatch.setattr(claude_cli, "_run", run)
    completed = llm.complete("s", "p")
    assert completed.first_token_s is None
    assert streamed.first_token_s is not None
    assert dataclasses.replace(streamed, latency_s=0, first_token_s=None) == dataclasses.replace(
        completed, latency_s=0
    )


def test_deltas_arrive_before_the_model_is_done(claude: FakeClaudeStream):
    stream = llm.stream("s", "p")
    assert next(stream) == "Press "
    assert claude.printed < len(claude.lines)  # the result line has not been read yet


# --- backend claude_cli ----------------------------------------------------------------------


def test_claude_cli_runs_stream_json_with_partial_messages(claude: FakeClaudeStream):
    list(llm.stream("Answer from the manual only.", "How do I reset it?", model="claude-opus-5-5"))
    [(args, stdin)] = claude.calls

    def flag(name: str) -> str:
        return args[args.index(name) + 1]

    assert (args[0], "-p" in args) == ("claude", True)
    assert flag("--output-format") == "stream-json"
    assert "--verbose" in args  # required by the CLI with stream-json
    assert "--include-partial-messages" in args
    assert flag("--system-prompt") == "Answer from the manual only."
    assert (flag("--tools"), flag("--model")) == ("", "claude-opus-5-5")
    assert stdin == "How do I reset it?"  # a whole manual goes through stdin, not the arguments
    assert "How do I reset it?" not in args


@pytest.mark.parametrize(
    ("lines", "failure", "message"),
    [
        (cli_stream(["Press "])[:2], "claude CLI exited with 1: boom", "boom"),
        (cli_stream([], result="Not logged in", error=True), "", "Not logged in"),
        (cli_stream(["a"])[:2], "", "result"),
        (["not json"], "", "unexpected"),
    ],
)
def test_claude_cli_stream_failures_raise_with_the_reason(
    monkeypatch: pytest.MonkeyPatch, lines: list[str], failure: str, message: str
):
    monkeypatch.setattr(claude_cli, "_stream", FakeClaudeStream(lines, failure))
    with pytest.raises(llm.LLMError, match=message):
        list(llm.stream("s", "p"))


def test_the_real_subprocess_reader_yields_lines_as_printed_and_raises_on_failure():
    # Not a model call: a child process stands in for `claude`, to test the reader itself.
    stream = claude_cli._stream  # pyright: ignore[reportPrivateUsage]
    code = "import sys; [print(w, flush=True) for w in sys.stdin.read().split()]; "
    assert list(stream([sys.executable, "-c", code], input="a b c")) == ["a\n", "b\n", "c\n"]
    fail = code + "sys.stderr.write('no login'); sys.exit(3)"
    lines = stream([sys.executable, "-c", fail], input="a")
    assert next(lines) == "a\n"
    with pytest.raises(llm.LLMError, match="exited with 3: no login"):
        next(lines)
    with pytest.raises(llm.LLMError, match="could not run"):
        list(stream(["/nonexistent/claude"], input=""))


# --- backend gemini --------------------------------------------------------------------------


def test_gemini_streams_from_the_sse_endpoint_with_the_key_and_the_same_body(http: FakeHttpStream):
    split(llm.stream("Answer from the manual only.", "How do I pair it?"))
    [(url, headers, body)] = http.calls
    assert url.endswith("/models/gemini-3.8-flash:streamGenerateContent?alt=sse")
    assert headers["x-goog-api-key"] == "test-key-not-real"
    assert body["systemInstruction"] == {"parts": [{"text": "Answer from the manual only."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "How do I pair it?"}]}]


def test_gemini_stream_fails_before_any_call_without_a_price_or_a_key(
    monkeypatch: pytest.MonkeyPatch, http: FakeHttpStream
):
    with pytest.raises(llm.LLMError, match="no price"):
        list(llm.stream("s", "p", model="gemini-unpriced"))
    monkeypatch.delenv("GOOGLE_API_KEY")
    with pytest.raises(llm.LLMError, match="GOOGLE_API_KEY"):
        list(llm.stream("s", "p"))
    assert http.calls == []


def test_gemini_stream_retries_once_a_refused_call_but_never_replays_a_cut_stream(
    http: FakeHttpStream,
):
    http.replies = [gemini.HttpError(503, "busy"), gemini_chunks(["second ", "try"])]
    deltas, completion = split(llm.stream("s", "p"))
    assert "".join(deltas) == completion.text == "second try"
    assert len(http.calls) == 2
    # Cut after a delta: retrying would repeat words the caller has already received.
    http.replies = [[gemini_chunks(["Hold "])[0], gemini.HttpError(503, "cut")]]
    stream = llm.stream("s", "p")
    assert next(stream) == "Hold "
    with pytest.raises(llm.LLMError, match="503"):
        list(stream)
    assert len(http.calls) == 3


# --- backend choice --------------------------------------------------------------------------


def test_stream_defaults_to_claude_cli_and_gemini_is_asked_for(
    claude: FakeClaudeStream, http: FakeHttpStream, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("ASKMYSTUFF_LLM")
    list(llm.stream("s", "p"))
    assert (len(claude.calls), len(http.calls)) == (1, 0)
    monkeypatch.setenv("ASKMYSTUFF_LLM", "gemini")
    list(llm.stream("s", "p"))
    assert (len(claude.calls), len(http.calls)) == (1, 1)


def test_stream_unknown_backend_raises_naming_the_choices(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASKMYSTUFF_LLM", "openai")
    with pytest.raises(llm.LLMError, match=r"claude_cli.*gemini"):
        list(llm.stream("s", "p"))
