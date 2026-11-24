"""Backend `claude_cli`: `claude -p`, for development and CI. Tokens and cost come from its JSON."""

import json
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any

from askmystuff.llm.completion import Completion, LLMError

DEFAULT_MODEL = "claude-sonnet-5-5"
TIMEOUT_S = 600


def _run(args: list[str], *, input: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args, input=input, capture_output=True, text=True, timeout=TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LLMError(f"claude CLI could not run: {error}") from error


def _stream(args: list[str], *, input: str) -> Iterator[str]:
    """Run `claude` and yield its stdout line by line; raise once it ends if it failed."""
    pipe = subprocess.PIPE
    try:
        process = subprocess.Popen(args, stdin=pipe, stdout=pipe, stderr=pipe, text=True)
    except OSError as error:
        raise LLMError(f"claude CLI could not run: {error}") from error
    assert process.stdin and process.stdout and process.stderr
    timer = threading.Timer(TIMEOUT_S, process.kill)  # a killed claude exits with a failure
    timer.start()
    with process:  # closes the pipes and waits
        try:
            try:
                process.stdin.write(input)
                process.stdin.close()
            except BrokenPipeError:
                pass  # claude exited early: its output tells why
            yield from process.stdout
            stderr = process.stderr.read()
            returncode = process.wait()
        finally:
            timer.cancel()
            process.kill()  # no-op once it exited; stops it if the caller stopped reading
    if returncode != 0:
        raise LLMError(f"claude CLI exited with {returncode}: {stderr.strip()}")


def _args(system: str, model: str, output_format: list[str]) -> list[str]:
    # The prompt may be a whole manual: it goes through stdin, not the argument list.
    return [
        "claude",
        "-p",
        "--output-format",
        *output_format,
        "--system-prompt",
        system,
        "--tools",
        "",
        "--no-session-persistence",
        "--model",
        model,
    ]


def _completion(
    reply: dict[str, Any],
    *,
    text: str,
    model: str,
    latency_s: float,
    first_token_s: float | None = None,
) -> Completion:
    """The Completion of the JSON `result` that ends both `json` and `stream-json` outputs."""
    try:
        usage: dict[str, int] = reply["usage"]
        # The whole prompt counts, cached or not.
        input_tokens = (
            usage["input_tokens"]
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        output_tokens = usage["output_tokens"]
        cost_usd = float(reply["total_cost_usd"])
    except (ValueError, KeyError, TypeError) as error:
        raise LLMError(f"claude CLI printed an unexpected JSON: {error!r}") from error
    return Completion(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_s=latency_s,
        model=model,
        first_token_s=first_token_s,
    )


def complete(system: str, prompt: str, model: str | None = None) -> Completion:
    model = model or DEFAULT_MODEL
    args = _args(system, model, ["json"])
    start = time.perf_counter()
    done = _run(args, input=prompt)
    latency_s = time.perf_counter() - start
    if done.returncode != 0:
        raise LLMError(f"claude CLI exited with {done.returncode}: {done.stderr.strip()}")
    try:
        reply: dict[str, Any] = json.loads(done.stdout)
        text = str(reply["result"])
    except (ValueError, KeyError, TypeError) as error:
        raise LLMError(f"claude CLI printed an unexpected JSON: {error!r}") from error
    if reply.get("is_error"):
        raise LLMError(f"claude CLI failed: {text}")
    return _completion(reply, text=text, model=model, latency_s=latency_s)


def _text_delta(event: dict[str, Any]) -> str:
    """The text a `stream_event` adds, "" for every other event."""
    try:
        inner: dict[str, Any] = event["event"]
        delta: dict[str, Any] = inner["delta"]
        if inner["type"] != "content_block_delta" or delta["type"] != "text_delta":
            return ""
        return str(delta["text"])
    except (KeyError, TypeError):
        return ""


def stream(system: str, prompt: str, model: str | None = None) -> Iterator[str | Completion]:
    model = model or DEFAULT_MODEL
    # `stream-json` needs `--verbose` in print mode; partial messages carry the text deltas.
    args = _args(system, model, ["stream-json", "--verbose", "--include-partial-messages"])
    start = time.perf_counter()
    first_token_s: float | None = None
    written = ""
    result: dict[str, Any] | None = None
    for line in _stream(args, input=prompt):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            kind = event["type"]
        except (ValueError, KeyError, TypeError) as error:
            raise LLMError(f"claude CLI printed an unexpected JSON: {error!r}") from error
        if kind == "stream_event":
            delta = _text_delta(event)
            if delta:
                if first_token_s is None:
                    first_token_s = time.perf_counter() - start
                written += delta
                yield delta
        elif kind == "result":
            result = event
    if result is None:
        raise LLMError("claude CLI ended without a result")
    final = str(result.get("result", ""))
    if result.get("is_error"):
        raise LLMError(f"claude CLI failed: {final}")
    # Without partial messages the text comes only with the result: yield what was not streamed.
    if final != written and final.startswith(written):
        if first_token_s is None:
            first_token_s = time.perf_counter() - start
        rest = final[len(written) :]
        written = final
        yield rest
    yield _completion(
        result,
        text=written,
        model=model,
        latency_s=time.perf_counter() - start,
        first_token_s=first_token_s,
    )
