"""Backend `claude_cli`: `claude -p`, for development and CI. Tokens and cost come from its JSON."""

import json
import subprocess
import time
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


def complete(system: str, prompt: str, model: str | None = None) -> Completion:
    model = model or DEFAULT_MODEL
    # The prompt may be a whole manual: it goes through stdin, not the argument list.
    args = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--system-prompt",
        system,
        "--tools",
        "",
        "--no-session-persistence",
        "--model",
        model,
    ]
    start = time.perf_counter()
    done = _run(args, input=prompt)
    latency_s = time.perf_counter() - start
    if done.returncode != 0:
        raise LLMError(f"claude CLI exited with {done.returncode}: {done.stderr.strip()}")
    try:
        reply: dict[str, Any] = json.loads(done.stdout)
        text = str(reply["result"])
        if reply.get("is_error"):
            raise LLMError(f"claude CLI failed: {text}")
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
    )
