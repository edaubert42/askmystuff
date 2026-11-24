"""The one interface every model call goes through.

`complete()` answers with the backend chosen by `ASKMYSTUFF_LLM` (`claude_cli`, the default, or
`gemini`). `embed()` always uses Gemini. Never call a provider SDK anywhere else.
"""

import os
from collections.abc import Iterator

from askmystuff.llm import claude_cli, gemini
from askmystuff.llm.completion import Completion, LLMError

__all__ = ["Completion", "LLMError", "complete", "embed", "stream"]

BACKENDS = ("claude_cli", "gemini")


def _backend() -> str:
    backend = os.environ.get("ASKMYSTUFF_LLM") or "claude_cli"
    if backend not in BACKENDS:
        raise LLMError(f"unknown ASKMYSTUFF_LLM {backend!r}: use one of {', '.join(BACKENDS)}")
    return backend


def complete(system: str, prompt: str, model: str | None = None) -> Completion:
    if _backend() == "claude_cli":
        return claude_cli.complete(system, prompt, model)
    return gemini.complete(system, prompt, model)


def stream(system: str, prompt: str, model: str | None = None) -> Iterator[str | Completion]:
    """Yield the answer's text deltas as the model writes them, then the `Completion`.

    The `Completion` is the one `complete()` returns, plus `first_token_s`; the deltas joined
    are exactly its `text`. Errors raise `LLMError` when the stream reaches them.
    """
    if _backend() == "claude_cli":
        return claude_cli.stream(system, prompt, model)
    return gemini.stream(system, prompt, model)


def embed(texts: list[str], task: str) -> list[list[float]]:
    """Embed texts (768 dimensions). `task` is "document" for pages, "query" for questions."""
    return gemini.embed(texts, task)
