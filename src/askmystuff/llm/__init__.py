"""The one interface every model call goes through.

`complete()` answers with the backend chosen by `ASKMYSTUFF_LLM` (`claude_cli`, the default, or
`gemini`). `embed()` always uses Gemini. Never call a provider SDK anywhere else.
"""

import os

from askmystuff.llm import claude_cli, gemini
from askmystuff.llm.completion import Completion, LLMError

__all__ = ["Completion", "LLMError", "complete", "embed"]

BACKENDS = ("claude_cli", "gemini")


def complete(system: str, prompt: str, model: str | None = None) -> Completion:
    backend = os.environ.get("ASKMYSTUFF_LLM") or "claude_cli"
    if backend == "claude_cli":
        return claude_cli.complete(system, prompt, model)
    if backend == "gemini":
        return gemini.complete(system, prompt, model)
    raise LLMError(f"unknown ASKMYSTUFF_LLM {backend!r}: use one of {', '.join(BACKENDS)}")


def embed(texts: list[str], task: str) -> list[list[float]]:
    """Embed texts (768 dimensions). `task` is "document" for pages, "query" for questions."""
    return gemini.embed(texts, task)
