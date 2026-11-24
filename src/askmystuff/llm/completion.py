"""What every backend returns, and the one error they raise."""

from dataclasses import dataclass


class LLMError(Exception):
    """A model call failed: bad configuration, a failed subprocess or an HTTP error."""


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    model: str
    # Seconds until the first text arrived: measured by `stream()` only, None for `complete()`
    # and when the model wrote nothing.
    first_token_s: float | None = None
