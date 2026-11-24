"""USD prices per 1M tokens, for backends whose API does not report a cost.

Standard prices, from https://ai.google.dev/gemini-api/docs/pricing. Not the introductory
rate ($0.75 / $3.75), which ends on 2026-12-31: benchmark costs must stay valid after that.
"""

from dataclasses import dataclass

from askmystuff.llm.completion import LLMError


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float


PRICES: dict[str, Price] = {
    "gemini-3.8-flash": Price(input_per_million=1.50, output_per_million=7.50),
}


def price_of(model: str) -> Price:
    try:
        return PRICES[model]
    except KeyError:
        raise LLMError(f"no price for model {model!r}: add it to llm/prices.py") from None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = price_of(model)
    return (
        input_tokens * price.input_per_million + output_tokens * price.output_per_million
    ) / 1_000_000
