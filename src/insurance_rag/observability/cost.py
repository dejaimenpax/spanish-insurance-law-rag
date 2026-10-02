"""Per-request cost from token usage (USD per million tokens)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    input: float
    output: float
    cache_read: float
    cache_write: float


# Anthropic first-party prices; cache writes are 1.25x input (5-minute TTL).
PRICING: dict[str, Pricing] = {
    "claude-opus-5-5": Pricing(4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": Pricing(2.00, 10.00, 0.20, 2.50),
    "claude-haiku-4-5": Pricing(1.00, 5.00, 0.10, 1.25),
    "claude-fable-5-1": Pricing(10.00, 50.00, 0.25, 12.50),
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def cost_usd(model: str, usage: Usage) -> float | None:
    """Cost of one request, or None for a model without a price entry."""
    price = PRICING.get(model)
    if price is None:
        return None
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write
    )
    return round(total / 1_000_000, 6)
