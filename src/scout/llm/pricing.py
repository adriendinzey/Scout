"""What a run costs, in dollars.

Scout reports tokens *and* dollars per query because the evaluation compares
agent variants whose value has to be weighed against what they cost to run: a
relaxation loop that buys two points of precision for triple the spend is a
finding, not an improvement.

Rates are USD per million tokens, from Anthropic's published pricing. They are
data, not a live lookup — verify them when reporting figures, and record the
date alongside any published number.
"""

from __future__ import annotations

from dataclasses import dataclass

RATES_VERIFIED_ON = "2026-09-17"


@dataclass(frozen=True)
class ModelRate:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    # Cache reads bill at roughly a tenth of the input rate; cache writes at
    # about 1.25x. Scout caches the Parse grounding block, which is byte-identical
    # across every query, so this is the difference between a cheap eval and an
    # expensive one.
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25


RATES: dict[str, ModelRate] = {
    "claude-haiku-4-5": ModelRate(input_per_mtok=1.00, output_per_mtok=5.00),
    "claude-sonnet-5": ModelRate(input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-opus-5": ModelRate(input_per_mtok=5.00, output_per_mtok=25.00),
}

# The Batch API runs asynchronously at half price. The eval harness is the
# textbook case for it: a hundred queries with nobody waiting on the answer.
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class Usage:
    """Token counts for a single call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class UnknownModelError(KeyError):
    """Raised for a model with no published rate in this table."""


def cost_usd(model: str, usage: Usage, *, batch: bool = False) -> float:
    """Cost of one call in USD.

    Raises UnknownModelError rather than guessing: a silently-zero cost would
    make the evaluation's spend column a lie.
    """
    try:
        rate = RATES[model]
    except KeyError as exc:
        raise UnknownModelError(
            f"no published rate for model {model!r}; add it to RATES "
            f"(rates last verified {RATES_VERIFIED_ON})"
        ) from exc

    per_token = 1_000_000.0
    total = (
        usage.input_tokens * rate.input_per_mtok
        + usage.output_tokens * rate.output_per_mtok
        + usage.cache_read_tokens * rate.input_per_mtok * rate.cache_read_multiplier
        + usage.cache_write_tokens * rate.input_per_mtok * rate.cache_write_multiplier
    ) / per_token

    return total * BATCH_DISCOUNT if batch else total
