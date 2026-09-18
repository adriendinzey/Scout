"""Cost accounting is arithmetic Scout publishes, so it gets tested."""

from __future__ import annotations

import pytest

from scout.llm.pricing import RATES, UnknownModelError, Usage, cost_usd


def test_cost_of_a_plain_call() -> None:
    # 1M input at $1 + 1M output at $5 on Haiku 4.5.
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", usage) == pytest.approx(6.00)


def test_cache_reads_are_cheaper_than_fresh_input() -> None:
    cached = Usage(cache_read_tokens=1_000_000)
    fresh = Usage(input_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", cached) < cost_usd("claude-haiku-4-5", fresh)


def test_batch_is_half_price() -> None:
    usage = Usage(input_tokens=500_000, output_tokens=100_000)
    full = cost_usd("claude-sonnet-5", usage)
    assert cost_usd("claude-sonnet-5", usage, batch=True) == pytest.approx(full / 2)


def test_unknown_model_raises_rather_than_reporting_zero() -> None:
    with pytest.raises(UnknownModelError, match="no published rate"):
        cost_usd("claude-not-a-model", Usage(input_tokens=10))


def test_usage_adds() -> None:
    total = Usage(input_tokens=1, output_tokens=2) + Usage(input_tokens=3, cache_read_tokens=4)
    assert total == Usage(input_tokens=4, output_tokens=2, cache_read_tokens=4)


def test_every_configured_default_model_has_a_rate() -> None:
    """A default Scout cannot price is a default that breaks the spend guard."""
    from scout.config import Settings

    settings = Settings(_env_file=None)
    for model in (settings.model_parse, settings.model_agent, settings.model_answer):
        assert model in RATES, f"{model} has no rate"
