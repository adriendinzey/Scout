"""Per-run token and dollar totals.

Scout publishes cost per query next to quality, so these numbers have to be
right — and have to be loud when they cannot be computed.
"""

from __future__ import annotations

import pytest

from scout.llm.backend import LlmRefusalError, LlmResponse, LlmResponseError, TextBlock
from scout.llm.pricing import RATES, UnknownModelError, Usage, cost_usd
from scout.llm.usage import UsageAccumulator

HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-5"


def response(model: str, usage: Usage) -> LlmResponse:
    return LlmResponse(
        model=model,
        content=(TextBlock("ok"),),
        stop_reason="end_turn",
        usage=usage,
    )


def test_a_fresh_accumulator_has_spent_nothing() -> None:
    accumulator = UsageAccumulator()

    assert accumulator.call_count == 0
    assert accumulator.total_cost_usd == 0.0
    assert accumulator.total_usage == Usage()


def test_recording_a_response_prices_the_model_that_was_billed() -> None:
    accumulator = UsageAccumulator()
    usage = Usage(input_tokens=1_000, output_tokens=200)

    record = accumulator.record("parse", response(HAIKU, usage))

    assert record.label == "parse"
    assert record.model == HAIKU
    assert record.cost_usd == pytest.approx(cost_usd(HAIKU, usage))


def test_tokens_accumulate_across_calls() -> None:
    accumulator = UsageAccumulator()
    accumulator.record("parse", response(HAIKU, Usage(input_tokens=1_000, output_tokens=100)))
    accumulator.record("agent", response(HAIKU, Usage(input_tokens=2_000, output_tokens=300)))

    assert accumulator.call_count == 2
    assert accumulator.total_usage == Usage(input_tokens=3_000, output_tokens=400)


def test_dollars_accumulate_across_calls() -> None:
    accumulator = UsageAccumulator()
    first = Usage(input_tokens=1_000, output_tokens=100)
    second = Usage(input_tokens=4_000, output_tokens=900)
    accumulator.record("agent", response(HAIKU, first))
    accumulator.record("answer", response(SONNET, second))

    assert accumulator.total_cost_usd == pytest.approx(
        cost_usd(HAIKU, first) + cost_usd(SONNET, second)
    )


def test_calls_are_kept_in_order() -> None:
    accumulator = UsageAccumulator()
    for label in ("parse", "agent", "agent", "answer"):
        accumulator.record(label, response(HAIKU, Usage(input_tokens=10, output_tokens=1)))

    assert [call.label for call in accumulator.calls] == ["parse", "agent", "agent", "answer"]


def test_cost_breaks_down_by_node() -> None:
    """The evaluation reports the agent loop's spend apart from Parse's."""
    accumulator = UsageAccumulator()
    parse = Usage(input_tokens=1_000, output_tokens=50)
    agent = Usage(input_tokens=2_000, output_tokens=150)
    accumulator.record("parse", response(HAIKU, parse))
    accumulator.record("agent", response(HAIKU, agent))
    accumulator.record("agent", response(HAIKU, agent))

    by_label = accumulator.cost_by_label()
    assert set(by_label) == {"parse", "agent"}
    assert by_label["agent"] == pytest.approx(2 * cost_usd(HAIKU, agent))
    assert sum(by_label.values()) == pytest.approx(accumulator.total_cost_usd)


def test_tokens_break_down_by_model() -> None:
    accumulator = UsageAccumulator()
    accumulator.record("parse", response(HAIKU, Usage(input_tokens=1_000)))
    accumulator.record("answer", response(SONNET, Usage(input_tokens=5_000, output_tokens=800)))

    by_model = accumulator.usage_by_model()
    assert by_model[HAIKU] == Usage(input_tokens=1_000)
    assert by_model[SONNET] == Usage(input_tokens=5_000, output_tokens=800)


def test_cache_reads_are_recorded_and_priced_below_fresh_input() -> None:
    """Caching the grounding block is the difference between a cheap eval and an
    expensive one, so the saving has to show up in the total."""
    accumulator = UsageAccumulator()
    cached = Usage(input_tokens=50, cache_read_tokens=4_000, output_tokens=100)
    fresh = Usage(input_tokens=4_050, output_tokens=100)
    accumulator.record("parse", response(HAIKU, cached))

    assert accumulator.total_usage.cache_read_tokens == 4_000
    assert accumulator.total_cost_usd < cost_usd(HAIKU, fresh)


def test_cache_writes_are_recorded() -> None:
    accumulator = UsageAccumulator()
    accumulator.record("parse", response(HAIKU, Usage(cache_write_tokens=4_000)))

    assert accumulator.total_usage.cache_write_tokens == 4_000


def test_batch_pricing_halves_the_run() -> None:
    usage = Usage(input_tokens=10_000, output_tokens=1_000)
    live = UsageAccumulator()
    batched = UsageAccumulator(batch=True)
    live.record("agent", response(HAIKU, usage))
    batched.record("agent", response(HAIKU, usage))

    assert batched.total_cost_usd == pytest.approx(live.total_cost_usd / 2)


def test_an_unpriced_model_raises_instead_of_costing_nothing() -> None:
    accumulator = UsageAccumulator()
    assert "claude-not-a-model" not in RATES

    with pytest.raises(UnknownModelError):
        accumulator.record("agent", response("claude-not-a-model", Usage(input_tokens=1)))


def test_a_refused_call_is_still_charged_to_the_run() -> None:
    """Otherwise a run is understated by exactly the calls that went wrong."""
    accumulator = UsageAccumulator()
    usage = Usage(input_tokens=1_500, output_tokens=20)
    error = LlmRefusalError("declined", model=HAIKU, usage=usage)

    record = accumulator.record_failure("agent", error)

    assert record is not None
    assert accumulator.total_usage == usage
    assert accumulator.total_cost_usd == pytest.approx(cost_usd(HAIKU, usage))


def test_a_failure_with_nothing_billed_records_nothing() -> None:
    accumulator = UsageAccumulator()

    assert accumulator.record_failure("parse", LlmResponseError("no response behind it")) is None
    assert accumulator.call_count == 0


def test_usage_can_be_recorded_without_a_response() -> None:
    """For a call priced from a source other than a live response — a batch
    result, or a replayed trace."""
    accumulator = UsageAccumulator()
    accumulator.record_usage("answer", SONNET, Usage(input_tokens=100, output_tokens=20))

    assert accumulator.call_count == 1
    assert accumulator.calls[0].model == SONNET
