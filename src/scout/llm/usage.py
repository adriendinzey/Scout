"""What one run spent, call by call.

Scout reports tokens *and* dollars per query, because the evaluation compares an
agent that chooses its own tools against a hardcoded path, and a relaxation loop
that buys two points of precision for triple the spend is a finding rather than
an improvement. That comparison needs the figures broken down by node — Parse is
one cheap cached call, the agent is a loop, Answer is one expensive call — not a
single total.

The cost ceiling that stops a runaway loop reads :attr:`total_cost_usd` as the
loop runs, so this is priced from real token counts rather than estimated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scout.llm.backend import LlmResponse, LlmResponseError
from scout.llm.pricing import Usage, cost_usd


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One priced call.

    ``label`` names the node that made it — ``parse``, ``agent``, ``answer`` —
    which is the axis the evaluation reports on.
    """

    label: str
    model: str
    usage: Usage
    cost_usd: float


class UsageAccumulator:
    """Running totals for one query.

    ``batch`` prices the whole run at the Batch API's half rate; the eval harness
    sets it, and a live query never does.
    """

    def __init__(self, *, batch: bool = False) -> None:
        self._batch = batch
        self._calls: list[CallRecord] = []

    def record(self, label: str, response: LlmResponse) -> CallRecord:
        """Price and record a completed call.

        Takes the response rather than loose numbers so the model that was
        actually billed is the model that gets priced — a caller cannot pass the
        model it *meant* to use.
        """
        return self.record_usage(label, response.model, response.usage)

    def record_usage(self, label: str, model: str, usage: Usage) -> CallRecord:
        """Record usage that did not come from an :class:`LlmResponse`.

        Propagates :class:`~scout.llm.pricing.UnknownModelError` for a model with
        no published rate: a silently-zero cost would make the evaluation's spend
        column a lie.
        """
        record = CallRecord(
            label=label,
            model=model,
            usage=usage,
            cost_usd=cost_usd(model, usage, batch=self._batch),
        )
        self._calls.append(record)
        return record

    def record_failure(self, label: str, error: LlmResponseError) -> CallRecord | None:
        """Record what a refused or unusable response still cost.

        A refusal, a truncated body, and output that does not match the schema
        are all billed. Leaving them out would understate a run by exactly the
        calls that went wrong, and let the per-query ceiling be overrun by them.

        Returns ``None`` when the failure happened before anything was billed —
        an :class:`~scout.llm.backend.LlmApiError`, or an error raised without a
        response behind it.
        """
        if error.model is None or error.usage is None:
            return None
        return self.record_usage(label, error.model, error.usage)

    @property
    def calls(self) -> Sequence[CallRecord]:
        """Every call, in the order it was made."""
        return tuple(self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for record in self._calls:
            total = total + record.usage
        return total

    @property
    def total_cost_usd(self) -> float:
        return sum(record.cost_usd for record in self._calls)

    def cost_by_label(self) -> dict[str, float]:
        """Dollars per node, for the evaluation's per-stage breakdown."""
        totals: dict[str, float] = {}
        for record in self._calls:
            totals[record.label] = totals.get(record.label, 0.0) + record.cost_usd
        return totals

    def usage_by_model(self) -> dict[str, Usage]:
        """Tokens per model, so a cheap Parse is not averaged into a costly Answer."""
        totals: dict[str, Usage] = {}
        for record in self._calls:
            totals[record.model] = totals.get(record.model, Usage()) + record.usage
        return totals
