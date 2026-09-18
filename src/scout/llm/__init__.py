"""LLM boundary: the backend protocol, its two implementations, and cost accounting.

Nodes import from here and never from :mod:`anthropic`. :func:`build_backend`
picks the implementation from settings, which default to ``fake`` so an
unconfigured checkout cannot spend money.
"""

from __future__ import annotations

from scout.config import Settings, get_settings
from scout.llm.backend import (
    DEFAULT_MAX_TOKENS,
    FINISHED_STOP_REASONS,
    ContentBlock,
    LlmApiError,
    LlmBackend,
    LlmEmptyResponseError,
    LlmError,
    LlmRefusalError,
    LlmRequest,
    LlmResponse,
    LlmResponseError,
    LlmSchemaError,
    Message,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from scout.llm.fake_backend import (
    FakeBackend,
    FakeTurn,
    error_turn,
    json_turn,
    text_turn,
    tool_use_turn,
)
from scout.llm.pricing import Usage, cost_usd
from scout.llm.usage import CallRecord, UsageAccumulator

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "FINISHED_STOP_REASONS",
    "CallRecord",
    "ContentBlock",
    "FakeBackend",
    "FakeTurn",
    "LlmApiError",
    "LlmBackend",
    "LlmEmptyResponseError",
    "LlmError",
    "LlmRefusalError",
    "LlmRequest",
    "LlmResponse",
    "LlmResponseError",
    "LlmSchemaError",
    "Message",
    "StopReason",
    "TextBlock",
    "ThinkingBlock",
    "ToolDefinition",
    "ToolResultBlock",
    "ToolUseBlock",
    "Usage",
    "UsageAccumulator",
    "build_backend",
    "cost_usd",
    "error_turn",
    "json_turn",
    "text_turn",
    "tool_use_turn",
]


def build_backend(settings: Settings | None = None) -> LlmBackend:
    """The backend this run should use.

    The fake comes back with an empty script, so the first call raises rather
    than inventing an answer. Anything that wants canned responses scripts them;
    anything that wants real ones sets ``SCOUT_LLM_BACKEND=anthropic``.
    """
    settings = settings or get_settings()
    if settings.llm_backend == "fake":
        return FakeBackend()
    # Imported here so that the fake path — the default, and the one CI takes —
    # never pulls the SDK in.
    from scout.llm.anthropic_backend import AnthropicBackend

    return AnthropicBackend(api_key=settings.require_anthropic_key())
