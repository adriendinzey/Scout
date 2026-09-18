"""The real Claude, behind the same interface as the fake.

This is the only module in Scout that imports ``anthropic``. Everything above it
speaks :mod:`scout.llm.backend`'s own block types, which is what lets the graph
run unchanged against the fake.

Two things it is careful about:

**The cacheable prefix.** The API renders a request as ``tools`` → ``system`` →
``messages`` and the cache is a prefix match, so one ``cache_control`` breakpoint
on the last system block covers the tool definitions and the grounding block
together, while the user's query — which changes every time — sits after it and
is never part of the cached prefix. Any byte that moves in that prefix
invalidates the cache silently; the only symptom is a bigger bill, which is why
:class:`~scout.llm.backend.LlmRequest` is a comparable value and the translation
below is a pure function of it.

**What a failure looks like.** Every SDK exception becomes an
:class:`~scout.llm.backend.LlmApiError` that says whether retrying could help. A
refusal, an empty turn, and a body that is not the requested JSON each raise
their own error. Nothing here returns an empty result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, get_args

import anthropic
from anthropic import transform_schema
from anthropic.types import ContentBlock as AnthropicContentBlock
from anthropic.types import (
    ContentBlockParam,
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ToolParam,
    ToolUnionParam,
)
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage
from pydantic import SecretStr

from scout.llm.backend import (
    ContentBlock,
    LlmApiError,
    LlmEmptyResponseError,
    LlmRefusalError,
    LlmRequest,
    LlmResponse,
    LlmResponseError,
    Message,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolUseBlock,
    structured_output,
)
from scout.llm.pricing import Usage

_STOP_REASONS: frozenset[str] = frozenset(get_args(StopReason))

# The SDK defaults to a 600-second read timeout and two retries — half an hour
# against a per-query budget of sixty seconds. A wall clock checked between turns
# of the agent loop cannot interrupt a blocked socket, so the ceiling has to sit
# on the client itself. The node that owns the budget should pass its own figures
# derived from `query_timeout_s`; these are only a safe default.
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 1

# 408 and 409 are transient by definition; 429 and 5xx are the rate limit and the
# server's own trouble. Everything else is a request that will fail the same way
# however many times it is sent.
_RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 409, 429})


class AnthropicBackend:
    """Calls the Anthropic Messages API.

    ``client`` is injectable so the translation and the error mapping can be
    tested against a stub. No test constructs a real client, and none may.
    """

    def __init__(
        self,
        api_key: SecretStr,
        *,
        client: anthropic.Anthropic | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._client = client or anthropic.Anthropic(
            api_key=api_key.get_secret_value(),
            timeout=timeout_s,
            max_retries=max_retries,
        )

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Run one turn against the API."""
        system = to_system_blocks(request.system, cache=request.cache_prefix)
        tools = to_tool_params(request.tools)
        output_config = to_output_config(request.response_schema)

        # TODO: the agent loop may want at most one tool call per turn
        # (tool_choice with disable_parallel_tool_use) so the tool-call budget
        # counts turns and calls alike. That is the loop's policy, not the
        # boundary's.
        try:
            message = self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                system=system or anthropic.omit,
                messages=to_message_params(request.messages),
                tools=cast(list[ToolUnionParam], tools) or anthropic.omit,
                output_config=output_config if output_config is not None else anthropic.omit,
            )
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code in _RETRYABLE_STATUSES or exc.status_code >= 500
            raise LlmApiError(
                f"Anthropic API returned {exc.status_code}: {exc.message}",
                retryable=retryable,
                status_code=exc.status_code,
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise LlmApiError(f"Anthropic API call timed out: {exc}", retryable=True) from exc
        except anthropic.APIConnectionError as exc:
            raise LlmApiError(f"could not reach the Anthropic API: {exc}", retryable=True) from exc
        except anthropic.AnthropicError as exc:
            # Whatever the clauses above did not name: credential and federation
            # failures, a response the SDK could not validate, and the SDK's own
            # retryable errors — which land here only once its retries are spent.
            # None of them is worth Scout trying the identical call again.
            raise LlmApiError(f"Anthropic client failure: {exc}", retryable=False) from exc

        return to_response(message, response_schema=request.response_schema)


# --- Translation, out ------------------------------------------------------


def to_system_blocks(system: str, *, cache: bool) -> list[TextBlockParam]:
    """The system prompt, with the cache breakpoint on its last block.

    Tools render before ``system``, so this one breakpoint caches the tool
    definitions too. A request with tools but no system prompt therefore caches
    nothing — which is fine, because every node Scout has sends one.
    """
    if not system:
        return []
    block: TextBlockParam = {"type": "text", "text": system}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def to_tool_params(tools: Sequence[ToolDefinition]) -> list[ToolParam]:
    """Tool definitions in wire shape, in the order given.

    The order is the caller's and is never sorted here: tools are part of the
    cached prefix, so a set rendered in a different order each run would quietly
    cost money.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": cast(Any, dict(tool.input_schema)),
        }
        for tool in tools
    ]


def to_output_config(response_schema: Mapping[str, Any] | None) -> OutputConfigParam | None:
    """The structured-output request, or ``None`` when none was asked for.

    The schema compiler takes a subset of JSON Schema: no numeric or length
    constraints, and every object must close itself with
    ``additionalProperties: false``. A schema straight off a Pydantic model has
    neither property, so the SDK's own ``transform_schema`` normalises it —
    which, better than dropping the constraints, restates them in each field's
    description so the model still sees them as guidance. The Pydantic model
    then re-applies them for real when the response is validated.
    """
    if response_schema is None:
        return None
    return {"format": {"type": "json_schema", "schema": transform_schema(dict(response_schema))}}


def to_message_params(messages: Sequence[Message]) -> list[MessageParam]:
    """Scout's messages in wire shape, tool blocks included."""
    return [
        {"role": message.role, "content": [_to_block_param(b) for b in message.content]}
        for message in messages
    ]


def _to_block_param(block: ContentBlock) -> ContentBlockParam:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": dict(block.arguments),
        }
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": block.content,
        "is_error": block.is_error,
    }


# --- Translation, back -----------------------------------------------------


def to_response(message: AnthropicMessage, *, response_schema: Any | None = None) -> LlmResponse:
    """Turn an API response into an :class:`LlmResponse`, or raise.

    Separated from the call so the mapping — including every way it refuses —
    can be tested without a client at all.
    """
    stop_reason = message.stop_reason

    usage = to_usage(message.usage)

    if stop_reason == "refusal":
        details = message.stop_details
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None) or "no explanation given"
        raise LlmRefusalError(
            f"the model declined the request: {explanation}",
            category=category,
            model=message.model,
            usage=usage,
        )

    if stop_reason is None or stop_reason not in _STOP_REASONS:
        raise LlmResponseError(
            f"unrecognised stop reason {stop_reason!r}; Scout branches on this value, "
            f"so treating it as 'finished' would report a truncated run as a complete one",
            model=message.model,
            usage=usage,
        )

    content = tuple(
        _from_block(block, model=message.model, usage=usage) for block in message.content
    )
    if not content:
        raise LlmEmptyResponseError(
            f"the model returned no content blocks (stop reason {stop_reason!r})",
            model=message.model,
            usage=usage,
        )

    json_output = None
    if response_schema is not None:
        json_output = structured_output(
            "\n".join(b.text for b in content if isinstance(b, TextBlock)),
            stop_reason=cast(StopReason, stop_reason),
            model=message.model,
            usage=usage,
        )

    return LlmResponse(
        model=message.model,
        content=content,
        stop_reason=cast(StopReason, stop_reason),
        usage=usage,
        json_output=json_output,
    )


def _from_block(block: AnthropicContentBlock, *, model: str, usage: Usage) -> ContentBlock:
    """One response block in Scout's own vocabulary.

    An unrecognised block type raises rather than being dropped: silently losing
    part of an answer is indistinguishable from the model having said less. The
    call's cost rides along, because the turn was billed whether or not Scout
    could read it.

    The block Scout is most likely to meet here is ``redacted_thinking``, which
    extended thinking emits when the reasoning is flagged. Nothing turns thinking
    on today; the node that does has to handle it.
    """
    if block.type == "text":
        return TextBlock(block.text)
    if block.type == "thinking":
        return ThinkingBlock(thinking=block.thinking, signature=block.signature)
    if block.type == "tool_use":
        return ToolUseBlock(
            id=block.id, name=block.name, arguments=dict(cast(Mapping[str, Any], block.input))
        )
    raise LlmResponseError(
        f"content block {block.type!r} in the assistant turn is one Scout does not model, "
        f"so part of this answer cannot be read",
        model=model,
        usage=usage,
    )


def to_usage(usage: AnthropicUsage) -> Usage:
    """Token counts in the shape :func:`~scout.llm.pricing.cost_usd` prices.

    Cache reads and writes are kept apart from plain input tokens because they
    bill at roughly a tenth and 1.25x of the input rate; folding them together
    would make the reported spend wrong in both directions.
    """
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
    )
