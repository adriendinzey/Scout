"""The real backend's translation, caching, and error mapping.

Every test here drives a stub in place of the SDK client. Nothing in this file
opens a socket or reads an API key, which is why the suite is green with
ANTHROPIC_API_KEY unset.
"""

from __future__ import annotations

import json
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message as SdkMessage
from anthropic.types import RefusalStopDetails
from anthropic.types import TextBlock as SdkTextBlock
from anthropic.types import ThinkingBlock as SdkThinkingBlock
from anthropic.types import ToolUseBlock as SdkToolUseBlock
from anthropic.types import Usage as SdkUsage
from pydantic import BaseModel, Field, SecretStr

from scout.llm.anthropic_backend import (
    AnthropicBackend,
    to_message_params,
    to_output_config,
    to_response,
    to_system_blocks,
    to_tool_params,
    to_usage,
)
from scout.llm.backend import (
    LlmApiError,
    LlmEmptyResponseError,
    LlmRefusalError,
    LlmRequest,
    LlmResponseError,
    LlmSchemaError,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from scout.llm.pricing import Usage

GROUNDING = "You are Scout. Neighbourhoods are integer ids; never invent a column name."

SEARCH_TOOL = ToolDefinition(
    name="search_listings",
    description="Rank listings matching a filter set.",
    input_schema={
        "type": "object",
        "properties": {"limit": {"type": "integer", "minimum": 1}},
        "required": ["limit"],
    },
)


# --- A stub in place of the SDK client -------------------------------------


class StubMessages:
    def __init__(self, result: SdkMessage | Exception) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> SdkMessage:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class StubClient:
    def __init__(self, result: SdkMessage | Exception) -> None:
        self.messages = StubMessages(result)


def sdk_message(
    *,
    content: list[Any] | None = None,
    stop_reason: str | None = "end_turn",
    usage: SdkUsage | None = None,
    stop_details: RefusalStopDetails | None = None,
) -> SdkMessage:
    return SdkMessage(
        id="msg_stub",
        content=content if content is not None else [SdkTextBlock(type="text", text="hi")],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason=cast(Any, stop_reason),
        stop_details=stop_details,
        type="message",
        usage=usage or SdkUsage(input_tokens=10, output_tokens=5),
    )


def backend_for(result: SdkMessage | Exception) -> tuple[AnthropicBackend, StubClient]:
    stub = StubClient(result)
    backend = AnthropicBackend(SecretStr("unused"), client=cast(anthropic.Anthropic, stub))
    return backend, stub


def request(**kwargs: Any) -> LlmRequest:
    defaults: dict[str, Any] = {
        "model": "claude-haiku-4-5",
        "system": GROUNDING,
        "messages": (Message.user("quiet place near the water"),),
    }
    return LlmRequest(**{**defaults, **kwargs})


class StubHttpResponse:
    """Only what the SDK's exception classes read off a response.

    Stubbed rather than built from the SDK's HTTP library, because which library
    that is has changed before and this test is about Scout's error mapping.
    """

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.request = None


def api_error(status: int) -> anthropic.APIStatusError:
    return anthropic.APIStatusError("boom", response=cast(Any, StubHttpResponse(status)), body=None)


# --- The cacheable prefix --------------------------------------------------


def test_the_stable_prefix_is_marked_cacheable() -> None:
    blocks = to_system_blocks(GROUNDING, cache=True)

    assert blocks == [{"type": "text", "text": GROUNDING, "cache_control": {"type": "ephemeral"}}]


def test_caching_can_be_turned_off_to_measure_what_it_is_worth() -> None:
    assert to_system_blocks(GROUNDING, cache=False) == [{"type": "text", "text": GROUNDING}]


def test_an_empty_system_prompt_produces_no_blocks() -> None:
    assert to_system_blocks("", cache=True) == []


def test_the_volatile_query_stays_out_of_the_cached_prefix() -> None:
    """The breakpoint sits on the system block; the user's query is after it."""
    backend, stub = backend_for(sdk_message())
    backend.complete(request(tools=(SEARCH_TOOL,)))

    system = stub.messages.kwargs["system"]
    messages = stub.messages.kwargs["messages"]
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    assert "quiet place near the water" not in json.dumps(system)
    assert not any("cache_control" in block for m in messages for block in m["content"])


def test_the_prefix_is_byte_identical_across_two_builds() -> None:
    """Any drift here invalidates the cache silently; the only symptom is cost."""
    first = json.dumps(
        {
            "tools": to_tool_params((SEARCH_TOOL,)),
            "system": to_system_blocks(GROUNDING, cache=True),
        }
    )
    second = json.dumps(
        {
            "tools": to_tool_params((SEARCH_TOOL,)),
            "system": to_system_blocks(GROUNDING, cache=True),
        }
    )

    assert first == second


def test_tool_definitions_keep_the_order_they_were_given() -> None:
    other = ToolDefinition(name="count_matches", description="Count.", input_schema={})
    assert [t["name"] for t in to_tool_params((SEARCH_TOOL, other))] == [
        "search_listings",
        "count_matches",
    ]


def test_tool_definitions_carry_their_schema() -> None:
    (param,) = to_tool_params((SEARCH_TOOL,))

    assert param["description"] == "Rank listings matching a filter set."
    assert param["input_schema"]["properties"]["limit"]["type"] == "integer"


def test_no_tools_and_no_schema_are_omitted_from_the_call() -> None:
    backend, stub = backend_for(sdk_message())
    backend.complete(request())

    assert stub.messages.kwargs["tools"] is anthropic.omit
    assert stub.messages.kwargs["output_config"] is anthropic.omit


# --- Structured output -----------------------------------------------------


def test_structured_output_is_requested_as_a_json_schema() -> None:
    config = to_output_config({"type": "object", "properties": {"max_price": {"type": "number"}}})

    assert config is not None
    assert config["format"]["type"] == "json_schema"
    assert config["format"]["schema"]["properties"] == {"max_price": {"type": "number"}}


def test_the_schema_is_normalised_before_it_is_sent() -> None:
    """Pydantic emits constraints the schema compiler refuses."""
    config = to_output_config(
        {"type": "object", "properties": {"guests": {"type": "integer", "minimum": 1}}}
    )

    assert config is not None
    sent = config["format"]["schema"]
    assert "minimum" not in sent["properties"]["guests"]
    assert sent["additionalProperties"] is False


def test_a_constraint_is_restated_to_the_model_rather_than_lost() -> None:
    """The compiler will not enforce it, so the model is told about it instead."""
    config = to_output_config(
        {"type": "object", "properties": {"guests": {"type": "integer", "minimum": 1}}}
    )

    assert config is not None
    assert "minimum: 1" in config["format"]["schema"]["properties"]["guests"]["description"]


def test_a_schema_off_a_pydantic_model_is_accepted_whole() -> None:
    """The case this exists for: a model written the way the Parse node will."""

    class Window(BaseModel):
        nights: int = Field(ge=1, le=30)

    class ParsedQuery(BaseModel):
        semantic_query: str = Field(max_length=200)
        max_price: float | None = Field(default=None, gt=0)
        accommodates: int = Field(default=1, ge=1)
        neighbourhoods: list[str] = Field(default_factory=list)
        window: Window | None = None

    config = to_output_config(ParsedQuery.model_json_schema())

    assert config is not None
    sent = config["format"]["schema"]
    assert not _has_unsupported(sent)
    assert sent["additionalProperties"] is False
    assert sent["$defs"]["Window"]["additionalProperties"] is False
    assert set(sent["properties"]) == {
        "semantic_query",
        "max_price",
        "accommodates",
        "neighbourhoods",
        "window",
    }


#: What the structured-output compiler refuses: numeric, string, and the richer
#: array and object constraints, plus `default`.
UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "default",
    }
)


def _has_unsupported(node: object) -> bool:
    if isinstance(node, dict):
        return any(key in UNSUPPORTED_KEYWORDS for key in node) or any(
            _has_unsupported(value) for value in node.values()
        )
    if isinstance(node, list):
        return any(_has_unsupported(item) for item in node)
    return False


def test_a_schema_request_comes_back_decoded() -> None:
    body = json.dumps({"max_price": 200})
    backend, _ = backend_for(sdk_message(content=[SdkTextBlock(type="text", text=body)]))

    response = backend.complete(request(response_schema={"type": "object"}))
    assert response.json_output == {"max_price": 200}


def test_prose_where_json_was_asked_for_raises() -> None:
    backend, _ = backend_for(
        sdk_message(content=[SdkTextBlock(type="text", text="about two hundred pounds")])
    )

    with pytest.raises(LlmSchemaError):
        backend.complete(request(response_schema={"type": "object"}))


# --- Messages out, response back -------------------------------------------


def test_tool_use_and_tool_result_blocks_round_trip_to_the_wire() -> None:
    messages = (
        Message.user("under 150"),
        Message.assistant(ToolUseBlock(id="toolu_1", name="count_matches", arguments={"x": 1})),
        Message.tool_results(ToolResultBlock("toolu_1", '{"count": 3}')),
    )

    params = to_message_params(messages)

    assert params[1]["role"] == "assistant"
    assert params[1]["content"] == [
        {"type": "tool_use", "id": "toolu_1", "name": "count_matches", "input": {"x": 1}}
    ]
    assert params[2]["role"] == "user"
    assert params[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": '{"count": 3}',
            "is_error": False,
        }
    ]


def test_a_thinking_block_is_replayed_verbatim() -> None:
    params = to_message_params((Message.assistant(ThinkingBlock("why not", "sig-abc")),))

    assert params[0]["content"] == [
        {"type": "thinking", "thinking": "why not", "signature": "sig-abc"}
    ]


def test_a_structured_tool_error_goes_back_flagged() -> None:
    params = to_message_params(
        (Message.tool_results(ToolResultBlock("toolu_1", "unknown field", is_error=True)),)
    )

    assert params[0]["content"][0]["is_error"] is True


def test_a_tool_call_comes_back_with_its_arguments_and_stop_reason() -> None:
    message = sdk_message(
        content=[
            SdkToolUseBlock(type="tool_use", id="toolu_9", name="field_stats", input={"f": "price"})
        ],
        stop_reason="tool_use",
    )

    response = to_response(message)

    assert response.wants_tools
    assert response.tool_uses == (
        ToolUseBlock(id="toolu_9", name="field_stats", arguments={"f": "price"}),
    )


def test_text_and_thinking_come_back_as_scouts_own_blocks() -> None:
    message = sdk_message(
        content=[
            SdkThinkingBlock(type="thinking", thinking="considering", signature="sig"),
            SdkTextBlock(type="text", text="three places fit"),
        ]
    )

    response = to_response(message)

    assert response.content == (ThinkingBlock("considering", "sig"), TextBlock("three places fit"))
    assert response.text == "three places fit"


# --- Usage -----------------------------------------------------------------


def test_cache_reads_and_writes_are_captured_separately() -> None:
    usage = to_usage(
        SdkUsage(
            input_tokens=12,
            output_tokens=34,
            cache_read_input_tokens=5_000,
            cache_creation_input_tokens=600,
        )
    )

    assert usage == Usage(
        input_tokens=12, output_tokens=34, cache_read_tokens=5_000, cache_write_tokens=600
    )


def test_absent_cache_counts_are_zero_not_none() -> None:
    assert to_usage(SdkUsage(input_tokens=1, output_tokens=2)) == Usage(
        input_tokens=1, output_tokens=2
    )


def test_usage_reaches_the_caller_on_a_real_call() -> None:
    backend, _ = backend_for(
        sdk_message(usage=SdkUsage(input_tokens=7, output_tokens=3, cache_read_input_tokens=900))
    )

    assert backend.complete(request()).usage.cache_read_tokens == 900


# --- Failures, none of which are an empty result ---------------------------


@pytest.mark.parametrize("status", [429, 500, 503, 408, 409])
def test_a_retryable_api_failure_says_so(status: int) -> None:
    backend, _ = backend_for(api_error(status))

    with pytest.raises(LlmApiError) as caught:
        backend.complete(request())
    assert caught.value.retryable is True
    assert caught.value.status_code == status


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_a_permanent_api_failure_says_so(status: int) -> None:
    backend, _ = backend_for(api_error(status))

    with pytest.raises(LlmApiError) as caught:
        backend.complete(request())
    assert caught.value.retryable is False


def test_a_connection_failure_is_retryable() -> None:
    backend, _ = backend_for(anthropic.APIConnectionError(request=cast(Any, None)))

    with pytest.raises(LlmApiError) as caught:
        backend.complete(request())
    assert caught.value.retryable is True


def test_a_timeout_is_retryable() -> None:
    backend, _ = backend_for(anthropic.APITimeoutError(request=cast(Any, None)))

    with pytest.raises(LlmApiError) as caught:
        backend.complete(request())
    assert caught.value.retryable is True


def test_a_refusal_raises_with_its_category_and_its_cost() -> None:
    message = sdk_message(
        stop_reason="refusal",
        stop_details=RefusalStopDetails(type="refusal", category="cyber", explanation="no"),
        usage=SdkUsage(input_tokens=1_200, output_tokens=8),
    )
    backend, _ = backend_for(message)

    with pytest.raises(LlmRefusalError) as caught:
        backend.complete(request())
    assert caught.value.category == "cyber"
    assert caught.value.usage == Usage(input_tokens=1_200, output_tokens=8)
    assert caught.value.model == "claude-haiku-4-5"


def test_a_truncated_schema_response_names_max_tokens() -> None:
    message = sdk_message(
        content=[SdkTextBlock(type="text", text='{"max_price": 2')],
        stop_reason="max_tokens",
    )
    backend, _ = backend_for(message)

    with pytest.raises(LlmResponseError, match="max_tokens") as caught:
        backend.complete(request(response_schema={"type": "object"}))
    assert not isinstance(caught.value, LlmSchemaError)


def test_an_empty_turn_raises_rather_than_returning_no_content() -> None:
    backend, _ = backend_for(sdk_message(content=[]))

    with pytest.raises(LlmEmptyResponseError):
        backend.complete(request())


def test_an_unrecognised_stop_reason_raises_rather_than_reading_as_finished() -> None:
    """A stop reason the API grows later must not silently read as 'finished'."""
    message = sdk_message()
    message.stop_reason = cast(Any, "something_the_api_added_later")

    with pytest.raises(LlmResponseError, match="unrecognised stop reason"):
        to_response(message)


def test_a_missing_stop_reason_raises() -> None:
    with pytest.raises(LlmResponseError):
        to_response(sdk_message(stop_reason=None))


def test_a_truncated_turn_reports_max_tokens_rather_than_looking_complete() -> None:
    response = to_response(sdk_message(stop_reason="max_tokens"))

    assert response.stop_reason == "max_tokens"
    assert not response.finished


def test_an_unreadable_content_block_raises_rather_than_being_dropped() -> None:
    class UnknownBlock:
        type = "some_future_block"

    message = sdk_message()
    message.content = cast(Any, [UnknownBlock()])

    with pytest.raises(LlmResponseError, match="unsupported content block"):
        to_response(message)
