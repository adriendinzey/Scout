"""The fake has to be trustworthy, because every other test depends on it."""

from __future__ import annotations

import pytest

from scout.llm.backend import (
    LlmApiError,
    LlmBackend,
    LlmEmptyResponseError,
    LlmRefusalError,
    LlmRequest,
    LlmResponseError,
    LlmSchemaError,
    Message,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from scout.llm.fake_backend import (
    DEFAULT_FAKE_USAGE,
    FakeBackend,
    FakeTurn,
    ScriptExhaustedError,
    empty_turn,
    error_turn,
    json_turn,
    refusal_turn,
    text_turn,
    tool_use_turn,
)
from scout.llm.pricing import Usage

SEARCH_TOOL = ToolDefinition(
    name="search_listings",
    description="Rank listings matching a filter set.",
    input_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
)


def ask(
    text: str = "quiet place near the water for two",
    *,
    tools: tuple[ToolDefinition, ...] = (),
    response_schema: dict[str, object] | None = None,
) -> LlmRequest:
    """One request, with only the parts a test cares about spelled out."""
    return LlmRequest(
        model="claude-haiku-4-5",
        system="You are Scout.",
        messages=(Message.user(text),),
        tools=tools,
        response_schema=response_schema,
    )


def test_fake_satisfies_the_backend_protocol() -> None:
    assert isinstance(FakeBackend(), LlmBackend)


def test_a_node_can_be_written_against_the_protocol() -> None:
    """The acceptance criterion: no `anthropic` import anywhere in sight."""

    def parse_node(backend: LlmBackend, query: str) -> str:
        return backend.complete(ask(query)).text

    assert parse_node(FakeBackend([text_turn("parsed")]), "anything") == "parsed"


def test_same_script_gives_the_same_answers() -> None:
    first = FakeBackend([text_turn("one"), text_turn("two")])
    second = FakeBackend([text_turn("one"), text_turn("two")])

    assert [first.complete(ask()).text for _ in range(2)] == [
        second.complete(ask()).text for _ in range(2)
    ]


def test_turns_come_back_in_scripted_order() -> None:
    backend = FakeBackend([text_turn("first"), text_turn("second")])

    assert backend.complete(ask()).text == "first"
    assert backend.complete(ask()).text == "second"
    assert backend.pending == 0


def test_running_off_the_end_of_the_script_raises() -> None:
    backend = FakeBackend([text_turn("only one")])
    backend.complete(ask())

    with pytest.raises(ScriptExhaustedError, match="no scripted turn"):
        backend.complete(ask())


def test_script_exhaustion_is_not_an_llm_error() -> None:
    """A node's `except LlmError` must not swallow a test-authoring mistake."""
    from scout.llm.backend import LlmError

    assert not issubclass(ScriptExhaustedError, LlmError)


def test_requests_are_recorded_for_assertions() -> None:
    backend = FakeBackend([text_turn("ok")])
    request = ask("near the canal")
    backend.complete(request)

    assert backend.calls == [request]


def test_the_recorded_requests_show_which_part_moved() -> None:
    """The fake builds no prefix of its own, so this is about the record: a test
    asserting cache stability needs the stable and volatile parts kept apart.
    The prefix is built — and asserted byte-identical — in the real backend."""
    backend = FakeBackend([text_turn("a"), text_turn("b")])
    backend.complete(ask("first query", tools=(SEARCH_TOOL,)))
    backend.complete(ask("second query", tools=(SEARCH_TOOL,)))

    first, second = backend.calls
    assert first.system == second.system
    assert first.tools == second.tools
    assert first.messages != second.messages  # only the volatile part moved


# --- The multi-turn tool exchange ------------------------------------------


def test_scripts_a_whole_tool_use_conversation() -> None:
    """Two tool calls, then an answer — an agent loop with no network."""
    backend = FakeBackend(
        [
            tool_use_turn("count_matches", {"max_price": 150}, text="Checking how strict this is."),
            tool_use_turn("search_listings", {"max_price": 180, "limit": 10}),
            text_turn("Three places fit, all under 180 a night."),
        ]
    )

    messages: list[Message] = [Message.user("under 150 a night")]
    tool_names: list[str] = []
    for _ in range(3):
        response = backend.complete(
            LlmRequest(
                model="claude-haiku-4-5",
                system="You are Scout.",
                messages=tuple(messages),
                tools=(SEARCH_TOOL,),
            )
        )
        messages.append(response.as_message())
        if not response.wants_tools:
            break
        tool_names.extend(call.name for call in response.tool_uses)
        messages.append(
            Message.tool_results(
                *(ToolResultBlock(call.id, '{"count": 3}') for call in response.tool_uses)
            )
        )

    assert tool_names == ["count_matches", "search_listings"]
    assert messages[-1].content[0] == TextBlock("Three places fit, all under 180 a night.")
    assert backend.pending == 0


def test_stop_reason_tells_a_tool_call_from_a_finished_turn() -> None:
    backend = FakeBackend([tool_use_turn("count_matches", {}), text_turn("done")])

    asked = backend.complete(ask())
    assert asked.stop_reason == "tool_use"
    assert asked.wants_tools and not asked.finished

    finished = backend.complete(ask())
    assert finished.stop_reason == "end_turn"
    assert finished.finished and not finished.wants_tools


def test_tool_use_and_tool_result_blocks_round_trip() -> None:
    backend = FakeBackend([tool_use_turn("field_stats", {"field": "price"}), text_turn("ok")])

    call = backend.complete(ask()).tool_uses[0]
    result = ToolResultBlock(tool_use_id=call.id, content='{"median": 180}')
    backend.complete(
        LlmRequest(
            model="claude-haiku-4-5",
            system="You are Scout.",
            messages=(
                Message.user("anything"),
                Message.assistant(call),
                Message.tool_results(result),
            ),
        )
    )

    replayed = backend.calls[-1].messages
    assert replayed[1].content == (call,)
    assert replayed[2].content == (result,)
    assert replayed[2].role == "user"  # results go back as one user turn


def test_tool_use_ids_are_deterministic_and_unique() -> None:
    backend = FakeBackend([tool_use_turn("count_matches", {}), tool_use_turn("field_stats", {})])

    ids = [backend.complete(ask()).tool_uses[0].id for _ in range(2)]
    assert ids == ["toolu_fake_0001", "toolu_fake_0002"]


def test_an_explicit_tool_use_id_is_kept() -> None:
    backend = FakeBackend([tool_use_turn("count_matches", {}, tool_use_id="toolu_mine")])

    assert backend.complete(ask()).tool_uses[0].id == "toolu_mine"


def test_the_sentence_alongside_a_tool_call_is_kept() -> None:
    backend = FakeBackend([tool_use_turn("count_matches", {}, text="Checking the count first.")])

    response = backend.complete(ask())
    assert response.text == "Checking the count first."
    assert isinstance(response.content[0], TextBlock)
    assert isinstance(response.content[1], ToolUseBlock)


# --- The error cases -------------------------------------------------------


def test_structured_output_comes_back_decoded() -> None:
    backend = FakeBackend([json_turn({"max_price": 200, "neighbourhoods": ["Hackney"]})])

    response = backend.complete(ask(response_schema={"type": "object"}))
    assert response.json_output == {"max_price": 200, "neighbourhoods": ["Hackney"]}


def test_json_output_is_none_when_no_schema_was_requested() -> None:
    assert FakeBackend([text_turn("prose")]).complete(ask()).json_output is None


def test_unparseable_structured_output_raises_rather_than_returning_nothing() -> None:
    backend = FakeBackend([json_turn("here are your filters: maybe 200?")])

    with pytest.raises(LlmSchemaError, match="not valid JSON") as caught:
        backend.complete(ask(response_schema={"type": "object"}))
    assert caught.value.raw_output == "here are your filters: maybe 200?"


def test_a_json_array_is_not_an_acceptable_object() -> None:
    backend = FakeBackend([json_turn("[1, 2, 3]")])

    with pytest.raises(LlmSchemaError, match="expected a JSON object"):
        backend.complete(ask(response_schema={"type": "object"}))


def test_a_refusal_raises_rather_than_looking_like_an_empty_answer() -> None:
    backend = FakeBackend([refusal_turn(category="cyber")])

    with pytest.raises(LlmRefusalError) as caught:
        backend.complete(ask())
    assert caught.value.category == "cyber"


def test_a_truncated_schema_response_is_not_a_retryable_schema_error() -> None:
    backend = FakeBackend(
        [FakeTurn(content=(TextBlock('{"max_price": 2'),), stop_reason="max_tokens")]
    )

    with pytest.raises(LlmResponseError, match="max_tokens") as caught:
        backend.complete(ask(response_schema={"type": "object"}))
    assert not isinstance(caught.value, LlmSchemaError)


def test_a_failed_turn_still_reports_the_tokens_it_burned() -> None:
    usage = Usage(input_tokens=2_000, output_tokens=8)
    backend = FakeBackend([refusal_turn(usage=usage)])

    with pytest.raises(LlmRefusalError) as caught:
        backend.complete(ask())
    assert caught.value.usage == usage
    assert caught.value.model == "claude-haiku-4-5"


def test_an_empty_turn_raises() -> None:
    with pytest.raises(LlmEmptyResponseError):
        FakeBackend([empty_turn()]).complete(ask())


# --- Usage and routing -----------------------------------------------------


def test_every_turn_reports_non_zero_usage_by_default() -> None:
    """A fake that billed nothing would let a broken cost accumulator pass."""
    response = FakeBackend([text_turn("ok")]).complete(ask())

    assert response.usage == DEFAULT_FAKE_USAGE
    assert response.usage.input_tokens > 0


def test_a_turn_can_script_its_own_usage() -> None:
    usage = Usage(input_tokens=9_000, output_tokens=400, cache_read_tokens=8_000)
    response = FakeBackend([text_turn("ok", usage=usage)]).complete(ask())

    assert response.usage == usage


def test_a_router_answers_based_on_the_request() -> None:
    def route(request: LlmRequest) -> FakeTurn:
        asked = "\n".join(b.text for b in request.messages[-1].content if isinstance(b, TextBlock))
        return tool_use_turn("count_matches", {}) if "how many" in asked else text_turn("no idea")

    backend = FakeBackend(router=route)

    assert backend.complete(ask("how many are there")).wants_tools
    assert backend.complete(ask("tell me a story")).text == "no idea"


def test_a_transport_failure_can_be_scripted() -> None:
    """So the retry-on-transport-failure path has an offline driver."""
    failure = LlmApiError("connection reset", retryable=True)
    backend = FakeBackend([error_turn(failure), text_turn("worked on the retry")])

    with pytest.raises(LlmApiError) as caught:
        backend.complete(ask())
    assert caught.value is failure
    assert backend.complete(ask()).text == "worked on the retry"


def test_a_script_and_a_router_together_are_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        FakeBackend([text_turn("x")], router=lambda _: text_turn("y"))
