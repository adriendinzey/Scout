"""The Parse node, driven end to end by a scripted fake. No network, no database.

The vocabulary is a handful of made-up lookups rather than a snapshot's worth:
what is under test is that every name is checked against *some* vocabulary, not
which names London happens to have.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from anthropic import transform_schema
from pydantic import ValidationError

from scout.config import Settings
from scout.graph.parse import (
    MAX_ATTEMPTS,
    PASSTHROUGH_FIELDS,
    USAGE_LABEL,
    Grounding,
    ParseError,
    ParseOutput,
    ParseResult,
    UngroundedValue,
    UnsupportedConstraint,
    build_request,
    describe_validation_error,
    ground,
    parse_query,
    response_schema,
)
from scout.graph.prompts.parse import INSTRUCTIONS, render_vocabulary
from scout.llm.backend import (
    LlmApiError,
    LlmRefusalError,
    LlmRequest,
    LlmResponse,
    LlmResponseError,
    Message,
    TextBlock,
)
from scout.llm.fake_backend import (
    DEFAULT_FAKE_USAGE,
    FakeBackend,
    FakeTurn,
    error_turn,
    json_turn,
    refusal_turn,
)
from scout.llm.usage import UsageAccumulator
from scout.retrieval.filters import Filters
from scout.retrieval.query import where_clause

HACKNEY, ISLINGTON, CAMDEN = 7, 9, 3
ENTIRE_HOME = 1
RENTAL_UNIT = 4

GROUNDING = Grounding(
    city="London",
    neighbourhoods={"Hackney": HACKNEY, "Islington": ISLINGTON, "Camden": CAMDEN},
    room_types={"Entire home/apt": ENTIRE_HOME, "Private room": 2},
    property_types={"Entire rental unit": RENTAL_UNIT, "Private room in home": 5},
    amenities=frozenset({"Wifi", "Dedicated workspace"}),
)

# The column each indexed amenity would have, as the query builder is told it.
INDEXED_AMENITY_COLUMNS = {"Wifi": "has_wifi", "Dedicated workspace": "has_workspace"}

PARSE_MODEL = "claude-haiku-4-5"
SETTINGS = Settings(_env_file=None, model_parse=PARSE_MODEL)  # type: ignore[call-arg]

QUERY = "quiet flat for two in Hackney under 150 a night with wifi"


def body(**fields: Any) -> dict[str, Any]:
    """A response body with the required fields filled in and nothing constrained."""
    return {"semantic_query": "quiet flat", "unsupported_constraints": [], **fields}


def run(
    *turns: FakeTurn, query: str = QUERY, grounding: Grounding = GROUNDING
) -> tuple[ParseResult, FakeBackend, UsageAccumulator]:
    backend = FakeBackend(turns)
    usage = UsageAccumulator()
    result = parse_query(
        query, backend=backend, grounding=grounding, settings=SETTINGS, usage=usage
    )
    return result, backend, usage


def failing(*turns: FakeTurn) -> tuple[ParseError, FakeBackend, UsageAccumulator]:
    backend = FakeBackend(turns)
    usage = UsageAccumulator()
    with pytest.raises(ParseError) as caught:
        parse_query(QUERY, backend=backend, grounding=GROUNDING, settings=SETTINGS, usage=usage)
    return caught.value, backend, usage


# --- A good response -------------------------------------------------------


class TestAValidResponse:
    def test_becomes_filters_a_semantic_query_and_unsupported_constraints(self) -> None:
        result, _, _ = run(
            json_turn(
                body(
                    neighbourhoods=["Hackney"],
                    max_price=150,
                    min_accommodates=2,
                    amenities=["Wifi"],
                    semantic_query="quiet flat",
                    unsupported_constraints=[
                        {"constraint": "free in June", "reason": "dates are not searchable"}
                    ],
                )
            )
        )

        assert result.filters == Filters(
            neighbourhood_ids=(HACKNEY,),
            max_price=150,
            min_accommodates=2,
            amenities=("Wifi",),
        )
        assert result.semantic_query == "quiet flat"
        assert result.unsupported_constraints == (
            UnsupportedConstraint(constraint="free in June", reason="dates are not searchable"),
        )
        assert result.ungrounded == ()

    def test_names_become_ids_and_several_mean_any_of_them(self) -> None:
        result, _, _ = run(
            json_turn(
                body(
                    neighbourhoods=["Islington", "Hackney"],
                    room_types=["Entire home/apt"],
                    property_types=["Entire rental unit"],
                )
            )
        )

        assert result.filters.neighbourhood_ids == (HACKNEY, ISLINGTON)
        assert result.filters.room_type_ids == (ENTIRE_HOME,)
        assert result.filters.property_type_ids == (RENTAL_UNIT,)

    def test_a_filter_the_model_left_out_is_unconstrained(self) -> None:
        result, _, _ = run(json_turn(body()))

        assert result.filters.is_empty()
        assert result.filters.min_bedrooms is None

    def test_makes_exactly_one_call_and_prices_it(self) -> None:
        result, backend, usage = run(json_turn(body()))

        assert len(backend.calls) == 1
        assert [record.label for record in usage.calls] == [USAGE_LABEL]
        assert usage.total_cost_usd > 0
        assert len(result.attempts) == 1
        assert result.attempts[0].error is None

    def test_records_what_the_model_actually_said(self) -> None:
        payload = body(max_price=90)
        result, _, _ = run(json_turn(payload))

        assert json.loads(result.attempts[-1].raw_output) == payload

    def test_the_semantic_query_is_trimmed(self) -> None:
        result, _, _ = run(json_turn(body(semantic_query="  quiet flat \n")))

        assert result.semantic_query == "quiet flat"


class TestTheRequest:
    def test_uses_the_parse_model_from_settings(self) -> None:
        custom = Settings(_env_file=None, model_parse="claude-sonnet-5")  # type: ignore[call-arg]
        backend = FakeBackend([json_turn(body())])

        parse_query(
            QUERY,
            backend=backend,
            grounding=GROUNDING,
            settings=custom,
            usage=UsageAccumulator(),
        )

        assert backend.calls[0].model == "claude-sonnet-5"

    def test_is_one_user_turn_with_a_schema_and_no_tools(self) -> None:
        _, backend, _ = run(json_turn(body()))
        request = backend.calls[0]

        assert request.messages == (Message.user(QUERY),)
        assert request.tools == ()
        assert request.response_schema is not None
        assert request.cache_prefix

    def test_the_query_never_enters_the_cached_prefix(self) -> None:
        _, backend, _ = run(json_turn(body()))

        assert QUERY not in backend.calls[0].system

    def test_the_system_prompt_carries_every_name_in_the_vocabulary(self) -> None:
        system = build_request(QUERY, GROUNDING, model=PARSE_MODEL).system

        for names in GROUNDING.vocabularies().values():
            for name in names:
                assert f"- {name}\n" in system

    def test_a_blank_query_is_refused_before_anything_is_spent(self) -> None:
        backend = FakeBackend([json_turn(body())])

        with pytest.raises(ValueError, match="empty"):
            parse_query(
                "   ",
                backend=backend,
                grounding=GROUNDING,
                settings=SETTINGS,
                usage=UsageAccumulator(),
            )
        assert backend.calls == []


class TestTheGroundingIsReadOnly:
    """One instance serves every parse in the process; none may change it for the rest."""

    def test_its_mappings_cannot_be_edited_in_place(self) -> None:
        with pytest.raises(TypeError):
            GROUNDING.neighbourhoods["Atlantis"] = 99  # type: ignore[index]

    def test_editing_the_dict_it_was_built_from_changes_nothing(self) -> None:
        source = {"Hackney": HACKNEY}
        grounding = Grounding(
            city="London",
            neighbourhoods=source,
            room_types={},
            property_types={},
            amenities=frozenset(),
        )
        source["Atlantis"] = 99

        assert dict(grounding.neighbourhoods) == {"Hackney": HACKNEY}


# --- The cacheable prefix --------------------------------------------------


def shuffled_grounding() -> Grounding:
    """The same vocabulary as GROUNDING, every collection built in another order."""
    return Grounding(
        city="London",
        neighbourhoods=dict(reversed(GROUNDING.neighbourhoods.items())),
        room_types=dict(reversed(GROUNDING.room_types.items())),
        property_types=dict(reversed(GROUNDING.property_types.items())),
        amenities=frozenset(sorted(GROUNDING.amenities, reverse=True)),
    )


def prefix(request: LlmRequest) -> tuple[str, str]:
    """Everything a request sends ahead of its messages, as the bytes on the wire."""
    return request.system, json.dumps(request.response_schema)


class TestThePrefixIsByteStable:
    """Any byte that moves here silently empties the prompt cache."""

    def test_two_builds_of_the_grounding_render_identically(self) -> None:
        assert list(GROUNDING.neighbourhoods) != list(shuffled_grounding().neighbourhoods)

        assert GROUNDING.render().encode() == shuffled_grounding().render().encode()

    def test_two_builds_of_the_request_send_an_identical_prefix(self) -> None:
        first = build_request(QUERY, GROUNDING, model=PARSE_MODEL)
        second = build_request(QUERY, shuffled_grounding(), model=PARSE_MODEL)

        assert prefix(first) == prefix(second)

    def test_two_queries_share_one_prefix(self) -> None:
        first = build_request(QUERY, GROUNDING, model=PARSE_MODEL)
        second = build_request("a loft in Camden", GROUNDING, model=PARSE_MODEL)

        assert prefix(first) == prefix(second)
        assert first.messages != second.messages

    def test_names_are_rendered_in_sorted_order(self) -> None:
        rendered = GROUNDING.render()
        positions = [rendered.index(f"- {name}\n") for name in ("Camden", "Hackney", "Islington")]

        assert positions == sorted(positions)

    def test_the_renderer_sorts_whatever_order_it_is_handed(self) -> None:
        """It is public, so it cannot rely on a caller having sorted first."""
        forwards = render_vocabulary(
            city="London",
            neighbourhoods=["Camden", "Hackney"],
            room_types=["Private room", "Entire home/apt"],
            property_types=["Entire rental unit"],
            amenities=["Wifi", "Dedicated workspace"],
        )
        backwards = render_vocabulary(
            city="London",
            neighbourhoods=["Hackney", "Camden"],
            room_types=["Entire home/apt", "Private room"],
            property_types=["Entire rental unit"],
            amenities=["Dedicated workspace", "Wifi"],
        )

        assert forwards == backwards

    def test_a_retry_reuses_the_first_calls_prefix(self) -> None:
        _, backend, _ = run(json_turn("not json"), json_turn(body()))

        assert prefix(backend.calls[0]) == prefix(backend.calls[1])


# --- The response schema ---------------------------------------------------


class TestTheResponseSchema:
    def test_every_name_field_is_restricted_to_its_vocabulary(self) -> None:
        properties = response_schema(GROUNDING)["properties"]

        for field, names in GROUNDING.vocabularies().items():
            assert properties[field]["items"]["enum"] == names

    def test_offers_no_area_filter(self) -> None:
        """Coordinates the model recalled for a landmark would be a guess."""
        assert "near" not in response_schema(GROUNDING)["properties"]

    def test_an_empty_vocabulary_drops_its_field(self) -> None:
        """An empty enum is a schema nothing satisfies, so the field goes instead."""
        no_amenities = Grounding(
            city="London",
            neighbourhoods=GROUNDING.neighbourhoods,
            room_types=GROUNDING.room_types,
            property_types=GROUNDING.property_types,
            amenities=frozenset(),
        )

        assert "amenities" not in response_schema(no_amenities)["properties"]
        assert "amenities (0):\n- (none" in no_amenities.render()

    def test_requires_the_semantic_query_and_the_unsupported_list(self) -> None:
        required = set(response_schema(GROUNDING)["required"])

        assert required == {"semantic_query", "unsupported_constraints"}

    def test_sends_no_docstring_to_the_model(self) -> None:
        """A docstring edit must not move the cached prefix."""
        schema = response_schema(GROUNDING)

        assert "description" not in schema
        assert "description" not in schema["$defs"]["UnsupportedConstraint"]

    def test_sends_no_python_names_to_the_model(self) -> None:
        """Nor must renaming a class or a field's title."""
        assert '"title"' not in json.dumps(response_schema(GROUNDING))

    def test_sends_no_python_defaults_to_the_model(self) -> None:
        """The SDK folds a leftover `default` into the description as `{default: None}`."""
        transformed = json.dumps(transform_schema(response_schema(GROUNDING)))

        assert "default" not in transformed

    def test_survives_the_sdk_transform_the_real_backend_applies(self) -> None:
        transformed = transform_schema(response_schema(GROUNDING))

        assert transformed["additionalProperties"] is False
        assert transformed["properties"]["neighbourhoods"]["items"]["enum"] == sorted(
            GROUNDING.neighbourhoods
        )

    def test_mirrors_the_filters_model(self) -> None:
        """A new filter has to be either offered to the model or excluded on purpose."""
        translated = {"neighbourhood_ids", "room_type_ids", "property_type_ids", "amenities"}
        not_offered = {"near"}

        assert set(Filters.model_fields) == PASSTHROUGH_FIELDS | translated | not_offered
        for field in PASSTHROUGH_FIELDS:
            assert (
                ParseOutput.model_fields[field].annotation == Filters.model_fields[field].annotation
            )


# --- Grounding -------------------------------------------------------------


class TestAHallucinatedName:
    def test_neighbourhood_never_reaches_sql(self) -> None:
        result, _, _ = run(json_turn(body(neighbourhoods=["Hackney", "Atlantis"])))
        where = where_clause(result.filters, indexed_amenities=INDEXED_AMENITY_COLUMNS)

        assert result.filters.neighbourhood_ids == (HACKNEY,)
        assert where.params == (HACKNEY,)
        assert UngroundedValue("neighbourhoods", "Atlantis") in result.ungrounded

    def test_neighbourhood_is_reported_rather_than_dropped(self) -> None:
        result, _, _ = run(json_turn(body(neighbourhoods=["Atlantis"])))

        assert result.filters.neighbourhood_ids == ()
        assert result.unsupported_constraints == (
            UnsupportedConstraint(constraint="Atlantis", reason="not a neighbourhood Scout knows"),
        )

    def test_amenity_without_a_filter_never_reaches_sql(self) -> None:
        result, _, _ = run(json_turn(body(amenities=["Wifi", "Hot tub"])))
        where = where_clause(result.filters, indexed_amenities=INDEXED_AMENITY_COLUMNS)

        assert result.filters.amenities == ("Wifi",)
        assert where.post_filter_columns == ()
        assert "Hot tub" not in repr(where.params)
        assert result.unsupported_constraints == (
            UnsupportedConstraint(
                constraint="Hot tub", reason="Scout has no filter for this amenity"
            ),
        )

    def test_room_and_property_types_are_checked_too(self) -> None:
        result, _, _ = run(json_turn(body(room_types=["Yurt"], property_types=["Penthouse"])))

        assert result.filters.is_empty()
        assert result.ungrounded == (
            UngroundedValue("room_types", "Yurt"),
            UngroundedValue("property_types", "Penthouse"),
        )

    def test_a_name_must_match_exactly(self) -> None:
        """The schema enumerates exact spellings; a near miss is not one of them."""
        result, _, _ = run(json_turn(body(neighbourhoods=["hackney"])))

        assert result.filters.neighbourhood_ids == ()
        assert result.ungrounded == (UngroundedValue("neighbourhoods", "hackney"),)

    def test_is_not_reported_twice_when_the_model_already_flagged_it(self) -> None:
        flagged = {"constraint": "Hot tub", "reason": "no hot tub filter"}
        result, _, _ = run(
            json_turn(body(amenities=["Hot tub"], unsupported_constraints=[flagged]))
        )

        assert result.unsupported_constraints == (UnsupportedConstraint(**flagged),)
        assert result.ungrounded == (UngroundedValue("amenities", "Hot tub"),)

    def test_is_not_reported_twice_when_the_model_flagged_it_in_the_users_words(self) -> None:
        flagged = {"constraint": "somewhere in atlantis", "reason": "not a London area"}
        result, _, _ = run(
            json_turn(body(neighbourhoods=["Atlantis"], unsupported_constraints=[flagged]))
        )

        assert result.unsupported_constraints == (UnsupportedConstraint(**flagged),)

    def test_a_name_inside_another_word_does_not_count_as_reported(self) -> None:
        """'ac' is inside 'space'; treating that as covered would drop the ask entirely."""
        flagged = {"constraint": "a parking space for my car", "reason": "no parking filter"}
        result, _, _ = run(json_turn(body(amenities=["AC"], unsupported_constraints=[flagged])))

        assert [item.constraint for item in result.unsupported_constraints] == [
            "a parking space for my car",
            "AC",
        ]

    def test_a_name_ending_in_punctuation_is_still_recognised(self) -> None:
        """Both a word-boundary match and an unescaped pattern would miss this one."""
        flagged = {"constraint": "a TV (cable) in the bedroom", "reason": "no TV filter"}
        result, _, _ = run(
            json_turn(body(amenities=["TV (cable)"], unsupported_constraints=[flagged]))
        )

        assert result.unsupported_constraints == (UnsupportedConstraint(**flagged),)

    def test_a_repeated_name_is_one_hallucination(self) -> None:
        """The evaluation counts these, so a repeat must not count double."""
        result, _, _ = run(json_turn(body(neighbourhoods=["Atlantis", "Hackney", "Atlantis"])))

        assert result.ungrounded == (UngroundedValue("neighbourhoods", "Atlantis"),)
        assert [item.constraint for item in result.unsupported_constraints] == ["Atlantis"]

    def test_is_not_a_validation_error_so_costs_no_retry(self) -> None:
        result, backend, _ = run(json_turn(body(neighbourhoods=["Atlantis"])))

        assert len(backend.calls) == 1
        assert len(result.attempts) == 1


class TestUnsupportedAsks:
    def test_the_models_own_list_survives_in_order_with_rejected_names_after_it(self) -> None:
        asked = [
            {"constraint": "free from 3 to 7 June", "reason": "dates are not searchable"},
            {"constraint": "near the British Museum", "reason": "landmarks cannot be located"},
            {"constraint": "a sauna", "reason": "no filter for this amenity"},
        ]
        result, _, _ = run(
            json_turn(body(unsupported_constraints=asked, neighbourhoods=["Bloomsbury"]))
        )

        assert [item.constraint for item in result.unsupported_constraints] == [
            "free from 3 to 7 June",
            "near the British Museum",
            "a sauna",
            "Bloomsbury",
        ]

    def test_ground_is_pure_and_usable_without_a_backend(self) -> None:
        result = ground(body(neighbourhoods=["Camden"]), GROUNDING)

        assert result.filters.neighbourhood_ids == (CAMDEN,)
        assert result.attempts == ()


# --- Retry once ------------------------------------------------------------


class TestInvalidThenValid:
    def test_malformed_json_is_retried_and_the_second_answer_used(self) -> None:
        result, backend, _ = run(json_turn("{not json"), json_turn(body(max_price=120)))

        assert len(backend.calls) == 2
        assert result.filters.max_price == 120
        assert result.attempts[0].raw_output == "{not json"
        assert "not valid JSON" in (result.attempts[0].error or "")
        assert result.attempts[1].error is None

    def test_the_retry_shows_the_model_its_answer_and_what_was_wrong(self) -> None:
        bad = json.dumps(body(min_rating=7))
        _, backend, _ = run(json_turn(bad), json_turn(body(min_rating=4.5)))
        retry = backend.calls[1].messages

        assert retry[0] == Message.user(QUERY)
        assert retry[1] == Message.assistant(TextBlock(bad))
        assert retry[2].role == "user"
        feedback = retry[2].content[0]
        assert isinstance(feedback, TextBlock)
        assert "min_rating" in feedback.text
        assert "before anything was searched" in feedback.text

    def test_a_bound_filters_enforces_is_fed_back(self) -> None:
        """Bounds live on Filters, so a violation surfaces after the shape check."""
        result, backend, _ = run(
            json_turn(body(min_price=300, max_price=100)), json_turn(body(max_price=100))
        )

        assert "exceeds max_price" in (result.attempts[0].error or "")
        assert len(backend.calls) == 2

    def test_an_invented_field_is_fed_back(self) -> None:
        result, _, _ = run(json_turn(body(max_distance_km=2)), json_turn(body()))

        assert "max_distance_km" in (result.attempts[0].error or "")

    def test_a_blank_semantic_query_is_fed_back(self) -> None:
        result, _, _ = run(json_turn(body(semantic_query="  ")), json_turn(body()))

        assert "semantic_query" in (result.attempts[0].error or "")

    def test_a_blank_answer_is_not_replayed_as_an_empty_turn(self) -> None:
        """The API refuses an empty text block, so the retry would fail outright."""
        _, backend, _ = run(json_turn("   "), json_turn(body()))
        retry = backend.calls[1].messages

        assert [message.role for message in retry] == ["user", "user"]

    def test_both_calls_are_priced(self) -> None:
        """A rejected answer was billed, and the cost ceiling has to see it."""
        _, _, usage = run(json_turn("{not json"), json_turn(body()))

        assert [record.label for record in usage.calls] == [USAGE_LABEL, USAGE_LABEL]


class TestInvalidTwice:
    def test_raises_rather_than_returning_empty_filters(self) -> None:
        error, _, _ = failing(json_turn("{not json"), json_turn(body(min_rating=7)))

        assert QUERY in str(error)
        assert "min_rating" in str(error)

    def test_stops_after_exactly_one_retry(self) -> None:
        _, backend, _ = failing(*(json_turn("{not json") for _ in range(MAX_ATTEMPTS + 1)))

        assert MAX_ATTEMPTS == 2
        assert len(backend.calls) == MAX_ATTEMPTS
        assert backend.pending == 1

    def test_carries_every_raw_answer(self) -> None:
        error, _, _ = failing(json_turn("first"), json_turn("second"))

        assert error.query == QUERY
        assert [attempt.raw_output for attempt in error.attempts] == ["first", "second"]
        assert all(attempt.error for attempt in error.attempts)

    def test_both_failed_calls_are_priced(self) -> None:
        _, _, usage = failing(json_turn("first"), json_turn("second"))

        assert usage.call_count == MAX_ATTEMPTS


# --- Failures that are not the model's mistake ------------------------------


class ForgetsToDecode:
    """A backend that breaks its contract: a schema went out, no decoded body came back."""

    def __init__(self) -> None:
        self.calls: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        return LlmResponse(
            model=request.model,
            content=(TextBlock(json.dumps(body())),),
            stop_reason="end_turn",
            usage=DEFAULT_FAKE_USAGE,
            json_output=None,
        )


class TestFailuresThatAreNotRetried:
    def test_a_backend_that_returns_no_structured_output_is_not_blamed_on_the_model(
        self,
    ) -> None:
        """Feeding a backend bug back as the model's mistake would waste the retry."""
        backend = ForgetsToDecode()
        usage = UsageAccumulator()

        with pytest.raises(LlmResponseError, match="no structured output"):
            parse_query(QUERY, backend=backend, grounding=GROUNDING, settings=SETTINGS, usage=usage)
        assert len(backend.calls) == 1
        assert usage.call_count == 1

    def test_a_refusal_is_passed_on_and_priced(self) -> None:
        backend = FakeBackend([refusal_turn(), json_turn(body())])
        usage = UsageAccumulator()

        with pytest.raises(LlmRefusalError):
            parse_query(QUERY, backend=backend, grounding=GROUNDING, settings=SETTINGS, usage=usage)
        assert len(backend.calls) == 1
        assert usage.call_count == 1

    def test_a_body_cut_off_at_max_tokens_is_not_retried(self) -> None:
        """Asking again truncates in the same place."""
        truncated = FakeTurn(
            content=(TextBlock('{"semantic_query": "qu'),), stop_reason="max_tokens"
        )
        backend = FakeBackend([truncated, json_turn(body())])

        with pytest.raises(LlmResponseError, match="max_tokens"):
            parse_query(
                QUERY,
                backend=backend,
                grounding=GROUNDING,
                settings=SETTINGS,
                usage=UsageAccumulator(),
            )
        assert len(backend.calls) == 1

    def test_a_transport_failure_is_passed_on_unretried(self) -> None:
        """The client has already applied its own retries; a second layer would
        multiply the wait against the query's wall clock."""
        backend = FakeBackend(
            [error_turn(LlmApiError("overloaded", retryable=True, status_code=529))]
        )
        usage = UsageAccumulator()

        with pytest.raises(LlmApiError):
            parse_query(QUERY, backend=backend, grounding=GROUNDING, settings=SETTINGS, usage=usage)
        assert len(backend.calls) == 1
        assert usage.call_count == 0

    def test_a_transport_failure_on_the_retry_is_passed_on_too(self) -> None:
        backend = FakeBackend(
            [json_turn("{not json"), error_turn(LlmApiError("reset", retryable=True))]
        )

        with pytest.raises(LlmApiError):
            parse_query(
                QUERY,
                backend=backend,
                grounding=GROUNDING,
                settings=SETTINGS,
                usage=UsageAccumulator(),
            )
        assert len(backend.calls) == 2


# --- The prompt ------------------------------------------------------------


class TestThePromptStates:
    """What the model must be told, so a rewrite cannot quietly drop it."""

    @pytest.mark.parametrize(
        "statement",
        [
            "Unconstrained means leaving a filter out, never setting it to zero",
            "A listing with no value for a field satisfies no condition on it",
            "Prices are per night, in GBP",
            "Scout has no dates or availability",
            "An amenity that is not in the amenity vocabulary is an unsupported constraint",
            "A contradictory request is parsed as written",
            "No other vague word sets a bound",
        ],
    )
    def test(self, statement: str) -> None:
        assert statement in " ".join(INSTRUCTIONS.split())


def test_a_validation_error_is_described_without_links_or_noise() -> None:
    with pytest.raises(ValidationError) as caught:
        ParseOutput.model_validate({"semantic_query": "x", "min_price": "cheap"})

    described = describe_validation_error(caught.value)

    assert "min_price" in described
    assert "unsupported_constraints" in described
    assert "http" not in described
