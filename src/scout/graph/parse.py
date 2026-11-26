"""The Parse node: one request in, a validated and grounded filter set out.

Parse is deliberately not agentic. It makes one schema-constrained call, and
nothing it returns goes anywhere until it has passed two checks:

1. **Validation.** The body is validated into :class:`ParseOutput` and then into
   :class:`~scout.retrieval.filters.Filters` itself, so a rating of 7 or a price
   floor above its ceiling is caught while the model can still fix it. An
   invalid response is retried once with the problem fed back; a second failure
   raises :class:`ParseError`. It never falls back to ``Filters()``, which would
   search the whole corpus and present whatever came back as an answer.
2. **Grounding.** Every name the model emitted is looked up in the vocabulary
   the database actually holds. A neighbourhood or amenity that is not there is
   a hallucination: it is moved to ``unsupported_constraints``, recorded in
   ``ungrounded``, and never reaches SQL.

The model works in names, never lookup ids. Ids are an artefact of load order;
names are what "wifi" or "Hackney" can be mapped onto, and the schema enumerates
them, so the API's constrained decoding cannot produce one outside the list.

Parse never emits ``near``. Scout has no gazetteer, so any coordinates the model
offered for "near the British Museum" would be its own recollection, and a
radius drawn around a guess would quietly exclude real matches. A named place is
an unsupported constraint, and its words stay in the semantic query.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Final

import psycopg
from psycopg.rows import TupleRow
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pydantic.config import JsonDict

from scout.config import Settings
from scout.data.parsers import OTHER_PROPERTY_TYPE
from scout.graph.prompts.parse import render_vocabulary, retry_feedback, system_prompt
from scout.llm.backend import (
    LlmBackend,
    LlmRequest,
    LlmResponseError,
    LlmSchemaError,
    Message,
    TextBlock,
)
from scout.llm.usage import UsageAccumulator
from scout.retrieval.filters import Filters

logger = logging.getLogger(__name__)

Connection = psycopg.Connection[TupleRow]

#: The first call and exactly one retry with the problem fed back. A third try
#: mostly buys a second identical mistake, at the price of another call.
MAX_ATTEMPTS: Final = 2

#: The label this node's calls are priced under.
USAGE_LABEL: Final = "parse"

_CONNECT_TIMEOUT_S: Final = 5


# --- Errors ----------------------------------------------------------------


class GroundingError(RuntimeError):
    """The vocabulary could not be read, or the database holds none to read."""


@dataclass(frozen=True, slots=True)
class ParseAttempt:
    """One call: what the model said, and why it was rejected if it was."""

    raw_output: str
    error: str | None = None


class ParseError(Exception):
    """Two responses in a row failed validation.

    Carries every attempt's raw output, because when a parse goes wrong the
    first question is always what the model actually said.
    """

    def __init__(self, message: str, *, query: str, attempts: tuple[ParseAttempt, ...]) -> None:
        super().__init__(message)
        self.query = query
        self.attempts = attempts


# --- The vocabulary --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Lookup:
    #: The Filters field the names' ids go into.
    filters_field: str
    #: What the answer says about a name the vocabulary does not contain.
    ungrounded_reason: str


# The response fields whose values are lookup names. The one place they are
# listed: the vocabulary, the grounding check, and the conversion to Filters
# all read this table, so a lookup cannot be added to one and missed in another.
_LOOKUPS: Final[Mapping[str, _Lookup]] = {
    "neighbourhoods": _Lookup("neighbourhood_ids", "not a neighbourhood Scout knows"),
    "room_types": _Lookup("room_type_ids", "not a room type Scout knows"),
    "property_types": _Lookup("property_type_ids", "not a property type Scout knows"),
}

_AMENITY_UNGROUNDED_REASON: Final = "Scout has no filter for this amenity"


@dataclass(frozen=True)
class Grounding:
    """Every name a filter may use, as the database holds it.

    Lookup names map to their ids. ``amenities`` holds only the amenities with
    an indexed boolean column: the full list runs to thousands of names, far too
    many to ground a prompt in, and an amenity without a column cannot be pushed
    into the index anyway.

    Read-only all the way down. One instance is shared by every parse in the
    process (see :func:`grounding_for`), so a caller editing a mapping in place
    would change the vocabulary, and the cached prefix, under everyone else.
    """

    city: str
    neighbourhoods: Mapping[str, int]
    room_types: Mapping[str, int]
    property_types: Mapping[str, int]
    amenities: frozenset[str]

    def __post_init__(self) -> None:
        for field in _LOOKUPS:
            # Frozen, so the copy has to go round the dataclass's own guard.
            object.__setattr__(self, field, MappingProxyType(dict(getattr(self, field))))
        object.__setattr__(self, "amenities", frozenset(self.amenities))

    def lookup(self, field: str) -> Mapping[str, int]:
        """The name -> id map behind one of the response's lookup fields."""
        vocabulary: Mapping[str, int] = getattr(self, field)
        return vocabulary

    def vocabularies(self) -> dict[str, list[str]]:
        """Each name list, sorted, keyed by the response field that draws from it."""
        names = {field: sorted(self.lookup(field)) for field in _LOOKUPS}
        return {**names, "amenities": sorted(self.amenities)}

    def render(self) -> str:
        """The grounding block for the system prompt."""
        return render_vocabulary(city=self.city, **self.vocabularies())


_SNAPSHOT_CITY: Final = "SELECT city FROM data_snapshot"
_NEIGHBOURHOODS: Final = "SELECT name, id FROM neighbourhoods"
_ROOM_TYPES: Final = "SELECT name, id FROM room_types"
# The bucket rare property types are collapsed into is a load artefact, not a
# kind of place. Offered to the model it would become the answer to "a
# penthouse", and match everything that happened to be rare.
_PROPERTY_TYPES: Final = "SELECT name, id FROM property_types WHERE name <> %s"
_INDEXED_AMENITIES: Final = "SELECT name FROM amenities WHERE is_indexed"


def load_grounding(conn: Connection) -> Grounding:
    """Read the vocabulary from the lookup tables.

    Raises:
        GroundingError: if a query fails, or if no snapshot has been loaded. An
            empty vocabulary is refused rather than rendered: every request
            would then come back "unsupported", which reads like a parse result
            instead of the missing data it is.
    """
    try:
        snapshot = conn.execute(_SNAPSHOT_CITY).fetchone()
        neighbourhoods = _names_to_ids(conn.execute(_NEIGHBOURHOODS).fetchall())
        room_types = _names_to_ids(conn.execute(_ROOM_TYPES).fetchall())
        property_types = _names_to_ids(
            conn.execute(_PROPERTY_TYPES, (OTHER_PROPERTY_TYPE,)).fetchall()
        )
        amenities = frozenset(str(name) for (name,) in conn.execute(_INDEXED_AMENITIES))
    except psycopg.Error as exc:
        raise GroundingError(f"could not read the parse vocabulary: {exc}") from exc

    if snapshot is None:
        raise GroundingError(
            "no snapshot is loaded (data_snapshot is empty), so there is no vocabulary "
            "to ground a parse in; run `scout data load` first"
        )
    if not neighbourhoods:
        raise GroundingError("the neighbourhoods table is empty; the snapshot load is incomplete")
    return Grounding(
        city=str(snapshot[0]),
        neighbourhoods=neighbourhoods,
        room_types=room_types,
        property_types=property_types,
        amenities=amenities,
    )


def _names_to_ids(rows: Iterable[tuple[Any, ...]]) -> dict[str, int]:
    return {str(name): int(lookup_id) for name, lookup_id in rows}


@functools.cache
def grounding_for(database_url: str) -> Grounding:
    """The vocabulary for one database, loaded once per process.

    It renders into the cached prefix of every parse, so it must not move under
    a running process: reloading per query would cost four queries and buy
    nothing, and a reload in the middle of an evaluation would split one run
    across two prompts. Anything that reloads the data in-process calls
    ``grounding_for.cache_clear()``.

    Raises:
        GroundingError: if the database cannot be reached or read.
    """
    try:
        conn = psycopg.connect(database_url, connect_timeout=_CONNECT_TIMEOUT_S)
    except psycopg.Error as exc:
        raise GroundingError(f"could not connect to read the parse vocabulary: {exc}") from exc
    with conn:
        return load_grounding(conn)


# --- The response schema ---------------------------------------------------


def _schema_without_description(schema: JsonDict) -> None:
    # The docstring is written for whoever maintains this file. The model is
    # instructed by the system prompt, and editing a docstring must not move the
    # prefix every parse is cached on.
    schema.pop("description", None)


class UnsupportedConstraint(BaseModel):
    """Something the user asked for that no filter can express."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_extra=_schema_without_description
    )

    constraint: str
    reason: str


class ParseOutput(BaseModel):
    """The response schema: what the model is asked to return.

    Mirrors :class:`Filters` field for field, except that the lookup fields
    carry names rather than ids and there is no ``near`` (see the module
    docstring). Bounds live on ``Filters`` alone and are applied when the
    output is converted into one, so the two models cannot disagree about
    them. As in ``Filters``, an absent filter is unconstrained.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_extra=_schema_without_description
    )

    neighbourhoods: tuple[str, ...] = ()
    room_types: tuple[str, ...] = ()
    property_types: tuple[str, ...] = ()

    min_price: float | None = None
    max_price: float | None = None
    min_accommodates: int | None = None
    min_bedrooms: int | None = None
    min_beds: int | None = None
    min_bathrooms: float | None = None
    max_minimum_nights: int | None = None
    min_rating: float | None = None
    min_location_score: float | None = None
    min_reviews: int | None = None
    instant_bookable: bool | None = None
    host_is_superhost: bool | None = None

    amenities: tuple[str, ...] = ()

    # Required, unlike the filters: an empty semantic query leaves nothing to
    # rank by, and an omitted list of unsupported asks is indistinguishable from
    # the model having forgotten to say.
    semantic_query: str
    unsupported_constraints: tuple[UnsupportedConstraint, ...]

    @field_validator("semantic_query")
    @classmethod
    def _semantic_query_is_not_blank(cls, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            raise ValueError("must not be empty; describe the place wanted in a few words")
        return stripped


# Copied to Filters as they are: same name, same meaning, and Filters applies
# their bounds.
PASSTHROUGH_FIELDS: Final[frozenset[str]] = (
    frozenset(ParseOutput.model_fields) & frozenset(Filters.model_fields)
) - {"amenities"}


def response_schema(grounding: Grounding) -> dict[str, Any]:
    """The JSON schema the response must follow, with every name list enumerated.

    Generated from :class:`ParseOutput`, so the schema the model sees and the
    validation applied to its answer cannot drift. Each name field's items are
    restricted to that vocabulary, sorted so the schema is byte-stable: changing
    the response format invalidates the prompt cache just as editing the system
    prompt does. A vocabulary that is empty drops its field altogether, because
    an empty enum is a schema no value can satisfy.
    """
    schema = ParseOutput.model_json_schema()
    for definition in (schema, *schema.get("$defs", {}).values()):
        # A title is a Python name: renaming a class must no more move the
        # cached prefix than editing its docstring does.
        definition.pop("title", None)
        for field_schema in definition["properties"].values():
            # A `default` would be folded into the field's description by the
            # SDK, as Python syntax. The prompt already says that an absent
            # filter is unconstrained.
            field_schema.pop("title", None)
            field_schema.pop("default", None)

    properties = schema["properties"]
    for field, names in grounding.vocabularies().items():
        if names:
            properties[field]["items"] = {"type": "string", "enum": names}
        else:
            del properties[field]
    return schema


def build_request(query: str, grounding: Grounding, *, model: str) -> LlmRequest:
    """The first parse call for ``query``.

    Everything but the one user message is a function of the grounding alone,
    so two queries against the same database share their cacheable prefix byte
    for byte.
    """
    return LlmRequest(
        model=model,
        system=system_prompt(grounding.render()),
        messages=(Message.user(query),),
        response_schema=response_schema(grounding),
    )


def with_feedback(request: LlmRequest, *, raw_output: str, problem: str) -> LlmRequest:
    """``request`` with the rejected answer and the reason it was rejected appended.

    System prompt and schema are untouched, so the retry reuses the first call's
    cached prefix. A blank answer is left out rather than replayed: the API
    refuses an empty text block, and there is nothing in it to correct.
    """
    turns = list(request.messages)
    if raw_output.strip():
        turns.append(Message.assistant(TextBlock(raw_output)))
    turns.append(Message.user(retry_feedback(problem)))
    return replace(request, messages=tuple(turns))


def describe_validation_error(exc: ValidationError) -> str:
    """A validation failure as the model should read it: each field and its problem.

    No documentation links and no echoed input -- the model has its own answer
    in the conversation already.
    """
    problems = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "response"
        problems.append(f"{location}: {error['msg']}")
    return "; ".join(problems)


# --- The result ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UngroundedValue:
    """A name the model emitted that the vocabulary does not contain.

    Kept apart from ``unsupported_constraints`` -- where each one is also
    listed, so the answer states it -- because inventing a neighbourhood is a
    worse failure than a constraint Scout honestly cannot express, and the
    evaluation counts the two separately.
    """

    field: str
    value: str


@dataclass(frozen=True)
class ParseResult:
    """What Parse hands the rest of the graph."""

    filters: Filters
    semantic_query: str
    #: The model's own unsupported asks, then any name grounding rejected.
    unsupported_constraints: tuple[UnsupportedConstraint, ...]
    ungrounded: tuple[UngroundedValue, ...]
    #: Every call in order, the accepted one last, each with its raw output.
    attempts: tuple[ParseAttempt, ...] = ()


def _partition(names: Iterable[str], known: Container[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    rejected: list[str] = []
    for name in names:
        (kept if name in known else rejected).append(name)
    return kept, rejected


def ground(body: Mapping[str, Any], grounding: Grounding) -> ParseResult:
    """Validate a response body and convert it into ``Filters``.

    Names the vocabulary lacks are dropped from the filters and reported, never
    passed on. Pure: no I/O, so every path is unit-testable. The result carries
    no attempts; :func:`parse_query` adds them.

    Raises:
        ValidationError: if the body does not match :class:`ParseOutput`, or
            its values break a bound :class:`Filters` enforces.
    """
    output = ParseOutput.model_validate(body)

    # Keyed on (field, name) and insertion-ordered, so a name the model repeats
    # is one hallucination rather than two in the evaluation's count.
    rejected: dict[UngroundedValue, str] = {}
    lookups: dict[str, tuple[int, ...]] = {}
    for field, lookup in _LOOKUPS.items():
        vocabulary = grounding.lookup(field)
        names: tuple[str, ...] = getattr(output, field)
        kept, unknown = _partition(names, vocabulary)
        lookups[lookup.filters_field] = tuple(vocabulary[name] for name in kept)
        rejected.update(
            (UngroundedValue(field, name), lookup.ungrounded_reason) for name in unknown
        )
    amenities, unknown_amenities = _partition(output.amenities, grounding.amenities)
    rejected.update(
        (UngroundedValue("amenities", name), _AMENITY_UNGROUNDED_REASON)
        for name in unknown_amenities
    )
    ungrounded = tuple(rejected)

    filters = Filters.model_validate(
        {
            **output.model_dump(include=set(PASSTHROUGH_FIELDS)),
            **lookups,
            "amenities": tuple(amenities),
        }
    )

    moved = tuple(
        UnsupportedConstraint(constraint=item.value, reason=reason)
        for item, reason in rejected.items()
        if not _already_reported(item.value, output.unsupported_constraints)
    )
    if ungrounded:
        logger.warning(
            "parse named %d value(s) outside the vocabulary, moved to unsupported: %s",
            len(ungrounded),
            ", ".join(f"{item.field}={item.value!r}" for item in ungrounded),
        )
    return ParseResult(
        filters=filters,
        semantic_query=output.semantic_query,
        unsupported_constraints=(*output.unsupported_constraints, *moved),
        ungrounded=ungrounded,
    )


def _already_reported(name: str, reported: Iterable[UnsupportedConstraint]) -> bool:
    """Whether the model already listed this name among its own unsupported asks.

    A containment test rather than equality, because the model records the
    user's own words: "in Atlantis" covers the rejected name "Atlantis", and
    listing both would have the answer report one ask twice.
    """
    needle = name.casefold()
    return any(needle in item.constraint.casefold() for item in reported)


# --- The node --------------------------------------------------------------


def parse_query(
    query: str,
    *,
    backend: LlmBackend,
    grounding: Grounding,
    settings: Settings,
    usage: UsageAccumulator,
) -> ParseResult:
    """Parse one natural-language request into grounded, validated filters.

    Makes at most :data:`MAX_ATTEMPTS` calls, each priced into ``usage`` --
    including the rejected ones, which are billed like any other. The model
    comes from ``settings.model_parse``.

    Raises:
        ValueError: if ``query`` is blank. There is nothing to parse, and
            sending it would pay for a guess.
        ParseError: if both attempts produced a response that failed
            validation.
        LlmApiError: if the call itself failed. The client has already applied
            its own retry policy, so this one is not retried again here.
        LlmResponseError: if the model refused, returned nothing, or ran out of
            tokens mid-answer. Each would recur on an identical retry, so none
            is retried.
    """
    if not query.strip():
        raise ValueError("the query is empty; there is nothing to parse")

    first_request = build_request(query, grounding, model=settings.model_parse)
    request = first_request
    attempts: list[ParseAttempt] = []
    while len(attempts) < MAX_ATTEMPTS:
        if attempts:
            rejected = attempts[-1]
            problem = rejected.error or "the response could not be used"
            logger.warning(
                "parse attempt %d of %d rejected, retrying: %s",
                len(attempts),
                MAX_ATTEMPTS,
                problem,
            )
            request = with_feedback(first_request, raw_output=rejected.raw_output, problem=problem)
        try:
            response = backend.complete(request)
        except LlmSchemaError as exc:
            usage.record_failure(USAGE_LABEL, exc)
            attempts.append(ParseAttempt(raw_output=exc.raw_output, error=str(exc)))
            continue
        except LlmResponseError as exc:
            usage.record_failure(USAGE_LABEL, exc)
            raise
        usage.record(USAGE_LABEL, response)

        if response.json_output is None:
            # Both backends decode the body whenever a schema was sent, so this
            # is a backend bug rather than the model's mistake, and feeding it
            # back as one would spend the retry on nothing. Usage is already
            # recorded, so the error carries none to be counted twice.
            raise LlmResponseError(
                "the backend returned no structured output for a schema-constrained request"
            )
        try:
            grounded = ground(response.json_output, grounding)
        except ValidationError as exc:
            attempts.append(
                ParseAttempt(raw_output=response.text, error=describe_validation_error(exc))
            )
            continue

        attempts.append(ParseAttempt(raw_output=response.text))
        return replace(grounded, attempts=tuple(attempts))

    raise ParseError(
        f"could not parse {query!r} into valid filters in {MAX_ATTEMPTS} attempts; "
        f"the last response was rejected because {attempts[-1].error}",
        query=query,
        attempts=tuple(attempts),
    )
