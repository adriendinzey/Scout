"""Turning a validated filter set into parameterized filtered-vector queries.

The index evaluates a predicate *during* graph traversal, which is the whole
reason Scout exists -- but only for the predicate shapes it can push down: `=`,
`<`, `<=`, `>`, `>=` and `BETWEEN`, joined by `AND`. `OR` and `NOT` are applied
after the scan instead, and a post-scan filter throws away candidates the ranked
scan already paid for.

Everything below follows from that one constraint:

* **Disjunctions fan out.** "Hackney or Islington" becomes one query per
  neighbourhood, each a pushed-down equality, merged by distance downstream.
* **Fan-out is capped.** The cross-product grows multiplicatively -- three
  neighbourhoods by two room types is already six queries -- so past the cap a
  single query expresses the disjunction as `= ANY(...)` and the plan records
  that it did. That weakens recall, and a silent fallback would quietly corrupt
  every number measured on top of it.
* **Some conditions can only ever be post-filters.** An amenity with no indexed
  boolean column is an array containment test. It is recorded the same way.

Nothing here performs I/O: a filter set goes in, composed SQL and its parameters
come out. The schema facts it needs -- which amenities have boolean columns --
are arguments, not lookups.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Final

from psycopg import sql

from scout.retrieval.filters import Filters, NearFilter

_TABLE: Final = sql.Identifier("listings")
_EMBEDDING: Final = sql.Identifier("embedding")

# A degree of latitude is ~111 km everywhere. A degree of longitude is that
# scaled by cos(latitude): at London's ~51.5 deg N it is only ~69 km, so using
# the same delta for both would make the box half again too wide.
KM_PER_DEGREE_LATITUDE: Final = 111.0

# Below this cosine the longitude delta diverges towards a division by ~0, so
# the box stops being expressible as a range and widens to every longitude
# instead. London is nowhere near a pole, but a guard costs less than the
# mystery a division by ~0 would produce.
_MIN_COS_LATITUDE: Final = 1e-6

# Columns the source genuinely omits. A comparison against NULL is never true,
# so filtering on one of these silently drops every listing that lacks it --
# `min_rating` excludes unrated listings rather than including them. Correct
# SQL, surprising to a reader, so a built plan reports which ones it applied.
_NULLABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "price_usd",
        "bedrooms",
        "beds",
        "bathrooms",
        "rating",
        "location_score",
        "host_is_superhost",
    }
)

# Enough to rank a result and show it without a second round trip, and no
# large text: `description`, `doc_text` and the amenity array are fetched per
# listing when one is actually cited.
DEFAULT_SELECT_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "source_listing_id",
    "name",
    "neighbourhood_id",
    "room_type_id",
    "property_type_id",
    "latitude",
    "longitude",
    "price_usd",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "rating",
    "location_score",
    "number_of_reviews",
    "instant_bookable",
    "host_is_superhost",
)


@dataclass(frozen=True)
class _ScalarSpec:
    """One filter field, as the column and operator it becomes."""

    column: str
    operator: sql.SQL


# Field -> column -> operator, as data. Iteration order fixes the order
# conditions appear in, so the same filter set always composes the same SQL.
_SCALAR_FILTERS: Final[Mapping[str, _ScalarSpec]] = {
    "min_price": _ScalarSpec("price_usd", sql.SQL(">=")),
    "max_price": _ScalarSpec("price_usd", sql.SQL("<=")),
    "min_accommodates": _ScalarSpec("accommodates", sql.SQL(">=")),
    "min_bedrooms": _ScalarSpec("bedrooms", sql.SQL(">=")),
    "min_beds": _ScalarSpec("beds", sql.SQL(">=")),
    "min_bathrooms": _ScalarSpec("bathrooms", sql.SQL(">=")),
    "max_minimum_nights": _ScalarSpec("minimum_nights", sql.SQL("<=")),
    "min_rating": _ScalarSpec("rating", sql.SQL(">=")),
    "min_location_score": _ScalarSpec("location_score", sql.SQL(">=")),
    "min_reviews": _ScalarSpec("number_of_reviews", sql.SQL(">=")),
    "instant_bookable": _ScalarSpec("instant_bookable", sql.SQL("=")),
    "host_is_superhost": _ScalarSpec("host_is_superhost", sql.SQL("=")),
}

# Field -> column, for the fields whose several values mean "any of".
_DISJUNCTIVE_FILTERS: Final[Mapping[str, str]] = {
    "neighbourhood_ids": "neighbourhood_id",
    "room_type_ids": "room_type_id",
    "property_type_ids": "property_type_id",
}


@dataclass(frozen=True)
class Condition:
    """One `WHERE` term, with the parameters its placeholders consume."""

    clause: sql.Composed
    params: tuple[object, ...]
    column: str
    # False when the index cannot evaluate this term during traversal, so it is
    # applied to rows the ranked scan already returned.
    pushed_down: bool

    @property
    def null_excluding(self) -> bool:
        return self.column in _NULLABLE_COLUMNS


@dataclass(frozen=True)
class BoundingBox:
    """Latitude and longitude ranges enclosing a circle.

    A superset of the circle: the corners of the box lie outside it. That is the
    right direction to be wrong in -- the box never drops a listing that is
    genuinely within the radius.

    `longitude` is None when every longitude qualifies: the range wraps past the
    antimeridian, spans the globe, or the circle encloses a pole. Splitting any
    of those into two ranges would need `OR`, which cannot be pushed down;
    widening keeps the superset property and keeps the predicate in the index.
    """

    latitude: tuple[float, float]
    longitude: tuple[float, float] | None


def bounding_box(near: NearFilter) -> BoundingBox:
    """Convert a point and radius into ranges the index can filter on.

    Deliberately no exact-distance post-filter on top. Trimming the corners
    would mean discarding candidates the ranked scan has already paid for, to
    buy precision the data does not have: coordinates are offset by up to ~150 m
    upstream, so sub-kilometre accuracy here would be fiction.
    """
    delta_latitude = near.radius_km / KM_PER_DEGREE_LATITUDE
    latitude = (
        max(near.latitude - delta_latitude, -90.0),
        min(near.latitude + delta_latitude, 90.0),
    )

    # A circle reaching a pole wraps around it: points on the far meridian are
    # inside the radius but outside any bounded longitude range, so bounding one
    # would break the superset property rather than merely loosen it.
    if latitude[0] <= -90.0 or latitude[1] >= 90.0:
        return BoundingBox(latitude=latitude, longitude=None)

    # Scaled at the edge of the latitude band, not at its centre. The circle
    # reaches latitudes further from the equator than the centre, and a degree
    # of longitude is shorter there -- scaling by the centre's cosine leaves the
    # box marginally too narrow, which at high latitudes is enough to drop
    # points that are genuinely inside the radius.
    outermost_latitude = max(abs(latitude[0]), abs(latitude[1]))
    shrink = math.cos(math.radians(outermost_latitude))
    # Clamping the divisor instead would cap the delta, which narrows the box
    # rather than widening it -- the one direction that breaks the superset
    # property. Give up on a bounded range instead.
    if shrink <= _MIN_COS_LATITUDE:
        return BoundingBox(latitude=latitude, longitude=None)
    delta_longitude = near.radius_km / (KM_PER_DEGREE_LATITUDE * shrink)

    west = near.longitude - delta_longitude
    east = near.longitude + delta_longitude
    if west < -180.0 or east > 180.0:
        return BoundingBox(latitude=latitude, longitude=None)
    return BoundingBox(latitude=latitude, longitude=(west, east))


def _compare(column: str, operator: sql.SQL, value: object) -> Condition:
    return Condition(
        clause=sql.SQL("{column} {operator} {value}").format(
            column=sql.Identifier(column), operator=operator, value=sql.Placeholder()
        ),
        params=(value,),
        column=column,
        pushed_down=True,
    )


def _between(column: str, low: float, high: float) -> Condition:
    return Condition(
        clause=sql.SQL("{column} BETWEEN {low} AND {high}").format(
            column=sql.Identifier(column), low=sql.Placeholder(), high=sql.Placeholder()
        ),
        params=(low, high),
        column=column,
        pushed_down=True,
    )


def _any_of(column: str, values: Sequence[int]) -> Condition:
    """`= ANY(...)`, the single-query form of a disjunction.

    Correct, but evaluated after the scan rather than inside it -- the reason
    fan-out exists and the reason exceeding the cap is worth recording.
    """
    return Condition(
        clause=sql.SQL("{column} = ANY({values})").format(
            column=sql.Identifier(column), values=sql.Placeholder()
        ),
        params=(list(values),),
        column=column,
        pushed_down=False,
    )


def _scalar_conditions(filters: Filters) -> list[Condition]:
    conditions = []
    for field, spec in _SCALAR_FILTERS.items():
        value = getattr(filters, field)
        if value is not None:
            conditions.append(_compare(spec.column, spec.operator, value))
    return conditions


def _near_conditions(filters: Filters) -> list[Condition]:
    if filters.near is None:
        return []
    box = bounding_box(filters.near)
    conditions = [_between("latitude", *box.latitude)]
    if box.longitude is not None:
        conditions.append(_between("longitude", *box.longitude))
    return conditions


def _amenity_conditions(filters: Filters, indexed_amenities: Mapping[str, str]) -> list[Condition]:
    """Conjunctive: every requested amenity must be present.

    An amenity with a boolean column is a pushed-down equality. The rest share
    one array-containment term, which the index cannot evaluate -- so it is a
    post-filter, and says so.
    """
    conditions = []
    unindexed = []
    for amenity in filters.amenities:
        column = indexed_amenities.get(amenity)
        if column is None:
            unindexed.append(amenity)
        else:
            conditions.append(_compare(column, sql.SQL("="), True))
    if unindexed:
        conditions.append(
            Condition(
                clause=sql.SQL("{column} @> {values}").format(
                    column=sql.Identifier("amenities"), values=sql.Placeholder()
                ),
                params=(unindexed,),
                column="amenities",
                pushed_down=False,
            )
        )
    return conditions


def _fanout_fields(filters: Filters) -> list[tuple[str, str, tuple[int, ...]]]:
    """The disjunctive fields with more than one value, in declaration order.

    A single value is not a disjunction: it is an equality that pushes down, and
    it costs no branch.
    """
    fields = []
    for field, column in _DISJUNCTIVE_FILTERS.items():
        values: tuple[int, ...] = getattr(filters, field)
        if len(values) > 1:
            fields.append((field, column, values))
    return fields


def _single_valued_conditions(filters: Filters) -> list[Condition]:
    conditions = []
    for field, column in _DISJUNCTIVE_FILTERS.items():
        values: tuple[int, ...] = getattr(filters, field)
        if len(values) == 1:
            conditions.append(_compare(column, sql.SQL("="), values[0]))
    return conditions


@dataclass(frozen=True)
class WhereClause:
    """A composed `WHERE` body, or None when nothing is constrained.

    `post_filter_columns` names the conditions the index cannot evaluate *in
    this statement*. On a ranked search that is a recall cost, and `QueryPlan`
    is the authority on it. On the un-ranked form it is not: a count scans rows
    and returns an exact answer whether or not a predicate could have been
    pushed down, so the same filter set is legitimately reported differently by
    the two -- see `where_clause`.
    """

    clause: sql.Composed | None
    params: tuple[object, ...]
    post_filter_columns: tuple[str, ...]
    null_excluding_columns: tuple[str, ...]

    @property
    def is_post_filtered(self) -> bool:
        return bool(self.post_filter_columns)


def _assemble(conditions: Sequence[Condition]) -> WhereClause:
    if not conditions:
        return WhereClause(
            clause=None, params=(), post_filter_columns=(), null_excluding_columns=()
        )
    return WhereClause(
        clause=sql.SQL(" AND ").join(condition.clause for condition in conditions),
        params=tuple(param for condition in conditions for param in condition.params),
        post_filter_columns=_distinct(c.column for c in conditions if not c.pushed_down),
        null_excluding_columns=_distinct(c.column for c in conditions if c.null_excluding),
    )


def _distinct(columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(columns)))


def where_clause(filters: Filters, *, indexed_amenities: Mapping[str, str]) -> WhereClause:
    """Build a single `WHERE` body covering every filter, disjunctions included.

    This is the un-ranked form, for queries with no vector to order by -- a count
    or a field's distribution. Those scan rows rather than index nodes, so
    expressing a disjunction as `= ANY(...)` costs them nothing and one query is
    simply cheaper than several. Ranked search wants `filters_to_sql` instead.

    Because of that the two forms deliberately disagree about a disjunction:
    here it is one un-pushed condition, there it is several pushed-down
    branches. Neither is degraded -- an exact count has no recall to lose.
    """
    conditions = [
        *_single_valued_conditions(filters),
        *(_any_of(column, values) for _, column, values in _fanout_fields(filters)),
        *_scalar_conditions(filters),
        *_near_conditions(filters),
        *_amenity_conditions(filters, indexed_amenities),
    ]
    return _assemble(conditions)


@dataclass(frozen=True)
class RetrievalQuery:
    """One composed statement and its parameters: a single fan-out branch."""

    statement: sql.Composed
    params: tuple[object, ...]
    # The disjunctive value this branch pins, e.g. {"neighbourhood_ids": 7}.
    # Empty when the filter set needed no fan-out.
    branch: Mapping[str, int]
    post_filter_columns: tuple[str, ...]


@dataclass(frozen=True)
class QueryPlan:
    """Every query one filter set becomes, and what it cost to express it.

    The bookkeeping is not diagnostics. `fanout_capped` and `post_filter_columns`
    record where the plan had to fall back to filtering after the scan, which
    weakens recall -- an evaluation that did not know which runs were degraded
    would be comparing two different things and reporting one number.
    """

    queries: tuple[RetrievalQuery, ...]
    # Branches the cross-product asked for, which exceeds len(queries) exactly
    # when the cap forced the fallback.
    requested_branches: int
    fanout_cap: int
    fanout_capped: bool
    post_filter_columns: tuple[str, ...]
    null_excluding_columns: tuple[str, ...]

    @property
    def is_post_filtered(self) -> bool:
        return bool(self.post_filter_columns)


def filters_to_sql(
    filters: Filters,
    *,
    indexed_amenities: Mapping[str, str],
    query_embedding: Sequence[float],
    limit: int,
    fanout_cap: int,
    select_columns: Sequence[str] = DEFAULT_SELECT_COLUMNS,
) -> QueryPlan:
    """Compose the ranked filtered-vector search for one filter set.

    Returns one query per disjunction branch, each ordered by cosine distance to
    `query_embedding` and returning at most `limit` rows -- so the caller gets up
    to `limit x len(queries)` rows in total and merges them by distance. Past
    `fanout_cap` branches it returns a single query that applies the disjunction
    after the scan, with `fanout_capped` set.

    `indexed_amenities` maps an amenity name to its boolean column; anything
    absent from it is matched against the amenity array as a post-filter.

    Raises:
        ValueError: if `limit` or `fanout_cap` is below 1, or `query_embedding`
            or `select_columns` is empty -- each of which would produce a query
            that is useless or that the server cannot parse.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if fanout_cap < 1:
        raise ValueError(f"fanout_cap must be at least 1, got {fanout_cap}")
    if not query_embedding:
        raise ValueError("query_embedding is empty; there is nothing to rank by")
    if not select_columns:
        raise ValueError("select_columns is empty; the statement would have no select list")

    fanout = _fanout_fields(filters)
    requested_branches = math.prod(len(values) for _, _, values in fanout)
    capped = requested_branches > fanout_cap

    shared = [
        *_single_valued_conditions(filters),
        *_scalar_conditions(filters),
        *_near_conditions(filters),
        *_amenity_conditions(filters, indexed_amenities),
    ]

    branch_conditions: list[tuple[list[Condition], dict[str, int]]]
    if capped:
        branch_conditions = [([_any_of(column, values) for _, column, values in fanout], {})]
    else:
        branch_conditions = [
            (
                [
                    _compare(column, sql.SQL("="), value)
                    for (_, column, _), value in zip(fanout, combination, strict=True)
                ],
                {field: value for (field, _, _), value in zip(fanout, combination, strict=True)},
            )
            for combination in product(*(values for _, _, values in fanout))
        ]

    vector = list(query_embedding)
    queries = []
    post_filter_columns: set[str] = set()
    null_excluding_columns: set[str] = set()
    for conditions, branch in branch_conditions:
        where = _assemble([*conditions, *shared])
        post_filter_columns.update(where.post_filter_columns)
        null_excluding_columns.update(where.null_excluding_columns)
        queries.append(
            RetrievalQuery(
                statement=_statement(where, select_columns),
                params=(vector, *where.params, vector, limit),
                branch=branch,
                post_filter_columns=where.post_filter_columns,
            )
        )

    return QueryPlan(
        queries=tuple(queries),
        requested_branches=requested_branches,
        fanout_cap=fanout_cap,
        fanout_capped=capped,
        post_filter_columns=_distinct(post_filter_columns),
        null_excluding_columns=_distinct(null_excluding_columns),
    )


def _distance_expression() -> sql.Composed:
    """Cosine distance to a bound query vector.

    `<=>` matches the index's operator class; a mismatch returns wrong
    neighbours without erroring. The cast is explicit because a Python sequence
    of floats adapts to `float8[]`, which the vector type does not accept
    directly.
    """
    return sql.SQL("{embedding} <=> {vector}::real[]::brindle_vector").format(
        embedding=_EMBEDDING, vector=sql.Placeholder()
    )


def _statement(where: WhereClause, select_columns: Sequence[str]) -> sql.Composed:
    """Compose the full ranked statement.

    The distance expression is bound twice -- once projected, once ordered by --
    rather than ordering by the output alias. The index serves an `ORDER BY` on
    the expression itself; an alias is a different thing to plan around, and the
    cost of getting that wrong is a sort on top of the index instead of an
    `Order By` inside it.
    """
    distance = _distance_expression()
    parts = [
        sql.SQL("SELECT {columns}, {distance} AS distance").format(
            columns=sql.SQL(", ").join(sql.Identifier(column) for column in select_columns),
            distance=distance,
        ),
        sql.SQL("FROM {table}").format(table=_TABLE),
    ]
    if where.clause is not None:
        parts.append(sql.SQL("WHERE {clause}").format(clause=where.clause))
    parts.extend(
        [
            sql.SQL("ORDER BY {distance}").format(distance=distance),
            sql.SQL("LIMIT {limit}").format(limit=sql.Placeholder()),
        ]
    )
    return sql.SQL(" ").join(parts)
