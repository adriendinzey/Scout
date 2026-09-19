"""The filter-to-SQL mapping, over-tested on purpose.

A wrong operator or an off-by-one bound here does not fail -- it returns
plausible listings that are quietly the wrong ones, and every metric measured
downstream inherits the error without a symptom. So each field is asserted
individually, the mapping tables are asserted to be fully covered, and the
structural properties (everything parameterized, one query per branch) are
asserted rather than assumed.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from scout.retrieval.filters import MAX_RADIUS_KM, Filters, NearFilter
from scout.retrieval.query import (
    # The mapping tables are private, but the tests below assert they are fully
    # covered -- a new filter field with no test should fail here, not ship.
    _DISJUNCTIVE_FILTERS,
    _SCALAR_FILTERS,
    DEFAULT_SELECT_COLUMNS,
    QueryPlan,
    RetrievalQuery,
    filters_to_sql,
    where_clause,
)

INDEXED_AMENITIES = {
    "Wifi": "has_wifi",
    "Kitchen": "has_kitchen",
    "Free parking": "has_free_parking",
}
EMBEDDING = [0.1, 0.2, 0.3]


def plan_for(filters: Filters, *, limit: int = 10, fanout_cap: int = 4, **kwargs: Any) -> QueryPlan:
    return filters_to_sql(
        filters,
        indexed_amenities=kwargs.pop("indexed_amenities", INDEXED_AMENITIES),
        query_embedding=kwargs.pop("query_embedding", EMBEDDING),
        limit=limit,
        fanout_cap=fanout_cap,
        **kwargs,
    )


def rendered(query: RetrievalQuery) -> str:
    return query.statement.as_string()


def only(plan: QueryPlan) -> RetrievalQuery:
    assert len(plan.queries) == 1, f"expected one query, got {len(plan.queries)}"
    return plan.queries[0]


def where_of(query: RetrievalQuery) -> str:
    text = rendered(query)
    assert " WHERE " in text, f"query has no WHERE clause:\n{text}"
    return text.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0]


# --------------------------------------------------------------- the model --


def test_an_unknown_field_is_rejected() -> None:
    """A hallucinated column must fail at the boundary, not narrow nothing."""
    with pytest.raises(ValidationError):
        Filters(has_hot_tub=True)  # type: ignore[call-arg]


def test_set_valued_fields_are_sorted_and_deduplicated() -> None:
    filters = Filters(neighbourhood_ids=(9, 3, 9, 1), amenities=("Wifi", "Kitchen", "Wifi"))
    assert filters.neighbourhood_ids == (1, 3, 9)
    assert filters.amenities == ("Kitchen", "Wifi")


def test_the_same_request_in_a_different_order_is_the_same_filter_set() -> None:
    """Equality and hashing are what let the loop refuse a repeated call."""
    one = Filters(neighbourhood_ids=(4, 2), amenities=("Wifi", "Kitchen"))
    other = Filters(neighbourhood_ids=(2, 4), amenities=("Kitchen", "Wifi"))
    assert one == other
    assert hash(one) == hash(other)


def test_a_blank_amenity_is_an_error_not_an_empty_filter() -> None:
    with pytest.raises(ValidationError):
        Filters(amenities=("Wifi", "   "))


def test_an_unsatisfiable_price_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exceeds max_price"):
        Filters(min_price=300, max_price=100)


def test_an_equal_price_window_is_allowed() -> None:
    assert Filters(min_price=100, max_price=100).min_price == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_rating": 5.1},
        {"min_location_score": -0.1},
        {"max_price": -1},
        {"min_accommodates": 0},
        {"neighbourhood_ids": (0,)},
        {"max_minimum_nights": 0},
    ],
)
def test_out_of_range_values_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Filters(**kwargs)


@pytest.mark.parametrize("radius_km", [0.0, -1.0, MAX_RADIUS_KM + 0.1])
def test_an_implausible_radius_is_rejected(radius_km: float) -> None:
    with pytest.raises(ValidationError):
        NearFilter(latitude=51.5, longitude=-0.12, radius_km=radius_km)


def test_is_empty_distinguishes_an_unconstrained_search() -> None:
    assert Filters().is_empty()
    assert not Filters(min_reviews=0).is_empty()


# ------------------------------------------------ None means unconstrained --


def test_no_filters_means_no_where_clause() -> None:
    """An empty filter set searches the whole corpus, and says so in the SQL."""
    plan = plan_for(Filters())
    query = only(plan)
    assert " WHERE " not in rendered(query)
    assert query.params == (EMBEDDING, EMBEDDING, 10)
    assert plan.post_filter_columns == ()
    assert plan.null_excluding_columns == ()
    assert plan.requested_branches == 1
    assert not plan.fanout_capped


@pytest.mark.parametrize("field", sorted(_SCALAR_FILTERS))
def test_an_unset_field_emits_no_condition(field: str) -> None:
    column = _SCALAR_FILTERS[field].column
    assert f'"{column}"' not in where_of_or_blank(plan_for(Filters()))


def where_of_or_blank(plan: QueryPlan) -> str:
    text = rendered(only(plan))
    return text.split(" WHERE ", 1)[1] if " WHERE " in text else ""


@pytest.mark.parametrize(
    ("field", "value"),
    [("min_bedrooms", 0), ("min_beds", 0), ("min_reviews", 0), ("min_bathrooms", 0.0)],
)
def test_a_zero_bound_is_a_real_filter_not_an_absent_one(field: str, value: float) -> None:
    """Zero is falsy; a truthiness check here would silently drop the filter."""
    column = _SCALAR_FILTERS[field].column
    query = only(plan_for(Filters(**{field: value})))
    assert f'"{column}" >= %s' in where_of(query)
    assert value in query.params


@pytest.mark.parametrize("field", ["instant_bookable", "host_is_superhost"])
def test_false_is_a_real_filter_not_an_absent_one(field: str) -> None:
    query = only(plan_for(Filters(**{field: False})))
    assert f'"{field}" = %s' in where_of(query)
    assert False in query.params


# -------------------------------------------------------- scalar mappings --

SCALAR_CASES = [
    ("min_price", 123.5, '"price_usd" >= %s'),
    ("max_price", 123.5, '"price_usd" <= %s'),
    ("min_accommodates", 4, '"accommodates" >= %s'),
    ("min_bedrooms", 2, '"bedrooms" >= %s'),
    ("min_beds", 3, '"beds" >= %s'),
    ("min_bathrooms", 1.5, '"bathrooms" >= %s'),
    ("max_minimum_nights", 3, '"minimum_nights" <= %s'),
    ("min_rating", 4.5, '"rating" >= %s'),
    ("min_location_score", 4.25, '"location_score" >= %s'),
    ("min_reviews", 10, '"number_of_reviews" >= %s'),
    ("instant_bookable", True, '"instant_bookable" = %s'),
    ("host_is_superhost", True, '"host_is_superhost" = %s'),
]


@pytest.mark.parametrize(("field", "value", "expected"), SCALAR_CASES)
def test_each_scalar_field_maps_to_its_column_and_operator(
    field: str, value: Any, expected: str
) -> None:
    query = only(plan_for(Filters(**{field: value})))
    assert where_of(query) == expected
    assert query.params == (EMBEDDING, value, EMBEDDING, 10)


def test_every_scalar_field_is_covered_by_a_case() -> None:
    """Guards the table above: a new filter field without a test fails here."""
    assert {field for field, _, _ in SCALAR_CASES} == set(_SCALAR_FILTERS)


def test_every_declared_filter_field_is_mapped_to_sql() -> None:
    """The other direction, and the one that fails silently.

    A field added to `Filters` but not to a mapping table emits no condition,
    raises nothing, and narrows nothing -- the search quietly ignores a
    constraint the user asked for. Nothing else in the suite would catch it,
    because every other test names the fields it exercises.
    """
    handled = set(_SCALAR_FILTERS) | set(_DISJUNCTIVE_FILTERS) | {"amenities", "near"}
    assert set(Filters.model_fields) == handled


def test_scalar_conditions_combine_with_and_in_a_stable_order() -> None:
    filters = Filters(max_price=200, min_accommodates=2, min_rating=4.5)
    first = where_of(only(plan_for(filters)))
    assert first == '"price_usd" <= %s AND "accommodates" >= %s AND "rating" >= %s'
    assert where_of(only(plan_for(filters))) == first


def test_both_price_bounds_become_two_conditions_on_one_column() -> None:
    query = only(plan_for(Filters(min_price=50, max_price=200)))
    assert where_of(query) == '"price_usd" >= %s AND "price_usd" <= %s'
    assert query.params == (EMBEDDING, 50.0, 200.0, EMBEDDING, 10)


# ---------------------------------------------------------------- the box --


def test_near_becomes_two_between_conditions() -> None:
    query = only(plan_for(Filters(near=NearFilter(latitude=51.5, longitude=-0.12, radius_km=2.0))))
    assert where_of(query) == '"latitude" BETWEEN %s AND %s AND "longitude" BETWEEN %s AND %s'
    _, south, north, west, east, _, _ = query.params
    assert south < 51.5 < north
    assert west < -0.12 < east
    # Longitude is compressed at this latitude, so its span is the wider one.
    assert (east - west) > (north - south)  # type: ignore[operator]


def test_a_box_spanning_every_longitude_emits_only_the_latitude_condition() -> None:
    query = only(plan_for(Filters(near=NearFilter(latitude=90.0, longitude=0.0, radius_km=1.0))))
    assert where_of(query) == '"latitude" BETWEEN %s AND %s'


def test_latitude_and_longitude_are_never_null_excluding() -> None:
    plan = plan_for(Filters(near=NearFilter(latitude=51.5, longitude=-0.12, radius_km=2.0)))
    assert plan.null_excluding_columns == ()


# -------------------------------------------------------------- amenities --


def test_an_indexed_amenity_uses_its_boolean_column_and_pushes_down() -> None:
    plan = plan_for(Filters(amenities=("Wifi",)))
    assert where_of(only(plan)) == '"has_wifi" = %s'
    assert only(plan).params == (EMBEDDING, True, EMBEDDING, 10)
    assert plan.post_filter_columns == ()
    assert not plan.is_post_filtered


def test_a_non_indexed_amenity_is_an_array_test_flagged_as_a_post_filter() -> None:
    plan = plan_for(Filters(amenities=("Pool table",)))
    assert where_of(only(plan)) == '"amenities" @> %s'
    assert only(plan).params == (EMBEDDING, ["Pool table"], EMBEDDING, 10)
    assert plan.post_filter_columns == ("amenities",)
    assert plan.is_post_filtered
    assert only(plan).post_filter_columns == ("amenities",)


def test_amenities_are_conjunctive() -> None:
    """All of them, not any -- one condition each, joined by AND."""
    plan = plan_for(Filters(amenities=("Wifi", "Kitchen")))
    assert where_of(only(plan)) == '"has_kitchen" = %s AND "has_wifi" = %s'


def test_non_indexed_amenities_share_one_containment_condition() -> None:
    plan = plan_for(Filters(amenities=("Wifi", "Pool table", "Sauna")))
    assert where_of(only(plan)) == '"has_wifi" = %s AND "amenities" @> %s'
    assert only(plan).params == (EMBEDDING, True, ["Pool table", "Sauna"], EMBEDDING, 10)
    assert plan.post_filter_columns == ("amenities",)


def test_the_indexed_amenity_set_is_a_parameter_not_a_constant() -> None:
    """With no boolean columns declared, every amenity is a post-filter."""
    plan = plan_for(Filters(amenities=("Wifi",)), indexed_amenities={})
    assert where_of(only(plan)) == '"amenities" @> %s'
    assert plan.post_filter_columns == ("amenities",)


# ------------------------------------------------------------ disjunctions --


@pytest.mark.parametrize(("field", "column"), sorted(_DISJUNCTIVE_FILTERS.items()))
def test_a_single_value_is_an_equality_not_a_branch(field: str, column: str) -> None:
    plan = plan_for(Filters(**{field: (7,)}))
    assert where_of(only(plan)) == f'"{column}" = %s'
    assert only(plan).params == (EMBEDDING, 7, EMBEDDING, 10)
    assert plan.requested_branches == 1
    assert not plan.fanout_capped
    assert plan.post_filter_columns == ()
    assert only(plan).branch == {}


def test_every_disjunctive_field_is_covered_by_a_case() -> None:
    assert set(_DISJUNCTIVE_FILTERS) == {
        "neighbourhood_ids",
        "room_type_ids",
        "property_type_ids",
    }


def test_two_values_become_one_query_each() -> None:
    plan = plan_for(Filters(neighbourhood_ids=(3, 7)))
    assert len(plan.queries) == 2
    assert plan.requested_branches == 2
    assert not plan.fanout_capped
    assert [dict(query.branch) for query in plan.queries] == [
        {"neighbourhood_ids": 3},
        {"neighbourhood_ids": 7},
    ]
    for query, expected in zip(plan.queries, (3, 7), strict=True):
        assert where_of(query) == '"neighbourhood_id" = %s'
        assert query.params == (EMBEDDING, expected, EMBEDDING, 10)


def test_branches_are_the_cross_product_of_the_disjunctive_fields() -> None:
    plan = plan_for(Filters(neighbourhood_ids=(1, 2), room_type_ids=(5, 6)))
    assert plan.requested_branches == 4
    assert [dict(query.branch) for query in plan.queries] == [
        {"neighbourhood_ids": 1, "room_type_ids": 5},
        {"neighbourhood_ids": 1, "room_type_ids": 6},
        {"neighbourhood_ids": 2, "room_type_ids": 5},
        {"neighbourhood_ids": 2, "room_type_ids": 6},
    ]
    for query in plan.queries:
        assert where_of(query) == '"neighbourhood_id" = %s AND "room_type_id" = %s'


def test_exactly_at_the_cap_still_fans_out() -> None:
    plan = plan_for(Filters(neighbourhood_ids=(1, 2), room_type_ids=(5, 6)), fanout_cap=4)
    assert len(plan.queries) == 4
    assert not plan.fanout_capped
    assert plan.post_filter_columns == ()


def test_over_the_cap_falls_back_to_one_post_filtered_query_and_records_it() -> None:
    """Three neighbourhoods by two room types is six branches -- a realistic ask."""
    plan = plan_for(Filters(neighbourhood_ids=(1, 2, 3), room_type_ids=(5, 6)), fanout_cap=4)
    assert plan.requested_branches == 6
    assert plan.fanout_capped
    assert len(plan.queries) == 1
    assert where_of(only(plan)) == '"neighbourhood_id" = ANY(%s) AND "room_type_id" = ANY(%s)'
    assert only(plan).params == (EMBEDDING, [1, 2, 3], [5, 6], EMBEDDING, 10)
    assert plan.post_filter_columns == ("neighbourhood_id", "room_type_id")
    assert plan.is_post_filtered
    assert only(plan).branch == {}


def test_a_cap_of_one_forces_the_fallback_for_any_disjunction() -> None:
    plan = plan_for(Filters(neighbourhood_ids=(1, 2)), fanout_cap=1)
    assert plan.fanout_capped
    assert where_of(only(plan)) == '"neighbourhood_id" = ANY(%s)'


def test_three_disjunctive_fields_multiply() -> None:
    plan = plan_for(
        Filters(neighbourhood_ids=(1, 2), room_type_ids=(5, 6), property_type_ids=(8, 9)),
        fanout_cap=8,
    )
    assert plan.requested_branches == 8
    assert len(plan.queries) == 8
    assert not plan.fanout_capped


def test_a_single_valued_field_costs_no_branch_alongside_a_disjunction() -> None:
    plan = plan_for(Filters(neighbourhood_ids=(1, 2), room_type_ids=(5,)))
    assert plan.requested_branches == 2
    assert len(plan.queries) == 2
    for query in plan.queries:
        # The branch's own equality leads; shared conditions follow it.
        assert where_of(query) == '"neighbourhood_id" = %s AND "room_type_id" = %s'


def test_shared_conditions_appear_in_every_branch() -> None:
    plan = plan_for(Filters(neighbourhood_ids=(1, 2), max_price=200, amenities=("Wifi",)))
    assert len(plan.queries) == 2
    for query in plan.queries:
        clause = where_of(query)
        assert clause == '"neighbourhood_id" = %s AND "price_usd" <= %s AND "has_wifi" = %s'
        assert query.params[2:4] == (200.0, True)


# ---------------------------------------------------------- NULL semantics --


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"min_rating": 4.5}, ("rating",)),
        ({"max_price": 200}, ("price_usd",)),
        ({"min_bedrooms": 2}, ("bedrooms",)),
        ({"min_beds": 2}, ("beds",)),
        ({"min_bathrooms": 1.0}, ("bathrooms",)),
        ({"min_location_score": 4.0}, ("location_score",)),
        ({"host_is_superhost": True}, ("host_is_superhost",)),
        ({"host_is_superhost": False}, ("host_is_superhost",)),
        ({"min_accommodates": 2}, ()),
        ({"min_reviews": 5}, ()),
        ({"instant_bookable": True}, ()),
        ({"max_minimum_nights": 3}, ()),
        ({"neighbourhood_ids": (1,)}, ()),
    ],
)
def test_null_excluding_filters_are_identifiable(
    kwargs: dict[str, Any], expected: tuple[str, ...]
) -> None:
    """A NULL satisfies no comparison, so these silently drop listings.

    The answer has to be able to say so, which means the plan has to report it.
    """
    assert plan_for(Filters(**kwargs)).null_excluding_columns == expected


def test_null_excluding_columns_are_collected_across_filters() -> None:
    plan = plan_for(Filters(min_rating=4.5, max_price=200, min_accommodates=2))
    assert plan.null_excluding_columns == ("price_usd", "rating")


# ------------------------------------------------------------- the shape ---


def test_the_vector_is_bound_for_both_the_projection_and_the_ordering() -> None:
    """Ordering by the expression, not the output alias, is what the index serves."""
    text = rendered(only(plan_for(Filters())))
    assert text.count('"embedding" <=> %s::real[]::brindle_vector') == 2
    assert " AS distance FROM " in text
    assert text.endswith("LIMIT %s")


def test_the_limit_is_bound_per_branch() -> None:
    plan = plan_for(Filters(neighbourhood_ids=(1, 2)), limit=25)
    for query in plan.queries:
        assert query.params[-1] == 25


def test_the_selected_columns_can_be_overridden() -> None:
    query = only(plan_for(Filters(), select_columns=("id", "name")))
    assert rendered(query).startswith('SELECT "id", "name", "embedding" <=>')


def test_the_default_columns_carry_what_ranking_and_display_need() -> None:
    text = rendered(only(plan_for(Filters())))
    for column in ("id", "name", "price_usd", "rating", "number_of_reviews"):
        assert f'"{column}"' in text
    # Large text is fetched per cited listing instead of on every candidate.
    for column in ("description", "doc_text"):
        assert column not in DEFAULT_SELECT_COLUMNS


# ------------------------------------------------- parameterization safety --


def test_no_user_supplied_value_appears_in_the_query_text() -> None:
    """Values reach the database as parameters; nothing is interpolated."""
    filters = Filters(
        neighbourhood_ids=(4242,),
        max_price=98765.5,
        min_accommodates=77,
        min_rating=4.875,
        amenities=("Wifi", "Trebuchet"),
        near=NearFilter(latitude=51.5051, longitude=-0.1234, radius_km=3.75),
    )
    text = rendered(only(plan_for(filters)))
    for value in ("4242", "98765", "77", "4.875", "Trebuchet", "51.5051", "0.1234", "3.75"):
        assert value not in text, f"{value!r} was interpolated into the SQL"


@pytest.mark.parametrize(
    "filters",
    [
        Filters(),
        Filters(max_price=200),
        Filters(neighbourhood_ids=(1, 2), room_type_ids=(5, 6)),
        Filters(neighbourhood_ids=(1, 2, 3), room_type_ids=(5, 6)),
        Filters(
            amenities=("Wifi", "Sauna"),
            near=NearFilter(latitude=51.5, longitude=-0.1, radius_km=2),
        ),
        Filters(
            min_price=50,
            max_price=200,
            min_accommodates=2,
            min_bedrooms=1,
            min_beds=2,
            min_bathrooms=1.0,
            max_minimum_nights=3,
            min_rating=4.5,
            min_location_score=4.0,
            min_reviews=5,
            instant_bookable=True,
            host_is_superhost=True,
            neighbourhood_ids=(1, 2),
            amenities=("Wifi", "Sauna"),
            near=NearFilter(latitude=51.5, longitude=-0.1, radius_km=2),
        ),
    ],
)
def test_every_placeholder_has_exactly_one_parameter(filters: Filters) -> None:
    for query in plan_for(filters).queries:
        assert rendered(query).count("%s") == len(query.params)


def test_the_parameter_order_matches_the_placeholder_order() -> None:
    query = only(plan_for(Filters(max_price=200, min_rating=4.5), limit=7))
    assert query.params == (EMBEDDING, 200.0, 4.5, EMBEDDING, 7)


# ------------------------------------------------------------------ errors --


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_below_one_is_rejected(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        plan_for(Filters(), limit=limit)


@pytest.mark.parametrize("cap", [0, -1])
def test_a_fanout_cap_below_one_is_rejected(cap: int) -> None:
    with pytest.raises(ValueError, match="fanout_cap must be at least 1"):
        plan_for(Filters(), fanout_cap=cap)


def test_an_empty_embedding_is_rejected() -> None:
    """There is nothing to rank by, and the query would silently mean nothing."""
    with pytest.raises(ValueError, match="query_embedding is empty"):
        plan_for(Filters(), query_embedding=[])


def test_an_empty_select_list_is_rejected_here_not_by_the_server() -> None:
    """`SELECT , "embedding" ...` is a syntax error raised far from its cause."""
    with pytest.raises(ValueError, match="select_columns is empty"):
        plan_for(Filters(), select_columns=())


# ------------------------------------------------------ the un-ranked form --


def test_where_clause_expresses_a_disjunction_without_fanning_out() -> None:
    """A count scans rows rather than index nodes, so ANY costs it nothing."""
    clause = where_clause(Filters(neighbourhood_ids=(1, 2, 3)), indexed_amenities=INDEXED_AMENITIES)
    assert clause.clause is not None
    assert clause.clause.as_string() == '"neighbourhood_id" = ANY(%s)'
    assert clause.params == ([1, 2, 3],)


def test_where_clause_is_none_when_nothing_is_constrained() -> None:
    clause = where_clause(Filters(), indexed_amenities=INDEXED_AMENITIES)
    assert clause.clause is None
    assert clause.params == ()
    assert not clause.is_post_filtered


def test_the_two_forms_agree_when_there_is_no_disjunction_to_fan_out() -> None:
    filters = Filters(min_rating=4.5, max_price=200, amenities=("Sauna",))
    clause = where_clause(filters, indexed_amenities=INDEXED_AMENITIES)
    plan = plan_for(filters)
    assert clause.null_excluding_columns == plan.null_excluding_columns
    assert clause.post_filter_columns == plan.post_filter_columns


def test_the_two_forms_deliberately_disagree_about_a_disjunction() -> None:
    """One ranks and fans out; the other counts and uses ANY. Both are right.

    Pinned because the flag reads as "recall was weakened", and that only means
    anything for the ranked form -- an exact count has no recall to lose. A
    caller must not read degradation into the un-ranked number.
    """
    filters = Filters(neighbourhood_ids=(1, 2))
    clause = where_clause(filters, indexed_amenities=INDEXED_AMENITIES)
    plan = plan_for(filters)

    assert clause.post_filter_columns == ("neighbourhood_id",)
    assert clause.is_post_filtered
    assert len(plan.queries) == 2
    assert plan.post_filter_columns == ()
    assert not plan.is_post_filtered


def test_a_zero_bound_still_excludes_unknown_values_and_reports_it() -> None:
    """`>= 0` constrains nothing arithmetically but drops every NULL.

    Dropping the condition instead would quietly admit listings the caller's
    own filter excluded, so it stands -- and the plan names the column so a
    caller loosening a filter can see that clearing beats zeroing.
    """
    plan = plan_for(Filters(min_bedrooms=0))
    assert where_of(only(plan)) == '"bedrooms" >= %s'
    assert plan.null_excluding_columns == ("bedrooms",)
    assert plan_for(Filters()).null_excluding_columns == ()
