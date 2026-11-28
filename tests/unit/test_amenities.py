"""The amenity groups, tested against strings the snapshot actually contains.

Every spelling quoted here was taken from the 2026-06-19 London file rather than
invented, because the failure mode this guards against is a pattern that looks
right and matches nothing: `has_self_checkin` was silently 0.0% of listings
until a real "Self check-in" was run through it, and `has_tv` missed the plain
"TV" that 60% of listings use. Invented strings would have passed both.

The traps have a section of their own. Each one is a merge that a reader would
expect to happen and must not: a hair dryer is not a tumble dryer, a dishwasher
is not a washer, a garden *view* is not outdoor space, and paid parking is not
free parking.
"""

from __future__ import annotations

import re

import pytest

from scout.data.amenities import (
    AMENITY_COLUMN_BUDGET,
    AMENITY_GROUPS,
    FIXED_INDEX_FILTER_COLUMNS,
    FIXED_INDEX_KEY_COLUMNS,
    INDEXED_AMENITY_GROUPS,
    MAX_INDEX_KEY_COLUMNS,
    AmenityReport,
    booleans_for_listing,
    columns_for_listing,
    columns_for_spelling,
    count_amenity_groups,
    fold_amenity,
    groups_for_spelling,
    indexed_amenity_columns,
    indexed_amenity_names,
    render_report,
)
from scout.data.migrate import default_migrations_dir

# --------------------------------------------------------------------------- #
# Folding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Wifi", "wifi"),
        ("  Wifi  ", "wifi"),
        ("Dishes  and\tsilverware", "dishes and silverware"),
        # The three separators the source uses interchangeably.
        ("Free dryer – In unit", "free dryer - in unit"),
        ("Backyard - Fully fenced", "backyard - fully fenced"),
        ("Crib — available upon request", "crib - available upon request"),
        ("Free washer –In building", "free washer - in building"),
        # A hyphen inside a word is part of the word, not a separator.
        ("Self check-in", "self check-in"),
        ("Wi-Fi", "wi-fi"),
        # Curly and straight apostrophes are one apostrophe.
        ("Pack ’n play/Travel crib", "pack 'n play/travel crib"),
        ("Children’s books and toys", "children's books and toys"),
    ],
)
def test_folding_settles_spelling_without_changing_meaning(raw: str, expected: str) -> None:
    assert fold_amenity(raw) == expected


def test_folding_is_idempotent() -> None:
    """A folded string folds to itself, or the cache would depend on call order."""
    for group in AMENITY_GROUPS:
        folded = fold_amenity(group.name)
        assert fold_amenity(folded) == folded


# --------------------------------------------------------------------------- #
# Grouping: real spellings land in the right group
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "column"),
    [
        ("Dedicated workspace", "has_dedicated_workspace"),
        ("Self check-in", "has_self_checkin"),
        ("Bathtub", "has_bathtub"),
        ("Dishwasher", "has_dishwasher"),
        ("Private entrance", "has_private_entrance"),
        ("Elevator", "has_elevator"),
        ("Pets allowed", "has_pets_allowed"),
        # Dryer: free or paid, in the unit or in the building, is one appliance.
        ("Dryer", "has_dryer"),
        ("Free dryer – In unit", "has_dryer"),
        ("Free dryer – In building", "has_dryer"),
        ("Paid dryer – In unit", "has_dryer"),
        ("Paid dryer", "has_dryer"),
        ("Dryer – In building", "has_dryer"),
        # Outdoor space: four words for one question, private or shared.
        ("Patio or balcony", "has_outdoor_space"),
        ("Private patio or balcony", "has_outdoor_space"),
        ("Shared patio or balcony", "has_outdoor_space"),
        ("Backyard", "has_outdoor_space"),
        ("Private backyard – Fully fenced", "has_outdoor_space"),
        ("Shared backyard – Not fully fenced", "has_outdoor_space"),
        ("Backyard - Fully fenced", "has_outdoor_space"),
        # Free parking, with every shape the source gives it.
        ("Free parking on premises", "has_free_parking"),
        ("Free parking on premises – 1 space", "has_free_parking"),
        ("Free driveway parking on premises – 2 spaces", "has_free_parking"),
        ("Free parking garage on premises", "has_free_parking"),
        ("Free residential garage on premises – 16 spaces", "has_free_parking"),
        ("Free carport on premises – 1 space", "has_free_parking"),
        # A cot, however it is phrased, and whether or not it has to be asked for.
        ("Crib", "has_crib"),
        ("Crib - available upon request", "has_crib"),
        ("Crib - always at the listing", "has_crib"),
        ("Pack ’n play/Travel crib", "has_crib"),
        ("Paid pack ’n play/travel crib - available upon request", "has_crib"),
        # Cooling, including the units the source names rather than describes.
        ("Air conditioning", "has_air_conditioning"),
        ("Central air conditioning", "has_air_conditioning"),
        ("Portable air conditioning", "has_air_conditioning"),
        ("Window AC unit", "has_air_conditioning"),
        ("AC - split type ductless system", "has_air_conditioning"),
        # A gym at the property or in its building.
        ("Gym", "has_gym"),
        ("Gym in building", "has_gym"),
        ("Shared gym in building", "has_gym"),
        ("Private gym in building", "has_gym"),
        ("Exercise equipment", "has_gym"),
        ("Exercise equipment: free weights, yoga mat", "has_gym"),
    ],
)
def test_a_real_spelling_sets_its_column(raw: str, column: str) -> None:
    assert columns_for_spelling(raw) == {column}


@pytest.mark.parametrize(
    ("raw", "group_column"),
    [
        # Groups that were counted and deliberately left in the array. They
        # still have to match, or the report that justified excluding them is
        # measuring nothing.
        ("Wifi", "has_wifi"),
        ("Wi-Fi", "has_wifi"),
        ("Fast wifi – 73 Mbps", "has_wifi"),
        ("Wifi – 36 Mbps", "has_wifi"),
        ("Pocket wifi", "has_wifi"),
        ("Kitchen", "has_kitchen"),
        ("Kitchenette", "has_kitchen"),
        ("Washer", "has_washer"),
        ("Free washer – In unit", "has_washer"),
        ("Paid washer – In building", "has_washer"),
        ("Heating", "has_heating"),
        ("Central heating", "has_heating"),
        ("Radiant heating", "has_heating"),
        ("TV", "has_tv"),
        ("HDTV", "has_tv"),
        ("TV with standard cable", "has_tv"),
        ("55 inch HDTV with Amazon Prime Video, Netflix", "has_tv"),
        ("Long term stays allowed", "has_long_term_stays"),
        ("Hot tub", "has_hot_tub"),
        ("Private hot tub - available all year, open 24 hours", "has_hot_tub"),
        ("Free street parking", "has_free_street_parking"),
    ],
)
def test_a_counted_but_unindexed_group_still_matches(raw: str, group_column: str) -> None:
    assert [group.column for group in groups_for_spelling(raw)] == [group_column]
    assert columns_for_spelling(raw) == frozenset()


# --------------------------------------------------------------------------- #
# The traps: merges a reader would expect that must not happen
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "forbidden"),
    [
        # A hair dryer is in 68% of listings. Merging it into the dryer group
        # would have been the single largest error available here.
        ("Hair dryer", "has_dryer"),
        ("Garnier and hair dryer shampoo", "has_dryer"),
        # Two appliances that share six letters.
        ("Dishwasher", "has_washer"),
        ("Washer", "has_dishwasher"),
        # Not at the property.
        ("Laundromat nearby", "has_washer"),
        ("Gym nearby", "has_gym"),
        ("Shared gym nearby", "has_gym"),
        ("Private gym nearby", "has_gym"),
        # Paid is not free, which is the whole question a parking filter asks.
        ("Paid parking on premises", "has_free_parking"),
        ("Paid parking off premises", "has_free_parking"),
        ("Paid street parking off premises", "has_free_parking"),
        ("Paid parking lot off premises", "has_free_parking"),
        ("Paid parking garage off premises", "has_free_parking"),
        # Free, but a fact about the street rather than a space the property has.
        ("Free street parking", "has_free_parking"),
        # A window is not outside space.
        ("Garden view", "has_outdoor_space"),
        ("Courtyard view", "has_outdoor_space"),
        ("Pool view", "has_outdoor_space"),
        # A fan is not air conditioning.
        ("Portable fans", "has_air_conditioning"),
        ("Ceiling fan", "has_air_conditioning"),
        # Hardware the source lists beside self check-in, which is not the same
        # claim -- and a person with a key is the opposite of it.
        ("Lockbox", "has_self_checkin"),
        ("Keypad", "has_self_checkin"),
        ("Smart lock", "has_self_checkin"),
        ("Building staff", "has_self_checkin"),
        ("Doorman", "has_self_checkin"),
        # A kitchen outdoors is not the kitchen a guest means.
        ("Outdoor kitchen", "has_kitchen"),
        ("Shared outdoor kitchen", "has_kitchen"),
    ],
)
def test_a_near_synonym_does_not_join_the_group(raw: str, forbidden: str) -> None:
    assert forbidden not in {group.column for group in groups_for_spelling(raw)}


@pytest.mark.parametrize(
    "raw",
    [
        # Host free text, which the source suffixes with the category it was
        # typed under. Every one of these mentions a grouped amenity; none of
        # them *is* one, and only whole-string matching can tell.
        "There is a fridge in the kitchen for your use refrigerator",
        "Shared in the kitchen refrigerator",
        "Pizza oven as well as a regular fan oven oven",
        "Lamona Built In Single Fan Oven and Grill stainless steel single oven",
        "Sonos To Kitchen sound system",
        "Variable depending on supplies but always fancy :) shampoo",
        "Tower Fan conditioner",
        "OGX hair conditioner conditioner",
        "Whirlpool refrigerator",
        # Common amenities that belong to no group at all.
        "Smoke alarm",
        "Essentials",
        "Iron",
        "Hangers",
        "Shampoo",
        "Ethernet connection",
        "Breakfast",
        "Smoking allowed",
        "Pool table",
    ],
)
def test_an_ungrouped_spelling_claims_nothing(raw: str) -> None:
    assert groups_for_spelling(raw) == ()
    assert columns_for_spelling(raw) == frozenset()


def test_no_real_spelling_belongs_to_two_groups() -> None:
    """Overlapping groups would double-count the report and the columns.

    Not a property of the patterns in general -- a spelling meaning two
    amenities is allowed for, and `columns_for_spelling` returns a set for that
    reason -- but it is true of every spelling in this snapshot, and a change
    that broke it would be a merge nobody intended.
    """
    spellings = [
        "Wifi",
        "Kitchen",
        "Washer",
        "Dryer",
        "Free dryer – In unit",
        "Dishwasher",
        "Hair dryer",
        "Heating",
        "Central heating",
        "TV",
        "HDTV with Netflix",
        "Patio or balcony",
        "Private backyard – Fully fenced",
        "Free parking on premises",
        "Free street parking",
        "Paid parking on premises",
        "Crib",
        "Pack ’n play/Travel crib",
        "Gym",
        "Exercise equipment",
        "Air conditioning",
        "Window AC unit",
        "Self check-in",
        "Elevator",
        "Bathtub",
        "Private entrance",
        "Pets allowed",
        "Dedicated workspace",
        "Long term stays allowed",
        "Hot tub",
        "Laundromat nearby",
        "Outdoor kitchen",
        "Garden view",
    ]
    overlapping = {
        raw: [group.column for group in groups_for_spelling(raw)]
        for raw in spellings
        if len(groups_for_spelling(raw)) > 1
    }
    assert overlapping == {}


def test_every_group_claims_its_own_canonical_name() -> None:
    """The name is what the lookup table and a parsed filter will carry.

    A group whose own name its patterns reject would never map an amenity a
    query names to the column that filters it.
    """
    for group in AMENITY_GROUPS:
        assert group.matches(group.name), f"{group.column} does not match its own name"
        assert groups_for_spelling(group.name) == (group,)


# --------------------------------------------------------------------------- #
# Per-listing derivation
# --------------------------------------------------------------------------- #


def test_a_listings_columns_are_the_union_of_its_amenities() -> None:
    amenities = ["Wifi", "Kitchen", "Dedicated workspace", "Free dryer – In unit", "Hair dryer"]
    assert columns_for_listing(amenities) == {"has_dedicated_workspace", "has_dryer"}


def test_a_listing_with_no_amenities_sets_nothing() -> None:
    assert columns_for_listing([]) == frozenset()
    assert booleans_for_listing([]) == (False,) * len(INDEXED_AMENITY_GROUPS)


def test_the_booleans_are_in_column_order() -> None:
    """The backfill writes these positionally, so order is part of the contract."""
    columns = indexed_amenity_columns()
    flags = booleans_for_listing(["Pets allowed", "Elevator"])

    assert len(flags) == len(columns)
    assert {column for column, flag in zip(columns, flags, strict=True) if flag} == {
        "has_pets_allowed",
        "has_elevator",
    }


# --------------------------------------------------------------------------- #
# The selection, and the budget it has to fit
# --------------------------------------------------------------------------- #


def test_the_index_key_budget_is_not_exceeded() -> None:
    """The arithmetic the amenity set was chosen against, as an assertion.

    PostgreSQL allows 32 key columns per index; the embedding and the fixed
    filter columns spend the rest. An amenity group added without checking this
    would not fail until the index build, by which point the data is loaded.

    The fixed count is derived from the column *names* rather than written as a
    17, so that a migration adding a filter column moves the budget here instead
    of leaving a literal that is quietly one too large. An integration test ties
    that list to the columns the database really has.
    """
    assert 1 + len(FIXED_INDEX_FILTER_COLUMNS) == FIXED_INDEX_KEY_COLUMNS
    assert len(set(FIXED_INDEX_FILTER_COLUMNS)) == len(FIXED_INDEX_FILTER_COLUMNS)
    assert AMENITY_COLUMN_BUDGET == MAX_INDEX_KEY_COLUMNS - FIXED_INDEX_KEY_COLUMNS
    assert len(INDEXED_AMENITY_GROUPS) <= AMENITY_COLUMN_BUDGET
    assert FIXED_INDEX_KEY_COLUMNS + len(INDEXED_AMENITY_GROUPS) <= MAX_INDEX_KEY_COLUMNS
    # No amenity column may collide with a filter column of the same name.
    assert not set(indexed_amenity_columns()) & set(FIXED_INDEX_FILTER_COLUMNS)


def test_one_name_is_offered_per_indexed_amenity() -> None:
    """The vocabulary a model sees is 13 names, not every indexed spelling.

    `columns_for_spelling` maps all 141 of the snapshot's indexed spellings onto
    the 13 columns, which is what a filter needs; a prompt needs the opposite,
    or it offers the model "Free driveway parking on premises - 20 spaces".
    """
    names = indexed_amenity_names()

    assert len(names) == len(indexed_amenity_columns())
    assert len(set(names)) == len(names)
    for name in names:
        # Each name is itself a spelling of exactly the amenity it names.
        assert len(columns_for_spelling(name)) == 1


def test_every_column_name_is_distinct_and_shaped_like_a_flag() -> None:
    columns = [group.column for group in AMENITY_GROUPS]
    assert len(columns) == len(set(columns))
    for column in columns:
        assert re.fullmatch(r"has_[a-z0-9_]+", column), column


def test_every_group_says_why_it_was_kept_or_dropped() -> None:
    """The set is baked into the index, so the reasoning has to ship with it."""
    for group in AMENITY_GROUPS:
        assert len(group.why) > 40, f"{group.column} has no rationale"


def test_the_schema_creates_exactly_the_columns_the_code_derives() -> None:
    """The migration and the group table have to name the same 13 columns.

    They are two files, and nothing but this test ties them together: a group
    added in code without a migration would write to a column that does not
    exist, and a column created without a group would stay false forever while
    looking like a filter that works.
    """
    migration = (default_migrations_dir() / "0003_amenity_columns.sql").read_text()

    created = set(re.findall(r"ADD COLUMN (has_[a-z_]+)\s+bool NOT NULL DEFAULT false", migration))
    assert created == set(indexed_amenity_columns())

    # The CHECK constraint pinning what amenities.indexed_column may hold is a
    # third copy of the list, and the one that fails a backfill loudly.
    constraint = re.search(
        r"CONSTRAINT amenities_indexed_column_exists CHECK \((.*?)\n\);", migration, re.S
    )
    assert constraint is not None
    assert set(re.findall(r"'(has_[a-z_]+)'", constraint.group(1))) == set(
        indexed_amenity_columns()
    )


# --------------------------------------------------------------------------- #
# Counting and reporting
# --------------------------------------------------------------------------- #


def corpus() -> list[list[str]]:
    return [
        ["Wifi", "Kitchen", "Dedicated workspace", "Pets allowed"],
        ["Wifi", "Kitchen", "Dedicated workspace", "Hair dryer"],
        ["Wifi", "Smoke alarm"],
        [],
    ]


def test_counting_is_per_listing_not_per_spelling() -> None:
    """A listing that says "Wifi" twice is one listing with wifi."""
    report = count_amenity_groups([["Wifi", "Wi-Fi", "Fast wifi – 73 Mbps"]])

    counts = {count.group.column: count.listings for count in report.counts}
    assert counts["has_wifi"] == 1
    assert report.distinct_spellings == 3


def test_counting_ranks_by_prevalence_and_names_what_it_missed() -> None:
    report = count_amenity_groups(corpus(), uncovered_limit=2)

    assert report.listings == 4
    counts = {count.group.column: count.listings for count in report.counts}
    assert counts["has_wifi"] == 3
    assert counts["has_dedicated_workspace"] == 2
    assert counts["has_pets_allowed"] == 1
    assert counts["has_dryer"] == 0

    ranked = [count.listings for count in report.counts]
    assert ranked == sorted(ranked, reverse=True)

    # Hair dryer and Smoke alarm are in one listing each; the limit keeps two.
    assert dict(report.uncovered) == {"Smoke alarm": 1, "Hair dryer": 1}


def test_a_share_of_an_empty_corpus_is_zero_rather_than_an_error() -> None:
    report = count_amenity_groups([])

    assert report.listings == 0
    assert all(count.share(report.listings) == 0.0 for count in report.counts)
    assert "0 listings" in render_report(report)


def test_the_report_shows_the_ranking_the_budget_and_the_omissions() -> None:
    text = render_report(count_amenity_groups(corpus()))

    assert "75.00%" in text, "wifi is in three of four listings"
    assert "Dedicated workspace (has_dedicated_workspace)" in text
    assert f"{FIXED_INDEX_KEY_COLUMNS} fixed" in text
    assert f"of {MAX_INDEX_KEY_COLUMNS} allowed" in text
    # Both halves of the decision, not only the winners.
    assert "indexed, and why:" in text
    assert "considered and left in the array:" in text
    assert "Smoke alarm" in text


def test_the_report_marks_which_groups_got_a_column() -> None:
    report: AmenityReport = count_amenity_groups(corpus())
    text = render_report(report)

    for line in text.splitlines():
        if "(has_wifi)" in line:
            assert "array" in line
        if "(has_pets_allowed)" in line:
            assert "column" in line
