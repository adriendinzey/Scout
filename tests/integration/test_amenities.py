"""What the amenity backfill leaves in the database.

The boolean columns are derived data: every one of them has to agree with the
`amenities` array it was computed from, for every listing, or an amenity filter
returns the wrong listings while looking like it works. That agreement is what
this file asserts -- in both directions, because a column that is never true is
as wrong as one that is always true, and only one of those is visible.

The listings here are synthetic, with amenity *spellings* taken from the real
snapshot. The spellings are the part worth testing against reality; the
listings they hang off are not, and real ones cannot live in the repository.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from scout.config import get_settings
from scout.data.amenities import (
    AmenityError,
    backfill_amenity_columns,
    columns_for_listing,
    indexed_amenity_columns,
    indexed_amenity_names,
)
from scout.data.migrate import apply_migrations

pytestmark = pytest.mark.integration

TEST_SCHEMA = "scout_amenities_test"

EVERYTHING = "every indexed amenity at once"
NEAR_MISSES = "the near misses, which must set nothing"
UNIVERSAL = "the universal ones, none of which earned a column"

# One synthetic listing per interesting shape, with real spellings. The traps
# are deliberate: a hair dryer beside no dryer, a dishwasher beside no washer,
# paid parking beside no free parking, a garden view beside no outdoor space.
FIXTURE_AMENITIES: tuple[tuple[str, list[str]], ...] = (
    (
        EVERYTHING,
        [
            "Dedicated workspace",
            "Self check-in",
            "Bathtub",
            "Free dryer – In unit",
            "Dishwasher",
            "Private backyard – Fully fenced",
            "Free driveway parking on premises – 2 spaces",
            "Private entrance",
            "Elevator",
            "Pack ’n play/Travel crib",
            "Pets allowed",
            "Window AC unit",
            "Exercise equipment: free weights, yoga mat",
        ],
    ),
    (
        NEAR_MISSES,
        [
            "Hair dryer",
            "Dishwasher",
            "Paid parking on premises",
            "Free street parking",
            "Garden view",
            "Portable fans",
            "Lockbox",
            "Gym nearby",
            "Laundromat nearby",
            "Outdoor kitchen",
        ],
    ),
    (UNIVERSAL, ["Wifi", "Kitchen", "Washer", "Heating"]),
    ("no amenities at all", []),
    ("one amenity, written three ways", ["Wifi", "Wi-Fi", "Fast wifi – 73 Mbps"]),
)


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection[TupleRow]]:
    dsn = get_settings().database_url
    try:
        connection = psycopg.connect(dsn, connect_timeout=5, autocommit=True)
    except psycopg.Error as exc:
        pytest.skip(f"database not reachable: {exc}")
    with connection:
        yield connection


@pytest.fixture(scope="module")
def loaded(conn: psycopg.Connection[TupleRow]) -> Iterator[dict[int, list[str]]]:
    """Scout's schema in a throwaway namespace, with the synthetic listings in it."""
    drop = sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(TEST_SCHEMA))
    conn.execute(drop)
    conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(TEST_SCHEMA)))
    conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(TEST_SCHEMA)))
    apply_migrations(conn)

    ids = conn.execute(
        """
        WITH n AS (INSERT INTO neighbourhoods (name) VALUES ('Hackney') RETURNING id),
             r AS (INSERT INTO room_types (name) VALUES ('Entire home/apt') RETURNING id),
             p AS (INSERT INTO property_types (name) VALUES ('Entire rental unit') RETURNING id)
        SELECT n.id, r.id, p.id FROM n, r, p
        """
    ).fetchone()
    assert ids

    by_listing: dict[int, list[str]] = {}
    for index, (name, amenities) in enumerate(FIXTURE_AMENITIES):
        row = conn.execute(
            """
            INSERT INTO listings (
                source_listing_id, name, neighbourhood_id, room_type_id, property_type_id,
                latitude, longitude, accommodates, minimum_nights, number_of_reviews, amenities
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (1_000 + index, name, *ids, 51.5, -0.1, 2, 1, 0, amenities),
        ).fetchone()
        assert row
        by_listing[int(row[0])] = amenities

    # The lookup table the load fills, so the backfill has rows to flag.
    conn.execute(
        "INSERT INTO amenities (name) SELECT DISTINCT unnest(%s::text[]) "
        "ON CONFLICT (name) DO NOTHING",
        (sorted({amenity for _, amenities in FIXTURE_AMENITIES for amenity in amenities}),),
    )

    yield by_listing
    conn.execute(drop)


def stored_flags(conn: psycopg.Connection[TupleRow], listing_id: int) -> dict[str, bool]:
    columns = indexed_amenity_columns()
    statement = sql.SQL("SELECT {columns} FROM listings WHERE id = {id}").format(
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        id=sql.Placeholder(),
    )
    row = conn.execute(statement, (listing_id,)).fetchone()
    assert row
    return dict(zip(columns, (bool(value) for value in row), strict=True))


def test_every_boolean_agrees_with_the_array_it_was_derived_from(conn, loaded):
    report = backfill_amenity_columns(conn)
    assert report.listings == len(FIXTURE_AMENITIES)
    assert report.columns == indexed_amenity_columns()

    for listing_id, amenities in loaded.items():
        expected = columns_for_listing(amenities)
        flags = stored_flags(conn, listing_id)
        true_columns = {column for column, value in flags.items() if value}
        assert true_columns == set(expected), f"listing {listing_id} ({amenities})"


def listing_called(conn: psycopg.Connection[TupleRow], label: str) -> int:
    """The fixture listing with this name.

    By label rather than by amenity count: counting would quietly pick a
    different listing the moment the indexed set changes size, and then assert
    the opposite of what the test is named for.
    """
    row = conn.execute("SELECT id FROM listings WHERE name = %s", (label,)).fetchone()
    assert row, f"no fixture listing called {label!r}"
    return int(row[0])


def test_the_listing_with_everything_sets_every_column(conn, loaded):
    backfill_amenity_columns(conn)

    assert all(stored_flags(conn, listing_called(conn, EVERYTHING)).values())


def test_the_near_misses_set_only_the_column_they_really_are(conn, loaded):
    """A hair dryer, paid parking and a garden view are not what they resemble.

    The listing holds ten spellings that each resemble an indexed amenity and
    one that is one: a dishwasher, which is in the list precisely because it
    must not reach `has_washer`.
    """
    backfill_amenity_columns(conn)

    flags = stored_flags(conn, listing_called(conn, NEAR_MISSES))
    assert {column for column, value in flags.items() if value} == {"has_dishwasher"}


def test_a_changed_amenity_list_takes_its_flag_with_it(conn, loaded):
    """The booleans are recomputed, never nudged: a stale true is unfindable."""
    backfill_amenity_columns(conn)
    listing_id = listing_called(conn, EVERYTHING)
    assert stored_flags(conn, listing_id)["has_elevator"]

    # Rolled back rather than restored by hand, so a failed assertion cannot
    # leave the fixture edited for whatever test runs next.
    with conn.transaction(force_rollback=True):
        conn.execute(
            "UPDATE listings SET amenities = %s WHERE id = %s",
            (["Wifi"], listing_id),
        )
        backfill_amenity_columns(conn)
        assert not any(stored_flags(conn, listing_id).values())

    assert stored_flags(conn, listing_id)["has_elevator"]


def test_running_it_again_writes_nothing(conn, loaded):
    """Both updates are restricted to rows that actually differ.

    Without that, every backfill rewrites all 92,000 listings whether or not a
    flag moved -- and on a reload that is a second full-table write inside the
    same transaction as the upsert that already rewrote it.
    """
    backfill_amenity_columns(conn)
    first = backfill_amenity_columns(conn)
    second = backfill_amenity_columns(conn)

    assert second.per_column == first.per_column
    assert second.lookup_rows_flagged == first.lookup_rows_flagged
    assert (second.listings_changed, second.lookup_rows_changed) == (0, 0)

    # ctid moves when a row is rewritten, which is the only way to see a write
    # that changed no value.
    before = conn.execute("SELECT id, ctid FROM listings ORDER BY id").fetchall()
    backfill_amenity_columns(conn)
    assert conn.execute("SELECT id, ctid FROM listings ORDER BY id").fetchall() == before


def test_a_row_whose_flags_moved_is_the_only_one_written(conn, loaded):
    backfill_amenity_columns(conn)
    listing_id = listing_called(conn, UNIVERSAL)

    with conn.transaction(force_rollback=True):
        conn.execute("UPDATE listings SET amenities = %s WHERE id = %s", (["Elevator"], listing_id))

        changed = backfill_amenity_columns(conn)
        assert changed.listings_changed == 1
        assert stored_flags(conn, listing_id)["has_elevator"]


def test_the_lookup_maps_each_spelling_to_the_column_that_filters_it(conn, loaded):
    backfill_amenity_columns(conn)

    mapped = dict(
        conn.execute("SELECT name, indexed_column FROM amenities WHERE is_indexed").fetchall()
    )
    # This mapping is what the retrieval layer reads to decide whether an
    # amenity filter can be pushed down, so a spelling variant has to resolve to
    # the same column as the plain name.
    assert mapped["Elevator"] == "has_elevator"
    assert mapped["Free dryer – In unit"] == "has_dryer"
    assert mapped["Free driveway parking on premises – 2 spaces"] == "has_free_parking"
    assert mapped["Pack ’n play/Travel crib"] == "has_crib"
    assert mapped["Exercise equipment: free weights, yoga mat"] == "has_gym"

    unindexed = {
        str(name)
        for (name,) in conn.execute("SELECT name FROM amenities WHERE NOT is_indexed").fetchall()
    }
    # Deliberately not indexed, however often they are asked for.
    assert {"Wifi", "Kitchen", "Washer", "Heating", "Free street parking"} <= unindexed


def test_the_lookup_offers_one_name_per_column_to_the_parse_vocabulary(conn, loaded):
    """`is_canonical` is the vocabulary; `indexed_column` is the filter mapping.

    The fixture holds several spellings of the same amenity, so selecting on
    `is_indexed` would offer a model two ways to say one thing -- and against
    the real snapshot, 141 ways to say thirteen.
    """
    report = backfill_amenity_columns(conn)

    canonical = {
        str(name)
        for (name,) in conn.execute("SELECT name FROM amenities WHERE is_canonical").fetchall()
    }
    assert canonical <= set(indexed_amenity_names())
    assert report.lookup_rows_canonical == len(canonical)
    # Every canonical row names a column, and no two name the same one.
    columns = conn.execute(
        "SELECT indexed_column, count(*) FROM amenities WHERE is_canonical GROUP BY 1"
    ).fetchall()
    assert all(column is not None and count == 1 for column, count in columns)

    # The spelling variants are mapped but not offered.
    variants = conn.execute(
        "SELECT count(*) FROM amenities WHERE is_indexed AND NOT is_canonical"
    ).fetchone()
    assert variants and variants[0] > 0


def test_a_canonical_spelling_of_no_column_is_refused(conn, loaded):
    """The two columns cannot drift into a state that means nothing."""
    backfill_amenity_columns(conn)

    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        conn.execute("UPDATE amenities SET is_canonical = true WHERE indexed_column IS NULL")


def test_a_schema_without_the_columns_says_so_rather_than_failing_obscurely(conn, loaded):
    """A database that predates the amenity set must not look like empty data."""
    backfill_amenity_columns(conn)
    missing = indexed_amenity_columns()[0]
    assert stored_flags(conn, listing_called(conn, EVERYTHING))[missing]

    # DDL is transactional in PostgreSQL, so the rollback puts the column back
    # with its derived values rather than re-adding an empty one.
    with pytest.raises(AmenityError, match="apply the migrations"), conn.transaction():
        conn.execute(sql.SQL("ALTER TABLE listings DROP COLUMN {}").format(sql.Identifier(missing)))
        backfill_amenity_columns(conn)

    assert stored_flags(conn, listing_called(conn, EVERYTHING))[missing]


def test_an_empty_table_cannot_hide_a_missing_column(conn, loaded):
    """The schema is checked before the row count can make it look fine.

    With no rows, nothing touches the boolean columns, so a backfill that only
    discovered the schema through a failing UPDATE would return a clean report
    on a database that has nowhere to put the answer -- a failure reported as an
    empty result, which is the one thing this project will not do.
    """
    missing = indexed_amenity_columns()[-1]

    with pytest.raises(AmenityError, match="apply the migrations"), conn.transaction():
        conn.execute("DELETE FROM reviews")
        conn.execute("DELETE FROM listings")
        conn.execute(sql.SQL("ALTER TABLE listings DROP COLUMN {}").format(sql.Identifier(missing)))
        backfill_amenity_columns(conn)

    assert conn.execute("SELECT count(*) FROM listings").fetchone() == (len(FIXTURE_AMENITIES),)


def test_an_empty_corpus_on_a_correct_schema_is_not_an_error(conn, loaded):
    """ "Nothing loaded yet" is a fact about the data, not a failure."""
    with conn.transaction(force_rollback=True):
        conn.execute("DELETE FROM reviews")
        conn.execute("DELETE FROM listings")

        report = backfill_amenity_columns(conn)
        assert (report.listings, report.listings_changed) == (0, 0)
        assert report.columns == indexed_amenity_columns()
