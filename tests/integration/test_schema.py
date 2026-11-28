"""What the migrations actually produced.

Two things are asserted here that fail silently everywhere else. The first is
privacy: no column matching a dropped personal field may exist in any Scout
table, and a stray column is far easier to catch here than in a review of the
loader four tasks later. The second is column typing: Brindle refuses a filter
column that is not one of its accepted scalar types, but it refuses it at
CREATE INDEX -- long after the schema was written and the data loaded.

The migrations are applied into a throwaway schema rather than `public`, so the
test is repeatable and leaves the sandbox database alone.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from scout.config import get_settings
from scout.data.amenities import (
    AMENITY_COLUMN_BUDGET,
    FIXED_INDEX_FILTER_COLUMNS,
    FIXED_INDEX_KEY_COLUMNS,
    indexed_amenity_columns,
)
from scout.data.migrate import apply_migrations

pytestmark = pytest.mark.integration

TEST_SCHEMA = "scout_schema_test"

SCOUT_TABLES = frozenset(
    {
        "listings",
        "reviews",
        "neighbourhoods",
        "room_types",
        "property_types",
        "amenities",
        "data_snapshot",
    }
)

# docs/DATA.md section 4. These never reach the database: they are dropped while
# parsing, before the first INSERT, rather than stored and nulled.
DROPPED_PERSONAL_FIELDS = frozenset(
    {
        "host_id",
        "host_name",
        "host_about",
        "host_url",
        "host_thumbnail_url",
        "host_picture_url",
        "listing_url",
        "reviewer_id",
        "reviewer_name",
    }
)

# The types Brindle will accept as index key columns. Anything else -- text,
# timestamps, arrays -- is refused when the index is created.
BRINDLE_FILTERABLE_TYPES = frozenset(
    {"boolean", "smallint", "integer", "bigint", "real", "double precision"}
)

# Every listings column intended to be a filter, with the type it has to keep.
FILTERABLE_COLUMNS = {
    "neighbourhood_id": "integer",
    "room_type_id": "smallint",
    "property_type_id": "smallint",
    "price_gbp": "real",
    "accommodates": "smallint",
    "bedrooms": "smallint",
    "beds": "smallint",
    "bathrooms": "real",
    "minimum_nights": "integer",
    "rating": "real",
    "location_score": "real",
    "number_of_reviews": "integer",
    "instant_bookable": "boolean",
    "host_is_superhost": "boolean",
    "latitude": "double precision",
    "longitude": "double precision",
}

MAX_INDEX_KEY_COLUMNS = 32  # PostgreSQL's limit, not Brindle's.
# What is left for amenity booleans once the vector and the filters above are
# spent. The migration documents this number and the amenity selection is
# sized against it.
AMENITY_COLUMN_HEADROOM = MAX_INDEX_KEY_COLUMNS - 1 - len(FILTERABLE_COLUMNS)
# The amenity booleans 0003 created, which now occupy part of that headroom.
AMENITY_COLUMNS = indexed_amenity_columns()

# Nullable on purpose. The rest of listings is NOT NULL, and the distinction is
# load-bearing: a NULL satisfies no comparison, so `rating >= 4.5` excludes
# every unrated listing instead of including it. instant_bookable joined them
# in 0002: the snapshot stopped publishing the field, and false would have been
# an answer the source never gave.
NULLABLE_LISTING_COLUMNS = frozenset(
    {
        "description",
        "price_gbp",
        "bedrooms",
        "beds",
        "bathrooms",
        "rating",
        "location_score",
        "cleanliness_score",
        "instant_bookable",
        "host_is_superhost",
        "doc_text",
        "embedding",
    }
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
def migrated(conn: psycopg.Connection[TupleRow]) -> Iterator[list[str]]:
    """Scout's schema, applied fresh into a throwaway namespace."""
    drop = sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(TEST_SCHEMA))
    conn.execute(drop)
    conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(TEST_SCHEMA)))
    # brindle_vector lives in public, so it has to stay on the path.
    conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(TEST_SCHEMA)))
    yield apply_migrations(conn)
    conn.execute(drop)


def columns_of(conn: psycopg.Connection[TupleRow], table: str) -> dict[str, tuple[str, bool]]:
    """column name -> (data type, is nullable) for one table in the test schema."""
    rows = conn.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s",
        (TEST_SCHEMA, table),
    ).fetchall()
    return {str(name): (str(kind), nullable == "YES") for name, kind, nullable in rows}


def test_migrations_create_every_expected_table(conn, migrated):
    assert migrated == ["0001", "0002", "0003"]

    present = {
        str(row[0])
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
            (TEST_SCHEMA,),
        ).fetchall()
    }
    assert present >= SCOUT_TABLES
    assert "schema_migrations" in present, "the runner must record what it applied"


def test_applying_again_is_a_no_op(conn, migrated):
    """Re-running must not re-apply: the CREATE TABLEs would fail if it did."""
    assert apply_migrations(conn) == []
    assert apply_migrations(conn) == []


def test_no_personal_column_exists_in_any_scout_table(conn, migrated):
    """Privacy is asserted, not trusted. A stray column here outranks features."""
    offenders = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND column_name = ANY(%s)",
        (TEST_SCHEMA, sorted(DROPPED_PERSONAL_FIELDS)),
    ).fetchall()
    assert offenders == [], f"personal columns reached the schema: {offenders}"

    # reviewer_* is never legitimate, whatever it is suffixed with. (host_* is
    # not checked this way: host_is_superhost is a service-level trait, kept.)
    reviewer_columns = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND column_name LIKE 'reviewer%%'",
        (TEST_SCHEMA,),
    ).fetchall()
    assert reviewer_columns == [], f"reviewer identity reached the schema: {reviewer_columns}"


def test_reviews_carry_only_what_a_citation_needs(conn, migrated):
    assert set(columns_of(conn, "reviews")) == {"id", "listing_id", "date", "comments"}


def test_filterable_columns_have_types_brindle_accepts(conn, migrated):
    """Brindle refuses a bad filter column at CREATE INDEX -- four tasks later."""
    listings = columns_of(conn, "listings")

    for column, expected_type in FILTERABLE_COLUMNS.items():
        assert column in listings, f"{column} is missing from listings"
        actual_type = listings[column][0]
        assert actual_type == expected_type, f"{column} is {actual_type}, expected {expected_type}"
        assert actual_type in BRINDLE_FILTERABLE_TYPES

    # The amenity booleans are filter columns like any other, and the one thing
    # the retrieval layer assumes about them is that they are never NULL: it
    # treats them as always-known, so a nullable one would turn an amenity
    # filter into a silent exclusion of every listing without the flag set.
    for column in AMENITY_COLUMNS:
        assert column in listings, f"{column} is missing from listings"
        column_type, is_nullable = listings[column]
        assert column_type == "boolean", f"{column} is {column_type}"
        assert not is_nullable, f"{column} must be NOT NULL"


def test_the_fixed_filter_columns_are_the_ones_the_code_budgets_for(conn, migrated):
    """The amenity budget is 32 minus these, so the two lists have to be one list.

    `src/scout/data/amenities.py` names the fixed filter columns to derive how
    many amenity slots are left. If a migration adds a filter column and that
    list is not updated, the budget silently stays one too large and the index
    build is where it surfaces.
    """
    assert set(FIXED_INDEX_FILTER_COLUMNS) == set(FILTERABLE_COLUMNS)
    assert 1 + len(FILTERABLE_COLUMNS) == FIXED_INDEX_KEY_COLUMNS
    assert AMENITY_COLUMN_BUDGET == AMENITY_COLUMN_HEADROOM


def test_the_index_key_budget_leaves_room_for_the_amenity_columns(conn, migrated):
    """The index this schema exists to carry can actually be built on it.

    Arithmetic over a list of column names would pass even if one of those
    columns were retyped to something Brindle refuses, so this builds the real
    thing on a copy of `listings`: the embedding, all 16 filter columns, the
    amenity booleans 0003 created, and enough stand-ins to fill the rest of the
    headroom -- exactly 32 key columns, PostgreSQL's limit. Empty table, so the
    build is instant.

    The stand-ins exist because the real amenity set deliberately leaves slots
    unspent. Testing only the columns that exist today would stop proving that
    the schema has room for the budget the migration claims.

    This is not the production index. That one is created after the rows land,
    because Brindle rewrites the whole index blob on every write.
    """
    assert 1 + len(FILTERABLE_COLUMNS) + AMENITY_COLUMN_HEADROOM == MAX_INDEX_KEY_COLUMNS
    unspent = AMENITY_COLUMN_HEADROOM - len(AMENITY_COLUMNS)
    assert unspent >= 0, "the amenity set is over the index key budget"

    probe = sql.Identifier(TEST_SCHEMA, "index_budget_probe")
    # LIKE INCLUDING ALL already copied the real amenity booleans, so only the
    # unspent slots are added here; adding 15 again would be 45 key columns.
    amenity_columns = [sql.Identifier(column) for column in AMENITY_COLUMNS]
    spare_columns = [sql.Identifier(f"has_spare_amenity_{n}") for n in range(unspent)]

    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(probe))
    conn.execute(
        sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING ALL)").format(
            probe, sql.Identifier(TEST_SCHEMA, "listings")
        )
    )
    for column in spare_columns:
        conn.execute(
            sql.SQL("ALTER TABLE {} ADD COLUMN {} bool NOT NULL DEFAULT false").format(
                probe, column
            )
        )

    keys = sql.SQL(", ").join(
        [sql.SQL("embedding brindle_vector_cosine_ops")]
        + [sql.Identifier(name) for name in FILTERABLE_COLUMNS]
        + amenity_columns
        + spare_columns
    )
    try:
        # Left to raise: a refused column type or an over-budget key count both
        # come back as a Postgres error naming the offending column, which says
        # more than any message this test could wrap it in.
        conn.execute(
            sql.SQL(
                "CREATE INDEX index_budget_probe_brindle ON {} USING brindle ({}) "
                "WITH (m = 16, ef_construction = 64)"
            ).format(probe, keys)
        )
    finally:
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(probe))


def test_nullability_is_deliberate(conn, migrated):
    listings = columns_of(conn, "listings")
    nullable = {column for column, (_, is_nullable) in listings.items() if is_nullable}

    assert nullable == set(NULLABLE_LISTING_COLUMNS)


def test_a_listing_and_its_reviews_round_trip(conn, migrated):
    """The types are usable, the keys line up, and the vector column accepts one."""
    neighbourhood_id = conn.execute(
        "INSERT INTO neighbourhoods (name) VALUES (%s) RETURNING id", ("Testington",)
    ).fetchone()
    room_type_id = conn.execute(
        "INSERT INTO room_types (name) VALUES (%s) RETURNING id", ("Entire home/apt",)
    ).fetchone()
    property_type_id = conn.execute(
        "INSERT INTO property_types (name) VALUES (%s) RETURNING id", ("Entire rental unit",)
    ).fetchone()
    assert neighbourhood_id and room_type_id and property_type_id

    listing_id = conn.execute(
        """
        INSERT INTO listings (
            source_listing_id, name, description,
            neighbourhood_id, room_type_id, property_type_id,
            latitude, longitude, price_gbp, accommodates, bedrooms, beds, bathrooms,
            minimum_nights, rating, location_score, cleanliness_score, number_of_reviews,
            instant_bookable, host_is_superhost, amenities, doc_text, embedding
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s::real[]::brindle_vector
        ) RETURNING id
        """,
        (
            123456789,
            "A quiet flat by the canal",
            "Two minutes from the water.",
            neighbourhood_id[0],
            room_type_id[0],
            property_type_id[0],
            51.5072,
            -0.1276,
            149.0,
            2,
            1,
            1,
            1.0,
            2,
            4.8,
            4.9,
            4.7,
            37,
            True,
            True,
            ["Wifi", "Kitchen"],
            "A quiet flat by the canal. Two minutes from the water.",
            [0.1, 0.2, 0.3],
        ),
    ).fetchone()
    assert listing_id

    conn.execute(
        "INSERT INTO reviews (listing_id, date, comments) VALUES (%s, %s, %s)",
        (listing_id[0], "2025-06-01", "Lovely and quiet, exactly as described."),
    )

    stored = conn.execute(
        "SELECT price_gbp, accommodates, amenities, embedding IS NOT NULL "
        "FROM listings WHERE id = %s",
        (listing_id[0],),
    ).fetchone()
    assert stored == (149.0, 2, ["Wifi", "Kitchen"], True)

    review_count = conn.execute(
        "SELECT count(*) FROM reviews WHERE listing_id = %s", (listing_id[0],)
    ).fetchone()
    assert review_count == (1,)

    # Reviews exist only to cite a listing; nothing keeps them once it is gone.
    conn.execute("DELETE FROM listings WHERE id = %s", (listing_id[0],))
    orphans = conn.execute("SELECT count(*) FROM reviews").fetchone()
    assert orphans == (0,)


def test_a_null_attribute_satisfies_no_comparison(conn, migrated):
    """`rating >= 4.5` excludes unrated listings. Parse and Check depend on it."""
    # Inserted here rather than leaned on from another test: a test that only
    # passes when the whole module runs in order is not much of a test.
    ids = conn.execute(
        """
        WITH n AS (
            INSERT INTO neighbourhoods (name) VALUES ('Nullington') RETURNING id
        ), r AS (
            INSERT INTO room_types (name) VALUES ('Private room') RETURNING id
        ), p AS (
            INSERT INTO property_types (name) VALUES ('Private room in home') RETURNING id
        )
        SELECT n.id, r.id, p.id FROM n, r, p
        """
    ).fetchone()
    assert ids and all(value is not None for value in ids)

    conn.execute(
        """
        INSERT INTO listings (
            source_listing_id, name, neighbourhood_id, room_type_id, property_type_id,
            latitude, longitude, accommodates, minimum_nights, number_of_reviews,
            instant_bookable
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (987654321, "Brand new, unrated", *ids, 51.5, -0.1, 2, 1, 0, False),
    )

    matched = conn.execute(
        "SELECT count(*) FROM listings WHERE source_listing_id = %s AND rating >= 0.0",
        (987654321,),
    ).fetchone()
    assert matched == (0,), "a NULL rating must satisfy no comparison"

    conn.execute("DELETE FROM listings WHERE source_listing_id = %s", (987654321,))
    conn.execute("DELETE FROM neighbourhoods WHERE name = 'Nullington'")
    conn.execute("DELETE FROM room_types WHERE name = 'Private room'")
    conn.execute("DELETE FROM property_types WHERE name = 'Private room in home'")


def test_the_amenity_lookup_names_the_column_that_filters_each_amenity(conn, migrated):
    """The lookup is what the retrieval layer reads to map a name to a column.

    `is_indexed` is generated from `indexed_column` rather than stored beside
    it: two hand-maintained columns that have to agree eventually do not, and
    the symptom would be an amenity the index can filter that nothing knows to
    filter with -- or worse, one it cannot.
    """
    columns = columns_of(conn, "amenities")
    assert columns["indexed_column"] == ("text", True)
    assert columns["is_indexed"] == ("boolean", False)

    indexed = conn.execute(
        "INSERT INTO amenities (name, indexed_column) VALUES (%s, %s) "
        "RETURNING indexed_column, is_indexed",
        ("Elevator", "has_elevator"),
    ).fetchone()
    assert indexed == ("has_elevator", True)

    unindexed = conn.execute(
        "INSERT INTO amenities (name) VALUES (%s) RETURNING indexed_column, is_indexed",
        ("Iron",),
    ).fetchone()
    assert unindexed == (None, False)

    with pytest.raises(psycopg.errors.GeneratedAlways):
        conn.execute("UPDATE amenities SET is_indexed = true WHERE name = 'Iron'")

    # A lookup row pointing at a column that does not exist would surface as a
    # failed index build or a query against a missing column, long after the
    # row was written.
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO amenities (name, indexed_column) VALUES (%s, %s)",
            ("Nonsense", "has_not_a_real_column"),
        )

    conn.execute("DELETE FROM amenities WHERE name IN ('Elevator', 'Iron')")


def test_every_indexed_amenity_column_the_code_derives_exists(conn, migrated):
    """The group table and the migration are two files with one list between them."""
    listings = columns_of(conn, "listings")
    missing = [column for column in AMENITY_COLUMNS if column not in listings]
    assert missing == [], f"the code derives columns the schema does not have: {missing}"

    # And nothing the schema has is left unwritten: a has_* column with no group
    # behind it stays false forever while looking like a filter that works.
    orphans = [
        column
        for column in listings
        if column.startswith("has_") and column not in set(AMENITY_COLUMNS)
    ]
    assert orphans == [], f"the schema has amenity columns no group derives: {orphans}"
