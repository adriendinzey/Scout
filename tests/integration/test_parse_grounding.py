"""The parse vocabulary, read from real lookup tables.

The rows are made up. What is under test is which rows the vocabulary takes --
indexed amenities only, no catch-all property bucket -- and that a database with
nothing loaded is refused rather than rendered as an empty vocabulary.

Each fixture applies the migrations into a schema this test owns, so the
sandbox's own database is left alone.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import TupleRow

from scout.config import get_settings
from scout.data.migrate import apply_migrations
from scout.data.parsers import OTHER_PROPERTY_TYPE
from scout.graph.parse import GroundingError, grounding_for, load_grounding

pytestmark = pytest.mark.integration

LOADED_SCHEMA = "scout_parse_grounding_test"
EMPTY_SCHEMA = "scout_parse_grounding_empty_test"
BARE_SCHEMA = "scout_parse_grounding_bare_test"

# Deliberately out of order, so a vocabulary that trusted insertion order would
# render differently from one that sorts.
NEIGHBOURHOODS = ("Islington", "Camden", "Hackney")
ROOM_TYPES = ("Private room", "Entire home/apt")
PROPERTY_TYPES = ("Entire rental unit", OTHER_PROPERTY_TYPE, "Private room in home")
INDEXED_AMENITIES = ("Wifi", "Dedicated workspace")
UNINDEXED_AMENITIES = ("Hot tub", "Hair dryer")


def dsn_for(schema: str) -> str:
    """The sandbox database, with `schema` first on every new session's path."""
    return make_conninfo(get_settings().database_url, options=f"-csearch_path={schema},public")


def fresh_schema(conn: psycopg.Connection[TupleRow], schema: str, *, migrate: bool) -> None:
    conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
    conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
    if migrate:
        apply_migrations(conn)


def connect(schema: str) -> psycopg.Connection[TupleRow]:
    try:
        return psycopg.connect(dsn_for(schema), connect_timeout=5, autocommit=True)
    except psycopg.Error as exc:
        pytest.skip(f"database not reachable: {exc}")


def insert_names(conn: psycopg.Connection[TupleRow], table: str, names: tuple[str, ...]) -> None:
    for name in names:
        conn.execute(
            sql.SQL("INSERT INTO {} (name) VALUES (%s)").format(sql.Identifier(table)), (name,)
        )


def ids_of(conn: psycopg.Connection[TupleRow], table: str) -> dict[str, int]:
    query = sql.SQL("SELECT name, id FROM {}").format(sql.Identifier(table))
    return dict(conn.execute(query).fetchall())


@pytest.fixture(scope="module")
def loaded() -> Iterator[psycopg.Connection[TupleRow]]:
    """Migrated, with a snapshot row and a handful of lookups."""
    with connect(LOADED_SCHEMA) as conn:
        fresh_schema(conn, LOADED_SCHEMA, migrate=True)
        conn.execute(
            "INSERT INTO data_snapshot (snapshot_date, city, listing_count, review_count) "
            "VALUES (%s, %s, %s, %s)",
            (date(2026, 6, 19), "London", 3, 0),
        )
        insert_names(conn, "neighbourhoods", NEIGHBOURHOODS)
        insert_names(conn, "room_types", ROOM_TYPES)
        insert_names(conn, "property_types", PROPERTY_TYPES)
        insert_names(conn, "amenities", UNINDEXED_AMENITIES + INDEXED_AMENITIES)
        conn.execute(
            "UPDATE amenities SET is_indexed = true WHERE name = ANY(%s)",
            (list(INDEXED_AMENITIES),),
        )
        yield conn
        conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(LOADED_SCHEMA)))


class TestALoadedDatabase:
    def test_maps_every_lookup_name_to_its_id(self, loaded: psycopg.Connection[TupleRow]) -> None:
        grounding = load_grounding(loaded)

        assert grounding.city == "London"
        assert dict(grounding.neighbourhoods) == ids_of(loaded, "neighbourhoods")
        assert dict(grounding.room_types) == ids_of(loaded, "room_types")

    def test_leaves_out_the_catch_all_property_bucket(
        self, loaded: psycopg.Connection[TupleRow]
    ) -> None:
        """Offered to the model, "Other" would become the answer to "a penthouse"."""
        grounding = load_grounding(loaded)
        expected = {
            name: lookup_id
            for name, lookup_id in ids_of(loaded, "property_types").items()
            if name != OTHER_PROPERTY_TYPE
        }

        assert dict(grounding.property_types) == expected

    def test_offers_only_the_amenities_with_an_indexed_column(
        self, loaded: psycopg.Connection[TupleRow]
    ) -> None:
        assert load_grounding(loaded).amenities == frozenset(INDEXED_AMENITIES)

    def test_renders_sorted_whatever_order_the_rows_are_stored_in(
        self, loaded: psycopg.Connection[TupleRow]
    ) -> None:
        """Rows come back in storage order, which a reload or a VACUUM can change."""
        stored = [name for (name,) in loaded.execute("SELECT name FROM neighbourhoods")]
        rendered = load_grounding(loaded).render()
        positions = [rendered.index(f"- {name}\n") for name in sorted(NEIGHBOURHOODS)]

        assert stored != sorted(stored)
        assert positions == sorted(positions)


class TestADatabaseWithNothingToGroundIn:
    def test_no_snapshot_is_refused_rather_than_rendered_empty(self) -> None:
        with connect(EMPTY_SCHEMA) as conn:
            fresh_schema(conn, EMPTY_SCHEMA, migrate=True)
            try:
                with pytest.raises(GroundingError, match="scout data load"):
                    load_grounding(conn)
            finally:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(EMPTY_SCHEMA)))

    def test_a_snapshot_without_neighbourhoods_is_refused(self) -> None:
        with connect(EMPTY_SCHEMA) as conn:
            fresh_schema(conn, EMPTY_SCHEMA, migrate=True)
            conn.execute(
                "INSERT INTO data_snapshot (snapshot_date, city, listing_count, review_count) "
                "VALUES (%s, %s, %s, %s)",
                (date(2026, 6, 19), "London", 0, 0),
            )
            try:
                with pytest.raises(GroundingError, match="neighbourhoods"):
                    load_grounding(conn)
            finally:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(EMPTY_SCHEMA)))

    def test_missing_tables_raise_with_context_rather_than_a_raw_driver_error(self) -> None:
        with connect(BARE_SCHEMA) as conn:
            fresh_schema(conn, BARE_SCHEMA, migrate=False)
            # With public still on the path, the lookups would fall through to
            # whatever the sandbox migrated there. Only this empty schema remains.
            conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(BARE_SCHEMA)))
            try:
                with pytest.raises(GroundingError, match="could not read") as caught:
                    load_grounding(conn)
                assert isinstance(caught.value.__cause__, psycopg.Error)
            finally:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(BARE_SCHEMA)))


class TestTheProcessCache:
    def test_loads_once_per_database(self, loaded: psycopg.Connection[TupleRow]) -> None:
        grounding_for.cache_clear()
        try:
            first = grounding_for(dsn_for(LOADED_SCHEMA))
            second = grounding_for(dsn_for(LOADED_SCHEMA))
        finally:
            grounding_for.cache_clear()

        assert first is second
        assert first == load_grounding(loaded)

    def test_an_unreachable_database_raises_and_is_not_cached(self) -> None:
        # This database's own settings aimed at a port nothing listens on, rather
        # than a connection string written out: a DSN in a test is how a sandbox
        # ends up talking to someone else's database.
        unreachable = make_conninfo(get_settings().database_url, host="127.0.0.1", port="1")
        grounding_for.cache_clear()

        with pytest.raises(GroundingError, match="could not connect"):
            grounding_for(unreachable)
        assert grounding_for.cache_info().currsize == 0
