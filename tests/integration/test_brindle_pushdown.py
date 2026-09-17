"""The premise, asserted.

Scout exists because a predicate can be pushed *into* the vector index instead of
applied after it. If that stops being true — a Brindle bump, a planner change, an
index declared with the filter columns in the wrong place — every number Scout
publishes becomes meaningless while everything still appears to work.

So it is a test, not an assumption. These run against the Compose database on a
small synthetic fixture; no real listing data is involved.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest

pytestmark = pytest.mark.integration

DSN = os.environ.get("SCOUT_DATABASE_URL", "postgresql://scout:scout@localhost:5433/scout")

# Small and low-dimensional: this fixture tests the query *plan* and the
# correctness of what comes back, not retrieval quality.
ROWS = 2_000
DIMS = 8


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    try:
        connection = psycopg.connect(DSN, connect_timeout=5)
    except psycopg.Error as exc:
        pytest.skip(f"database not reachable at {DSN}: {exc}")
    with connection:
        yield connection


@pytest.fixture(scope="module")
def fixture_table(conn: psycopg.Connection[tuple[object, ...]]) -> Iterator[str]:
    """A Scout-shaped table: a vector plus every filterable column type."""
    table = "test_pushdown_listings"
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"""
        CREATE TABLE {table} (
            id               int4 PRIMARY KEY,
            neighbourhood_id int4  NOT NULL,
            room_type_id     int2  NOT NULL,
            price_usd        float4,
            accommodates     int2  NOT NULL,
            rating           float4,
            has_wifi         bool  NOT NULL,
            embedding        brindle_vector
        )
    """)
    # Every 7th row has a NULL rating, so the NULL-semantics test below has
    # something to find.
    conn.execute(
        f"""
        INSERT INTO {table}
        SELECT g,
               (g %% 33) + 1,
               ((g %% 3) + 1)::int2,
               (40 + (g %% 400))::float4,
               ((g %% 6) + 1)::int2,
               CASE WHEN g %% 7 = 0 THEN NULL ELSE (3.0 + (g %% 20) / 10.0)::float4 END,
               (g %% 2 = 0),
               (SELECT array_agg(((sin(g * 0.7 + d) + 1) / 2)::real ORDER BY d)
                  FROM generate_series(1, %s) d)::real[]::brindle_vector
        FROM generate_series(1, %s) g
        """,
        (DIMS, ROWS),
    )
    # All rows first, THEN the index: Brindle's interim storage rewrites the whole
    # blob on every write, so row-by-row insertion into an indexed table is
    # pathological.
    conn.execute("SET maintenance_work_mem = '512MB'")
    conn.execute(f"""
        CREATE INDEX {table}_brindle ON {table} USING brindle (
            embedding brindle_vector_cosine_ops,
            neighbourhood_id, room_type_id, price_usd, accommodates, rating, has_wifi
        ) WITH (m = 16, ef_construction = 64)
    """)
    conn.execute(f"ANALYZE {table}")
    conn.commit()
    yield table
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


def _plan(conn: psycopg.Connection[tuple[object, ...]], sql: str) -> str:
    rows = conn.execute(f"EXPLAIN (COSTS OFF) {sql}").fetchall()
    return "\n".join(str(row[0]) for row in rows)


def test_extensions_are_present(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    """Both live in one database so retrieval baselines compare identical rows."""
    found = dict(
        conn.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname IN ('brindle', 'vector')"
        ).fetchall()  # type: ignore[arg-type]
    )
    assert "brindle" in found, "brindle extension missing"
    assert "vector" in found, "pgvector missing — the evaluation baseline needs it"


def test_predicate_is_pushed_into_the_index(
    conn: psycopg.Connection[tuple[object, ...]], fixture_table: str
) -> None:
    """The whole point: `Index Cond`, never a post-scan `Filter`.

    A plan that degrades to Filter still returns correct rows, so nothing looks
    broken — it just quietly stops being the thing Scout is measuring.
    """
    conn.execute("SET brindle.ef_search = 100")
    conn.execute("SET enable_seqscan = off")
    sql = f"""
        SELECT id FROM {fixture_table}
        WHERE price_usd < 200 AND accommodates >= 2 AND has_wifi
        ORDER BY embedding <=> (SELECT embedding FROM {fixture_table} WHERE id = 1)
        LIMIT 10
    """
    plan = _plan(conn, sql)

    assert f"{fixture_table}_brindle" in plan, f"the Brindle index was not used:\n{plan}"
    assert "Index Cond" in plan, f"predicate was not pushed into the index:\n{plan}"
    # The ordering must also come from the index, not a sort on top of it.
    assert "Order By" in plan, f"vector ordering was not served by the index:\n{plan}"


@pytest.mark.parametrize(
    ("where", "description"),
    [
        ("price_usd < 200", "float range"),
        ("accommodates >= 2", "int range"),
        ("has_wifi", "bool equality"),
        ("neighbourhood_id = 5", "int equality"),
        ("price_usd BETWEEN 50 AND 150", "BETWEEN"),
        ("rating >= 4.0 AND price_usd < 300", "conjunction of two ranges"),
    ],
)
def test_each_supported_predicate_shape_pushes_down(
    conn: psycopg.Connection[tuple[object, ...]],
    fixture_table: str,
    where: str,
    description: str,
) -> None:
    """Brindle pushes down =, <, <=, >, >=, BETWEEN joined by AND."""
    conn.execute("SET brindle.ef_search = 100")
    conn.execute("SET enable_seqscan = off")
    plan = _plan(
        conn,
        f"""
        SELECT id FROM {fixture_table} WHERE {where}
        ORDER BY embedding <=> (SELECT embedding FROM {fixture_table} WHERE id = 1)
        LIMIT 10
        """,
    )
    assert "Index Cond" in plan, f"{description} did not push down:\n{plan}"


def test_returned_rows_actually_satisfy_the_predicate(
    conn: psycopg.Connection[tuple[object, ...]], fixture_table: str
) -> None:
    """A row that violates a pushed filter is never acceptable."""
    conn.execute("SET brindle.ef_search = 100")
    rows = conn.execute(f"""
        SELECT price_usd, accommodates, has_wifi
        FROM {fixture_table}
        WHERE price_usd < 200 AND accommodates >= 2 AND has_wifi
        ORDER BY embedding <=> (SELECT embedding FROM {fixture_table} WHERE id = 1)
        LIMIT 10
    """).fetchall()

    assert rows, "filtered search returned nothing on a predicate with many matches"
    for price, accommodates, wifi in rows:
        assert price is not None and price < 200
        assert accommodates >= 2
        assert wifi is True


def test_a_ranked_scan_returns_at_most_ef_search_rows(
    conn: psycopg.Connection[tuple[object, ...]], fixture_table: str
) -> None:
    """LIMIT above ef_search silently under-returns rather than erroring.

    This is the trap scout.config guards against: ask for 100 candidates with
    ef_search at 10 and you get 10, which the Check node would misread as a
    genuinely thin result and start relaxing filters to fix.
    """
    conn.execute("SET brindle.ef_search = 10")
    rows = conn.execute(f"""
        SELECT id FROM {fixture_table}
        ORDER BY embedding <=> (SELECT embedding FROM {fixture_table} WHERE id = 1)
        LIMIT 100
    """).fetchall()
    assert len(rows) <= 10, "ef_search did not bound the ranked scan as documented"
    conn.execute("SET brindle.ef_search = 100")


def test_null_attributes_satisfy_no_comparison(
    conn: psycopg.Connection[tuple[object, ...]], fixture_table: str
) -> None:
    """`rating >= 4.5` silently drops every unrated listing.

    Scout's Parse and Check nodes have to know this, and the Answer node has to
    say so — otherwise a quality filter quietly excludes new listings and the
    user is never told.
    """
    unrated, total = conn.execute(f"""
        SELECT count(*) FILTER (WHERE rating IS NULL), count(*) FROM {fixture_table}
    """).fetchone()  # type: ignore[misc]
    assert unrated > 0, "fixture should contain unrated rows"

    matching = conn.execute(f"""
        SELECT count(*) FROM {fixture_table} WHERE rating >= 0.0
    """).fetchone()
    assert matching is not None
    assert matching[0] == total - unrated, "NULL ratings should satisfy no comparison"
