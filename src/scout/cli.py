"""Scout's command-line interface.

`scout ask` is the product; `scout data` builds what it searches; `scout doctor`
answers "is my environment actually working?" without needing either.
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

import psycopg
import typer

from scout import __version__
from scout.config import Settings, get_settings
from scout.data.amenities import (
    AmenityError,
    backfill_amenity_columns,
    read_amenity_report,
    render_report,
)
from scout.data.load import LoadError, load_data
from scout.data.migrate import MigrationError, run_migrations

app = typer.Typer(
    name="scout",
    help="Agentic search over short-term rental listings.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(name="data", help="Load, embed, and index the listing data.")
app.add_typer(data_app)


class NotImplementedYetError(typer.Exit):
    """A command whose milestone has not landed yet.

    Exits non-zero with a specific message rather than pretending to succeed —
    a stub that prints nothing and returns 0 is how a broken pipeline looks green.
    """

    def __init__(self, what: str, milestone: str) -> None:
        typer.secho(
            f"{what} is not implemented yet (lands in {milestone}).",
            fg=typer.colors.YELLOW,
            err=True,
        )
        super().__init__(code=2)


@app.command()
def version() -> None:
    """Print Scout's version."""
    print(f"scout {__version__}")


@app.command()
def doctor() -> None:
    """Check that the database, extensions, and configuration are usable."""
    settings = get_settings()
    ok = True

    print(f"scout {__version__}")
    print(f"  llm backend    : {settings.llm_backend}")
    print(f"  parse / agent  : {settings.model_parse} / {settings.model_agent}")
    print(f"  answer         : {settings.model_answer}")
    print(f"  embedding      : {settings.embedding_model} ({settings.embedding_dims}d)")
    print(f"  ef_search      : {settings.ef_search} (candidate pool {settings.candidate_pool})")
    print(f"  mode           : {settings.mode}")
    print(
        f"  agent limits   : {settings.max_tool_calls} tool calls, "
        f"{settings.max_searches} searches, ${settings.max_query_cost_usd:.2f}, "
        f"{settings.query_timeout_s:.0f}s"
    )

    if settings.llm_backend == "anthropic" and settings.anthropic_api_key is None:
        typer.secho("  ANTHROPIC_API_KEY is not set", fg=typer.colors.RED, err=True)
        ok = False

    ok = _check_database(settings) and ok

    if not ok:
        raise typer.Exit(code=1)
    typer.secho("environment looks good", fg=typer.colors.GREEN)


def _check_database(settings: Settings) -> bool:
    """Report on the database and the two extensions Scout compares."""
    try:
        with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
            server = conn.execute("SHOW server_version").fetchone()
            extensions = conn.execute(
                "SELECT extname, extversion FROM pg_extension "
                "WHERE extname IN ('brindle', 'vector') ORDER BY extname"
            ).fetchall()
    except psycopg.Error as exc:
        typer.secho(f"  database unreachable: {exc}", fg=typer.colors.RED, err=True)
        typer.secho("  start it with: docker compose up -d", err=True)
        return False

    print(f"  postgres       : {server[0] if server else 'unknown'}")

    found: dict[str, str] = dict(extensions)
    ok = True
    for required in ("brindle", "vector"):
        if required in found:
            print(f"  {required:<15}: {found[required]}")
        else:
            typer.secho(f"  {required:<15}: MISSING", fg=typer.colors.RED, err=True)
            ok = False
    return ok


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="What you are looking for, in plain English.")],
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="agent: Claude chooses which tool to call. fixed: the hardcoded path.",
        ),
    ] = "agent",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    trace: Annotated[
        bool,
        typer.Option("--trace", help="Print the step table: latency, tokens, and cost per step."),
    ] = False,
    no_relax: Annotated[
        bool,
        typer.Option(
            "--no-relax", help="Fixed mode only: search once, never relax. For comparison."
        ),
    ] = False,
) -> None:
    """Search listings with a natural-language request."""
    del query, mode, json_output, trace, no_relax
    raise NotImplementedYetError("scout ask", "M2")


@app.command()
def trace(
    run_id: Annotated[str, typer.Argument(help="The run to replay, as printed by `scout ask`.")],
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Print a stored run: every step, its latency, tokens, and cost."""
    del run_id, json_output
    raise NotImplementedYetError("scout trace", "M3.5")


@app.command()
def runs(
    last: Annotated[int, typer.Option("--last", help="How many recent runs to list.")] = 20,
) -> None:
    """List recent runs with their query, mode, step count, latency, and cost."""
    del last
    raise NotImplementedYetError("scout runs", "M3.5")


@data_app.command("load")
def data_load(
    quiet: Annotated[
        bool, typer.Option("--quiet", help="Print the summary only, without per-stage progress.")
    ] = False,
) -> None:
    """Load the Inside Airbnb CSVs into PostgreSQL, dropping personal fields."""
    settings = get_settings()
    # Progress goes to stderr, so the summary on stdout stays pipeable.
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    try:
        report = load_data(settings)
    except LoadError as exc:
        typer.secho(f"load failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    print(f"loaded the {report.snapshot_date} {report.city} snapshot in {report.elapsed_s:.0f}s")
    print(f"  listings       : {report.listings_loaded:,} of {report.listings_seen:,} rows")
    if report.listings_rejected:
        reasons = ", ".join(f"{column}: {count}" for column, count in report.listing_rejections)
        print(f"  rejected       : {report.listings_rejected:,} ({reasons})")
    if report.listings_duplicated:
        print(f"  duplicates     : {report.listings_duplicated:,} repeated ids, first kept")
    if report.listings_removed:
        print(f"  removed        : {report.listings_removed:,} no longer in the snapshot")
    print(f"  reviews        : {report.reviews_kept:,} kept of {report.reviews_seen:,} rows")
    if report.reviews_without_listing:
        print(f"  orphan reviews : {report.reviews_without_listing:,} for an unknown listing")
    print(
        f"  lookups        : {report.neighbourhoods} neighbourhoods, "
        f"{report.room_types} room types, {report.property_types} property types "
        f"({report.property_types_collapsed} collapsed), {report.amenities:,} amenities"
    )


@data_app.command("amenities")
def data_amenities(
    report: Annotated[
        bool,
        typer.Option(
            "--report",
            help=(
                "Rank the amenity groups by how many listings have them. "
                "The default, and printed after --backfill when both are given."
            ),
        ),
    ] = False,
    backfill: Annotated[
        bool,
        typer.Option(
            "--backfill",
            help="Recompute the indexed amenity booleans from each listing's amenity list.",
        ),
    ] = False,
) -> None:
    """Rank the amenity groups, or derive the indexed boolean columns from them.

    With neither flag, prints the report: that is the read-only answer to "what
    is in this data", and the one worth running by accident.
    """
    settings = get_settings()
    try:
        conn = psycopg.connect(settings.database_url, connect_timeout=5)
    except psycopg.Error as exc:
        typer.secho(f"database unreachable: {exc}", fg=typer.colors.RED, err=True)
        typer.secho("start it with: docker compose up -d", err=True)
        raise typer.Exit(code=1) from exc

    try:
        with conn:
            if backfill:
                # One transaction: half-derived booleans are worse than none,
                # because nothing downstream can tell which half is current.
                with conn.transaction():
                    written = backfill_amenity_columns(conn)
                print(
                    f"derived {len(written.columns)} amenity columns over "
                    f"{written.listings:,} listings "
                    f"({written.listings_changed:,} rows changed)"
                )
                for column in written.columns:
                    listings = written.per_column.get(column, 0)
                    share = listings / written.listings * 100 if written.listings else 0.0
                    print(f"  {column:<26} {listings:>8,}  {share:5.2f}%")
                print(
                    f"  {'amenities.indexed_column':<26} {written.lookup_rows_flagged:>8,} "
                    f"lookup rows ({written.lookup_rows_changed:,} changed)"
                )
            if report or not backfill:
                print(render_report(read_amenity_report(conn)))
    except AmenityError as exc:
        typer.secho(f"amenities failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except psycopg.Error as exc:
        # Everything inside already arrives as an AmenityError, so this is the
        # commit the context manager performs on the way out.
        typer.secho(f"could not commit the amenity columns: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("embed")
def data_embed() -> None:
    """Embed each listing's document text. Resumable."""
    raise NotImplementedYetError("scout data embed", "M1")


@data_app.command("index")
def data_index() -> None:
    """Build the Brindle index and the pgvector baselines, recording build times."""
    raise NotImplementedYetError("scout data index", "M1")


@data_app.command("migrate")
def data_migrate() -> None:
    """Create or update Scout's schema. Safe to run again; a no-op when current."""
    settings = get_settings()
    try:
        applied = run_migrations(settings.database_url)
    except MigrationError as exc:
        typer.secho(f"migration failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if not applied:
        print("schema is up to date; nothing to apply")
        return
    for version in applied:
        print(f"applied {version}")
    typer.secho(f"applied {len(applied)} migration(s)", fg=typer.colors.GREEN)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
