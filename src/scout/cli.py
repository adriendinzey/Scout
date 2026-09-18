"""Scout's command-line interface.

`scout ask` is the product; `scout data` builds what it searches; `scout doctor`
answers "is my environment actually working?" without needing either.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from scout import __version__
from scout.config import Settings, get_settings
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
    print(f"  parse / check  : {settings.model_parse} / {settings.model_check}")
    print(f"  answer         : {settings.model_answer}")
    print(f"  embedding      : {settings.embedding_model} ({settings.embedding_dims}d)")
    print(f"  ef_search      : {settings.ef_search} (candidate pool {settings.candidate_pool})")

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
        import psycopg
    except ImportError:
        typer.secho("  psycopg is not installed", fg=typer.colors.RED, err=True)
        return False

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
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    no_relax: Annotated[
        bool, typer.Option("--no-relax", help="Disable the Check loop, for comparisons.")
    ] = False,
    relaxer: Annotated[
        str, typer.Option("--relaxer", help="Relaxation strategy: rules or llm.")
    ] = "llm",
) -> None:
    """Search listings with a natural-language request."""
    del query, json_output, no_relax, relaxer
    raise NotImplementedYetError("scout ask", "M2")


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


@data_app.command("load")
def data_load() -> None:
    """Load the Inside Airbnb CSVs into PostgreSQL, dropping personal fields."""
    raise NotImplementedYetError("scout data load", "M1")


@data_app.command("embed")
def data_embed() -> None:
    """Embed each listing's document text. Resumable."""
    raise NotImplementedYetError("scout data embed", "M1")


@data_app.command("index")
def data_index() -> None:
    """Build the Brindle index and the pgvector baselines, recording build times."""
    raise NotImplementedYetError("scout data index", "M1")


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
