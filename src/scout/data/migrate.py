"""Applying the SQL files that define Scout's schema.

Scout is one application against one schema that it owns, so migrations are
plain numbered `.sql` files applied in order and recorded in a
`schema_migrations` table. There is no ORM and no migration DSL in between: what
a migration does is exactly what its SQL says, and a reviewer reads the file
rather than inferring the DDL from a model definition.

The runner is deliberately strict about history. Re-applying is a no-op, but a
migration whose text changed after it was applied, or one that is recorded in
the database with no file left in the repo, is an error rather than a shrug --
in both cases the database and the repo disagree about what the schema is, and
the failure that follows would otherwise land somewhere far away.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import TupleRow

# Zero-padded so that lexical order is numeric order; the runner relies on it.
_FILENAME = re.compile(r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.sql$")

# Created by the runner rather than by a migration, because the runner has to
# read it to decide whether the first migration has run.
_BOOKKEEPING_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text        PRIMARY KEY,
    name       text        NOT NULL,
    checksum   text        NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""

Connection = psycopg.Connection[TupleRow]


class MigrationError(RuntimeError):
    """A migration could not be read, planned, or applied."""


@dataclass(frozen=True, slots=True)
class Migration:
    """One migration file, read and fingerprinted."""

    version: str
    name: str
    sql: str
    checksum: str

    @property
    def filename(self) -> str:
        return f"{self.version}_{self.name}.sql"


def default_migrations_dir() -> Path:
    """The repository's `migrations/` directory.

    Located relative to the package rather than shipped as package data: Scout
    runs from a source checkout, since its pipeline reads local CSV files and
    its database is the one in Compose beside them.
    """
    return Path(__file__).resolve().parents[3] / "migrations"


def parse_migration_filename(filename: str) -> tuple[str, str]:
    """Split `0001_initial.sql` into `("0001", "initial")`.

    Raises:
        MigrationError: if the name is not `NNNN_lower_snake_case.sql`.
    """
    match = _FILENAME.match(filename)
    if match is None:
        raise MigrationError(
            f"{filename!r} is not a migration filename; expected "
            f"NNNN_lower_snake_case.sql, where the zero-padded number is what "
            f"makes lexical order the order they are applied in"
        )
    return match["version"], match["name"]


def load_migrations(directory: Path) -> list[Migration]:
    """Read every migration in `directory`, in the order they apply.

    Files that are not `.sql` are ignored, so a README can live alongside them.

    Raises:
        MigrationError: if the directory is missing, holds no migrations, names
            one badly, or gives two files the same version number.
    """
    if not directory.is_dir():
        raise MigrationError(f"no migrations directory at {directory}")

    migrations: list[Migration] = []
    seen: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix != ".sql":
            continue
        version, name = parse_migration_filename(path.name)
        if version in seen:
            raise MigrationError(
                f"two migrations share version {version}: {seen[version]} and {path.name}"
            )
        seen[version] = path.name
        sql = path.read_text(encoding="utf-8")
        migrations.append(Migration(version=version, name=name, sql=sql, checksum=_checksum(sql)))

    if not migrations:
        raise MigrationError(f"no .sql migrations found in {directory}")
    return migrations


def pending_migrations(
    available: Sequence[Migration], applied: Mapping[str, str]
) -> list[Migration]:
    """Decide which migrations still have to run. Pure: no database, no files.

    Args:
        available: every migration in the repository, in apply order.
        applied: version -> checksum, as recorded in `schema_migrations`.

    Returns:
        The migrations to apply, in order. Empty when the schema is current.

    Raises:
        MigrationError: if an applied migration's text has changed since, if one
            is recorded with no file left in the repository, or if a pending
            migration sorts before one that is already applied -- each means the
            database and the repository disagree about the schema's history.
    """
    known = {migration.version for migration in available}
    if orphans := sorted(set(applied) - known):
        raise MigrationError(
            f"the database has applied migrations that no longer exist in the "
            f"repository: {', '.join(orphans)}"
        )

    pending: list[Migration] = []
    for migration in available:
        recorded = applied.get(migration.version)
        if recorded is None:
            pending.append(migration)
        elif recorded != migration.checksum:
            raise MigrationError(
                f"{migration.filename} changed after it was applied; the "
                f"database no longer matches it. Write a new migration instead "
                f"of editing an applied one, or rebuild the database from "
                f"scratch."
            )

    if pending and applied:
        latest_applied = max(applied)
        if out_of_order := [m.filename for m in pending if m.version < latest_applied]:
            raise MigrationError(
                f"{', '.join(out_of_order)} would apply after the later "
                f"migration {latest_applied}, which is already in place"
            )
    return pending


def apply_migrations(conn: Connection, directory: Path | None = None) -> list[str]:
    """Apply every pending migration on an existing connection.

    Each migration runs in its own transaction with its bookkeeping row, so a
    failure half way through a run leaves the migrations before it applied and
    recorded, and the failing one not applied at all.

    Returns:
        The versions applied, in order. Empty when the schema was already
        current, which is what makes a second run a no-op.

    Raises:
        MigrationError: if the migrations cannot be read or planned, or if one
            fails to apply.
    """
    migrations = load_migrations(directory if directory is not None else default_migrations_dir())

    try:
        with conn.transaction():
            conn.execute(_BOOKKEEPING_DDL)
        applied = _recorded_checksums(conn)
    except psycopg.Error as exc:
        raise MigrationError(f"could not read the migration history: {exc}") from exc

    pending = pending_migrations(migrations, applied)
    for migration in pending:
        try:
            with conn.transaction():
                # The statement text is a static file in this repository, not
                # anything assembled from input; the bookkeeping row that
                # records it is parameterized.
                conn.execute(migration.sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)",
                    (migration.version, migration.name, migration.checksum),
                )
        except psycopg.Error as exc:
            raise MigrationError(f"{migration.filename} failed to apply: {exc}") from exc

    return [migration.version for migration in pending]


def run_migrations(database_url: str, directory: Path | None = None) -> list[str]:
    """Connect, apply every pending migration, and return the versions applied.

    Raises:
        MigrationError: if the database is unreachable or a migration fails.
    """
    try:
        conn = psycopg.connect(database_url, autocommit=True)
    except psycopg.OperationalError as exc:
        raise MigrationError(f"could not connect to {_describe(database_url)}: {exc}") from exc
    with conn:
        return apply_migrations(conn, directory)


def _checksum(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _recorded_checksums(conn: Connection) -> dict[str, str]:
    rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    return {str(version): str(checksum) for version, checksum in rows}


def _describe(database_url: str) -> str:
    """`host:port/dbname`, so a connection error can name its target.

    The DSN itself carries a password, and this string ends up in error output
    that gets pasted into issues.
    """
    try:
        info = conninfo_to_dict(database_url)
    except psycopg.ProgrammingError:
        return "the configured database"
    host = info.get("host") or "localhost"
    port = info.get("port") or "5432"
    dbname = info.get("dbname") or "?"
    return f"{host}:{port}/{dbname}"
