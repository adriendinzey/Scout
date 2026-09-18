"""The migration runner's decisions, tested without a database.

Which migrations are pending is pure logic over two inputs -- what is in the
repository and what the database says it has applied -- so the interesting
cases (a file edited after it was applied, a version recorded with no file left,
history that forked) are cheap to cover exhaustively here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scout.data.migrate import (
    Migration,
    MigrationError,
    default_migrations_dir,
    load_migrations,
    parse_migration_filename,
    pending_migrations,
)


def make_migration(version: str, name: str = "thing", sql: str = "SELECT 1") -> Migration:
    return Migration(
        version=version,
        name=name,
        sql=sql,
        checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
    )


class TestParseMigrationFilename:
    def test_splits_version_from_name(self):
        assert parse_migration_filename("0001_initial.sql") == ("0001", "initial")
        assert parse_migration_filename("0012_add_amenity_columns.sql") == (
            "0012",
            "add_amenity_columns",
        )

    @pytest.mark.parametrize(
        "filename",
        [
            "1_initial.sql",  # not zero-padded: lexical order stops being numeric order
            "0001-initial.sql",
            "0001_Initial.sql",
            "initial.sql",
            "0001_initial.txt",
            "0001_initial",
        ],
    )
    def test_rejects_anything_else(self, filename):
        with pytest.raises(MigrationError, match="not a migration filename"):
            parse_migration_filename(filename)


class TestLoadMigrations:
    def test_reads_in_version_order_and_fingerprints(self, tmp_path: Path):
        (tmp_path / "0002_second.sql").write_text("SELECT 2")
        (tmp_path / "0001_first.sql").write_text("SELECT 1")

        migrations = load_migrations(tmp_path)

        assert [m.version for m in migrations] == ["0001", "0002"]
        assert [m.name for m in migrations] == ["first", "second"]
        assert migrations[0].sql == "SELECT 1"
        assert migrations[0].checksum != migrations[1].checksum

    def test_ignores_files_that_are_not_sql(self, tmp_path: Path):
        (tmp_path / "0001_first.sql").write_text("SELECT 1")
        (tmp_path / "README.md").write_text("notes")

        assert [m.filename for m in load_migrations(tmp_path)] == ["0001_first.sql"]

    def test_rejects_two_files_with_the_same_version(self, tmp_path: Path):
        (tmp_path / "0001_first.sql").write_text("SELECT 1")
        (tmp_path / "0001_also_first.sql").write_text("SELECT 2")

        with pytest.raises(MigrationError, match="share version 0001"):
            load_migrations(tmp_path)

    def test_rejects_a_badly_named_migration(self, tmp_path: Path):
        (tmp_path / "add_stuff.sql").write_text("SELECT 1")

        with pytest.raises(MigrationError, match="not a migration filename"):
            load_migrations(tmp_path)

    def test_missing_directory_is_an_error_not_an_empty_plan(self, tmp_path: Path):
        with pytest.raises(MigrationError, match="no migrations directory"):
            load_migrations(tmp_path / "nope")

    def test_empty_directory_is_an_error_not_an_empty_plan(self, tmp_path: Path):
        with pytest.raises(MigrationError, match=r"no \.sql migrations found"):
            load_migrations(tmp_path)


class TestPendingMigrations:
    def test_everything_is_pending_on_an_empty_database(self):
        available = [make_migration("0001"), make_migration("0002")]

        assert pending_migrations(available, {}) == available

    def test_nothing_is_pending_when_all_are_recorded(self):
        available = [make_migration("0001"), make_migration("0002")]
        applied = {m.version: m.checksum for m in available}

        assert pending_migrations(available, applied) == []

    def test_only_the_unrecorded_ones_are_pending(self):
        first, second = make_migration("0001"), make_migration("0002")

        assert pending_migrations([first, second], {first.version: first.checksum}) == [second]

    def test_an_applied_migration_that_was_edited_is_an_error(self):
        migration = make_migration("0001", sql="SELECT 1")
        applied = {"0001": make_migration("0001", sql="SELECT 999").checksum}

        with pytest.raises(MigrationError, match="changed after it was applied"):
            pending_migrations([migration], applied)

    def test_a_recorded_migration_with_no_file_is_an_error(self):
        with pytest.raises(MigrationError, match="no longer exist in the repository"):
            pending_migrations([make_migration("0001")], {"0001": "x", "0007": "y"})

    def test_a_pending_migration_older_than_an_applied_one_is_an_error(self):
        """A file dropped in behind a migration that already ran forks history."""
        first, second = make_migration("0001"), make_migration("0002")
        applied = {second.version: second.checksum}

        with pytest.raises(MigrationError, match="already in place"):
            pending_migrations([first, second], applied)


def test_the_repositorys_own_migrations_are_well_formed():
    """Catches a misnamed or duplicated migration file without a database."""
    migrations = load_migrations(default_migrations_dir())

    assert migrations[0].filename == "0001_initial.sql"
    assert pending_migrations(migrations, {}) == migrations
