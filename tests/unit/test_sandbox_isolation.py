"""Every database-backed test reaches its database through settings.

A worktree sandbox exists so parallel tasks cannot collide: its own branch, its
own virtualenv, and its own Postgres on its own port, written into that
worktree's `.env` by `scripts/worktree.sh`.

`os.environ` does not read `.env`. So a test module that resolves its DSN with
`os.environ.get("SCOUT_DATABASE_URL", "...localhost:5433...")` ignores the
sandbox entirely and reaches for whatever is on port 5433 -- a sibling's
database, or the main clone's. The pushdown module did exactly that, and the
symptom was nothing at all: its fixtures `DROP` and `CREATE` a table, so with
something on 5433 it wrote into another database, and with nothing there it
skipped all eleven checks of the project's central premise while the sandbox's
own Postgres sat healthy and unused.

`get_settings()` reads `.env`, and honours `SCOUT_DATABASE_URL` from the
environment too, which is how CI passes it. So it is correct in both places,
and the hardcoded default is correct in neither.
"""

from __future__ import annotations

import ast
from pathlib import Path

INTEGRATION_TESTS = Path(__file__).resolve().parents[1] / "integration"

#: Reading settings through `os.environ` is what bypasses `.env`. A module that
#: needs a raw environment variable for something other than the database is
#: welcome to it, but nothing does today, so the simple rule is the enforceable
#: one: integration tests do not touch `os.environ` at all.
ENVIRONMENT_READERS = frozenset({"environ", "getenv"})

#: A connection string written into a test is the other way round settings.
DSN_SCHEMES = ("postgresql://", "postgres://")


def modules() -> list[Path]:
    """Every module under tests/integration, conftest included.

    A fixture moved into a conftest is still a database-backed test's DSN, and
    pytest loads it the same way.
    """
    return sorted(path for path in INTEGRATION_TESTS.rglob("*.py") if path.name != "__init__.py")


def integration_test_modules() -> list[Path]:
    return [path for path in modules() if path.name.startswith("test_")]


def tree_of(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def bypasses_settings(path: Path) -> list[str]:
    """Every way `path` reaches a database without going through settings.

    Read off the syntax tree, so a comment explaining the rule -- the pushdown
    module has one -- is not mistaken for breaking it.
    """
    found = []
    for node in ast.walk(tree_of(path)):
        if isinstance(node, ast.Attribute) and node.attr in ENVIRONMENT_READERS:
            found.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            found.extend(
                f"line {node.lineno}: from os import {alias.name}"
                for alias in node.names
                if alias.name in ENVIRONMENT_READERS
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(scheme in node.value for scheme in DSN_SCHEMES)
        ):
            found.append(f"line {node.lineno}: a literal connection string")
    return found


def asks_settings(path: Path) -> bool:
    """Whether the module's code, not merely its prose, refers to `get_settings`."""
    return any(
        (isinstance(node, ast.Name) and node.id == "get_settings")
        or (isinstance(node, ast.Attribute) and node.attr == "get_settings")
        or (isinstance(node, ast.ImportFrom) and any(a.name == "get_settings" for a in node.names))
        for node in ast.walk(tree_of(path))
    )


def test_no_integration_test_resolves_its_database_around_settings() -> None:
    offenders = {path.name: found for path in modules() if (found := bypasses_settings(path))}
    assert offenders == {}, (
        f"these integration modules bypass .env, so in a worktree sandbox they talk to "
        f"whatever is on the default port instead of the sandbox's own Postgres: {offenders}. "
        f"Use get_settings().database_url."
    )


def test_every_integration_test_goes_through_settings() -> None:
    """The positive half: each test module's code actually asks settings for the DSN.

    Without this, deleting the connection entirely would satisfy the test above.
    """
    missing = [path.name for path in integration_test_modules() if not asks_settings(path)]
    assert missing == [], f"these integration tests never ask settings for a database: {missing}"


def test_there_are_integration_modules_to_check() -> None:
    """Otherwise both tests above pass by scanning an empty directory."""
    assert len(integration_test_modules()) > 1
