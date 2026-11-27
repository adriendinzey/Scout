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


def modules() -> list[Path]:
    return sorted(
        path for path in INTEGRATION_TESTS.rglob("test_*.py") if path.name != "__init__.py"
    )


def reads_the_environment_directly(path: Path) -> bool:
    """True if the module mentions `os.environ` / `os.getenv` anywhere."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Attribute) and node.attr in ENVIRONMENT_READERS
        for node in ast.walk(tree)
    )


def test_no_integration_test_resolves_its_database_from_the_environment() -> None:
    offenders = [path.name for path in modules() if reads_the_environment_directly(path)]
    assert offenders == [], (
        f"these integration tests bypass .env, so in a worktree sandbox they talk to "
        f"whatever is on the default port instead of the sandbox's own Postgres: {offenders}. "
        f"Use get_settings().database_url."
    )


def test_every_integration_test_goes_through_settings() -> None:
    """The positive half: each module actually asks settings for the DSN.

    Without this, deleting the connection entirely would satisfy the test above.
    """
    missing = [
        path.name for path in modules() if "get_settings" not in path.read_text(encoding="utf-8")
    ]
    assert missing == [], f"these integration tests never ask settings for a database: {missing}"


def test_there_are_integration_modules_to_check() -> None:
    """Otherwise both tests above pass by scanning an empty directory."""
    assert len(modules()) > 1
