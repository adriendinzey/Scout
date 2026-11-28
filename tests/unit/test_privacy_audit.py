"""The two hard constraints that fail silently, asserted rather than trusted.

No personal data stored or displayed, and no Inside Airbnb data in the repo.
Both break without a symptom: a stray `host_name` column returns perfectly
correct rows, and a committed CSV passes every other test in the suite. By the
time either is noticed it is in the git history, which is the one place a
mistake cannot be taken back from.

So this module is the audit `docs/DATA.md` section 6 promises, and it is a *unit*
test on purpose: the schema check that already exists needs Postgres, while
everything here reads text -- the migrations, the loader's own statements,
`docs/DATA.md`, `git check-ignore` -- so it runs in well under a second with no
database, in the fast job every PR waits on first.

Two rules the scans follow, because both have an obvious wrong version:

**A comment is not a column.** The migrations and the loader spell out
`host_id, host_name, ...` in prose, in order to explain that they are dropped. A
scan that cannot tell that apart from `CREATE TABLE` fires on the very code that
proves the point, and gets deleted. SQL comments and single-quoted literals are
stripped; Python docstrings and the one list that names the columns are skipped.

**A scan that finds nothing may just be broken.** Each scanner here is pointed
at a planted violation as well as at the repo, because a scanner that cannot
fail still reads like coverage.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from scout.config import get_settings

# Private on purpose, and imported anyway: these two tuples *are* the set of
# columns the loader writes, and an audit that retyped them would be asserting
# against its own copy. Renaming one breaks this import, which is the intended
# failure -- quieter than a scan that silently stops matching anything.
from scout.data.load import _LISTING_COLUMNS, _REVIEW_COLUMNS
from scout.data.parsers import DROPPED_PERSONAL_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "scout"
MIGRATIONS = REPO_ROOT / "migrations"
DATA_DOC = REPO_ROOT / "docs" / "DATA.md"
PR_TEMPLATE = REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
BUG_TEMPLATE = REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"

#: The one name in the `host_*` family that is kept. It describes the listing's
#: service level, not a person, and it is a filter guests actually use
#: (docs/DATA.md section 4). Named here so that tightening the pattern below
#: into "drop anything starting with host_" fails instead of silently deleting
#: a working filter.
KEPT_HOST_COLUMN = "host_is_superhost"

#: Where `DROPPED_PERSONAL_COLUMNS` may be written out in full: the assignment
#: that exists in order to refuse those columns. Everywhere else in `src/`, one
#: of these names in a string literal is a column being written.
DECLARATION_SITES = frozenset({"DROPPED_PERSONAL_COLUMNS"})

#: Corpus-shaped file endings. docs/DATA.md section 3 forbids the CSVs, anything
#: derived in bulk from them, and the embeddings -- a vector per listing is
#: derived data, and redistributing it is still redistributing the corpus. The
#: compressed forms are spelled out because the source ships `.csv.gz`, whose
#: `Path.suffix` is only `.gz`.
CORPUS_SUFFIXES = frozenset(
    {
        ".csv",
        ".csv.gz",
        ".tsv",
        ".tsv.gz",
        ".parquet",
        ".dump",
        ".sql.gz",
        ".npy",
        ".npz",
        ".pkl",
        # An embedding saved by torch, and the other ways a bulk file is
        # compressed or dumped.
        ".pt",
        ".safetensors",
        ".csv.bz2",
        ".csv.xz",
        ".csv.zip",
        ".tsv.bz2",
        ".tsv.xz",
        ".sql.bz2",
        ".sql.xz",
        ".backup",
        # Archives and single-file databases: any of them can carry the corpus
        # whole, and Scout tracks none for a legitimate reason.
        ".zip",
        ".tar",
        ".sqlite",
        ".sqlite3",
    }
)

# Literals and comments in one pass, so whichever starts first wins. Stripped in
# two passes, comments first, a ` -- ` inside a quoted literal (migration 0002
# has them) is taken for a comment and eats the closing quote, and the literal
# pattern then swallows the real DDL up to the next quote -- a column after it
# goes unseen. Postgres escapes a quote by doubling it, so a literal runs to the
# first unpaired quote. Dollar-quoting is not used anywhere in Scout's SQL; if it
# ever is, the migration scan has to learn about it.
_SQL_NOT_EXECUTABLE = re.compile(r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/", re.DOTALL)

_HAS_DOCSTRING = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _personal_column_pattern(names: frozenset[str]) -> re.Pattern[str]:
    """Match any of `names` as a whole word.

    Whole-word, so `host_id` does not also match a column legitimately named
    `source_host_identifier`, and so `reviewer_id` is found inside
    `COPY t (listing_id, reviewer_id)` where a substring search on the bare name
    would be swamped by `_id`.
    """
    return re.compile(r"\b(?:" + "|".join(re.escape(name) for name in sorted(names)) + r")\b")


PERSONAL_COLUMNS = _personal_column_pattern(DROPPED_PERSONAL_COLUMNS)
# reviewer* is never legitimate whatever follows it -- reviewer_name, a bare
# reviewer, reviewerid -- so the family is caught rather than the two spellings
# the current snapshot happens to ship, matching the integration check's
# `LIKE 'reviewer%'`. host_* cannot be swept the same way: host_is_superhost is
# kept.
ANY_REVIEWER_COLUMN = re.compile(r"\breviewer\w*\b")


def executable_sql(sql: str) -> str:
    """`sql` with the parts that cannot create a column removed.

    Comments go, and so do single-quoted literals: the `COMMENT ON TABLE`
    statements in the first migration list the dropped columns by name, in a
    quoted string, precisely to record that they are absent. A double-quoted
    identifier is left alone -- `CREATE TABLE x ("host_name" text)` is a real
    column and has to be caught.
    """
    return _SQL_NOT_EXECUTABLE.sub(" ", sql)


def migration_sql() -> list[tuple[str, str]]:
    """Every migration, as (file name, the part that can create a column)."""
    return [
        (path.name, executable_sql(path.read_text(encoding="utf-8")))
        for path in sorted(MIGRATIONS.glob("*.sql"))
    ]


def string_literals(path: Path) -> Iterator[tuple[int, str]]:
    """Every string constant in a module, minus docstrings and the dropped list.

    An AST walk rather than a grep, for the same reason the LLM-boundary test
    walks the package: it polices every file including ones not yet written, and
    it can tell a column name in a `COPY` statement from the same name in a
    paragraph explaining why it is dropped.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, _HAS_DOCSTRING):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skipped.add(id(first.value))
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in DECLARATION_SITES
            for target in node.targets
        ):
            skipped.update(id(child) for child in ast.walk(node))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skipped
        ):
            yield node.lineno, node.value


def git(*args: str) -> subprocess.CompletedProcess[str]:
    """Ask git itself. The claim in `docs/DATA.md` is about git's own rules.

    Reimplementing `.gitignore` matching would be a second, worse answer to a
    question git answers exactly -- and the pattern that matters most here,
    `/runs/`, is the kind a naive matcher gets wrong.
    """
    executable = shutil.which("git")
    if executable is None:  # pragma: no cover - git is present wherever CI runs
        pytest.skip("git is not installed")
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        (executable, "-C", str(REPO_ROOT), *args),
        capture_output=True,
        text=True,
        check=False,
    )


def tracked_files(*paths: str) -> list[str]:
    """The files git tracks under `paths`, or every tracked file if none given.

    A failed `git ls-files` prints nothing on stdout, which reads exactly like
    "git tracks no corpus data" -- the same trap the agent's tools are written
    to avoid, and it would turn this audit into a test that cannot fail. So a
    non-zero exit is an error here, not an empty result.

    `-z` because a path with a space in it would otherwise split into two names
    that match nothing.
    """
    result = git("ls-files", "-z", "--", *paths)
    assert result.returncode == 0, f"git ls-files failed: {result.stderr.strip()}"
    return [name for name in result.stdout.split("\0") if name]


def require_work_tree() -> None:
    """Skip where there is no git checkout at all; fail where git cannot read one.

    An installed sdist has no `.git`, and CI checks the repository out, so the
    gitignore claims are still enforced where it matters. But a checkout git
    refuses to read -- a broken `GIT_DIR`, or the safe.directory check in a
    container -- is a failure, not an absence: skipping it would turn this whole
    class green without asking git a single question.
    """
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("no .git here, so there is no checkout to interrogate")
    result = git("rev-parse", "--is-inside-work-tree")
    assert result.returncode == 0, (
        f"there is a .git but git cannot read it: {result.stderr.strip()}"
    )


class TestTheScannersCanFail:
    """A scanner that finds nothing may be broken rather than reassuring.

    These are the planted violations. If one of them stops firing, every other
    test in this module has quietly become decorative.
    """

    def test_the_sql_scan_catches_a_personal_column(self) -> None:
        planted = "CREATE TABLE listings (id int4, host_name text)"
        assert PERSONAL_COLUMNS.search(executable_sql(planted))

    def test_the_sql_scan_catches_a_quoted_identifier(self) -> None:
        """A double-quoted identifier is a column, not prose."""
        planted = 'CREATE TABLE listings (id int4, "host_name" text)'
        assert PERSONAL_COLUMNS.search(executable_sql(planted))

    def test_the_sql_scan_ignores_a_column_named_in_a_comment(self) -> None:
        planted = "-- host_name is dropped at parse time\nCREATE TABLE listings (id int4)"
        assert not PERSONAL_COLUMNS.search(executable_sql(planted))

    def test_the_sql_scan_ignores_a_column_named_in_a_quoted_string(self) -> None:
        planted = "COMMENT ON TABLE listings IS 'Carries no host_id and no host_name.'"
        assert not PERSONAL_COLUMNS.search(executable_sql(planted))

    def test_the_sql_scan_sees_a_column_after_a_comment_marker_inside_a_literal(self) -> None:
        """Migration 0002 puts ` -- ` inside a COMMENT literal; what follows is still SQL."""
        planted = (
            "COMMENT ON TABLE reviews IS 'Kept for citation -- never searched.';\n"
            "ALTER TABLE reviews ADD COLUMN reviewer_name text;\n"
            "COMMENT ON COLUMN reviews.comments IS 'Scrubbed before display.';"
        )
        assert PERSONAL_COLUMNS.search(executable_sql(planted))
        assert ANY_REVIEWER_COLUMN.search(executable_sql(planted))

    def test_the_sql_scan_sees_a_column_after_a_quote_inside_a_comment(self) -> None:
        """The other order: an apostrophe in a comment opens no literal."""
        planted = (
            "-- the host's own name is never stored\n"
            "CREATE TABLE listings (id int4, host_name text);\n"
            "COMMENT ON TABLE listings IS 'one row per listing';"
        )
        assert PERSONAL_COLUMNS.search(executable_sql(planted))

    def test_the_sql_scan_still_sees_the_columns_that_are_kept(self) -> None:
        """Stripping comments and literals must not strip the statement too."""
        initial = dict(migration_sql())["0001_initial.sql"]
        assert KEPT_HOST_COLUMN in initial
        assert "CREATE TABLE listings" in initial

    def test_the_reviewer_scan_catches_a_suffix_nobody_thought_of(self) -> None:
        assert ANY_REVIEWER_COLUMN.search("CREATE TABLE reviews (reviewer_handle text)")

    def test_the_python_scan_catches_a_planted_column(self, tmp_path: Path) -> None:
        planted = tmp_path / "leaky.py"
        planted.write_text(
            '"""A docstring may discuss host_name freely."""\n'
            'COLUMNS = ("listing_id", "host_name")\n',
            encoding="utf-8",
        )
        found = {text for _, text in string_literals(planted) if PERSONAL_COLUMNS.search(text)}
        assert found == {"host_name"}

    def test_the_python_scan_catches_a_reviewer_column_nobody_listed(self, tmp_path: Path) -> None:
        planted = tmp_path / "leaky.py"
        planted.write_text(
            'SQL = "INSERT INTO reviews (listing_id, reviewer_handle) VALUES (%s, %s)"\n',
            encoding="utf-8",
        )
        found = [text for _, text in string_literals(planted) if ANY_REVIEWER_COLUMN.search(text)]
        assert len(found) == 1

    def test_the_python_scan_skips_docstrings_and_the_dropped_list(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.py"
        clean.write_text(
            '"""host_id and reviewer_name are dropped while parsing."""\n'
            'DROPPED_PERSONAL_COLUMNS = frozenset({"host_id", "reviewer_name"})\n'
            "\n"
            "def f() -> str:\n"
            '    """Returns nothing about host_name."""\n'
            '    return "SELECT name FROM listings"\n',
            encoding="utf-8",
        )
        assert [text for _, text in string_literals(clean) if PERSONAL_COLUMNS.search(text)] == []


class TestNoPersonalColumnExistsInTheSchema:
    """The schema, asserted without a database so it is asserted on every PR.

    The integration suite checks `information_schema` once the migrations have
    been applied, which is the stronger check and the slower one. This is the
    same guarantee read off the files, and it is the one that needs no
    database.
    """

    def offenders(self, pattern: re.Pattern[str]) -> list[str]:
        return [
            f"{name}: {match.group(0)}"
            for name, body in migration_sql()
            for match in pattern.finditer(body)
        ]

    def test_the_migrations_create_no_personal_column(self) -> None:
        found = self.offenders(PERSONAL_COLUMNS)
        assert found == [], f"a personal column reached a migration: {found}"

    def test_the_migrations_create_no_reviewer_column_of_any_kind(self) -> None:
        found = self.offenders(ANY_REVIEWER_COLUMN)
        assert found == [], f"reviewer identity reached a migration: {found}"

    def test_there_are_migrations_to_scan(self) -> None:
        """Otherwise both tests above pass by scanning nothing."""
        assert migration_sql()


class TestNoPersonalColumnExistsAnywhereInTheSource:
    """Every string Scout's own code can put into a statement.

    The parse-time check in the parser tests is the load-bearing one -- a field
    that does not exist cannot be written. This is the backstop for the other
    direction: SQL that names a column the typed rows do not have. It fails at
    the first `COPY`, but only if someone runs the load, and only if they are
    watching.
    """

    def test_no_string_literal_in_the_package_names_a_personal_column(self) -> None:
        offenders = [
            f"{path.relative_to(SRC)}:{line}: {match.group(0)}"
            for path in sorted(SRC.rglob("*.py"))
            for line, text in string_literals(path)
            for pattern in (PERSONAL_COLUMNS, ANY_REVIEWER_COLUMN)
            for match in pattern.finditer(text)
        ]
        assert offenders == [], f"a personal column name reached the source: {offenders}"

    def test_there_are_modules_to_scan(self) -> None:
        """Otherwise the test above passes by walking an empty directory."""
        assert len(list(SRC.rglob("*.py"))) > 1

    def test_only_the_parsers_declare_the_dropped_list(self) -> None:
        """The scan's one blind spot, closed.

        The names in a `DROPPED_PERSONAL_COLUMNS = ...` assignment are skipped,
        because that list exists in order to spell them out. A second module
        assigning that name would get the same exemption and could hide a real
        column inside it, so there is exactly one such assignment and this says
        where.
        """
        declaring = sorted(
            str(path.relative_to(SRC))
            for path in SRC.rglob("*.py")
            if any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in DECLARATION_SITES
                    for target in node.targets
                )
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            )
        )
        assert declaring == ["data/parsers.py"]


class TestTheLoaderWritesNoPersonalColumn:
    """The write path, stated as a guarantee rather than read off the text.

    The scan above covers the loader's SQL, because that SQL is string literals
    in `src/`. This is the same fact asserted through the loader's own column
    lists, so renaming a constant fails the import here instead of leaving a
    scan that quietly matches nothing.
    """

    def test_no_column_the_loader_copies_is_a_personal_one(self) -> None:
        written = frozenset(_LISTING_COLUMNS) | frozenset(_REVIEW_COLUMNS)
        assert written & DROPPED_PERSONAL_COLUMNS == frozenset()
        assert [column for column in sorted(written) if ANY_REVIEWER_COLUMN.fullmatch(column)] == []

    def test_host_is_superhost_is_written_because_it_describes_the_listing(self) -> None:
        """The deliberate exception, asserted so it is not swept up by mistake."""
        assert KEPT_HOST_COLUMN in _LISTING_COLUMNS
        assert KEPT_HOST_COLUMN not in DROPPED_PERSONAL_COLUMNS
        assert not PERSONAL_COLUMNS.fullmatch(KEPT_HOST_COLUMN)


class TestTheDocumentedListAndTheCodeAgree:
    """One source of truth, because a hand-kept copy had already drifted.

    `docs/DATA.md` section 4 is the binding list; `DROPPED_PERSONAL_COLUMNS` is
    what the code enforces. A column added to the prose and not the code is a
    promise nothing keeps, and the reverse is a dropped column nobody can
    explain.
    """

    def documented_columns(self) -> frozenset[str]:
        """The backticked names in the first column of the section-4 table.

        Only the table rows, and only their first cell: the prose around the
        table backticks other things -- `INSERT`, `host_is_superhost` -- and a
        looser pattern picks those up and compares the wrong two sets.
        """
        doc = DATA_DOC.read_text(encoding="utf-8")
        section = doc.split("## 4. Personal data: dropped at load time", 1)
        assert len(section) == 2, "docs/DATA.md section 4 has been renamed"
        after_table = section[1].split("`host_is_superhost` **is** kept", 1)
        assert len(after_table) == 2, (
            "docs/DATA.md section 4 no longer says `host_is_superhost` **is** kept, "
            "so the end of the dropped-fields table cannot be found"
        )
        table = after_table[0]

        documented: set[str] = set()
        for line in table.splitlines():
            if not line.startswith("|") or line.startswith("|--"):
                continue
            source_column = line.split("|")[1]
            documented.update(re.findall(r"`(\w+)`", source_column))
        return frozenset(documented)

    def test_the_section_four_table_names_exactly_what_the_code_drops(self) -> None:
        assert self.documented_columns() == DROPPED_PERSONAL_COLUMNS

    def test_the_table_was_actually_parsed(self) -> None:
        """A parser that finds no rows would make the comparison meaningless."""
        assert len(self.documented_columns()) > 5


def trace_dir_in_repo() -> Path | None:
    """`SCOUT_TRACE_DIR` relative to the repository, or None when it lies outside.

    Absolute is not the same as outside: `/home/me/scout/runs` is inside a
    checkout at `/home/me/scout`, and a trace there can be committed.
    """
    trace_dir = get_settings().trace_dir
    if not trace_dir.is_absolute():
        return trace_dir
    root = REPO_ROOT.resolve()
    resolved = trace_dir.resolve()
    return resolved.relative_to(root) if resolved.is_relative_to(root) else None


class TestNoCorpusDataIsInTheRepo:
    """Hard constraint 1. Gitignored is the mechanism; untracked is the claim."""

    @pytest.fixture(autouse=True)
    def _work_tree(self) -> None:
        require_work_tree()

    def test_the_data_directories_are_ignored(self) -> None:
        for path in ("data/london/listings.csv.gz", "eval/labels/relevance.json"):
            assert git("check-ignore", "-q", path).returncode == 0, f"{path} is not ignored"

    def test_the_trace_directory_is_ignored(self) -> None:
        """A step record quotes listing and review text, so a trace stays local."""
        trace_dir = trace_dir_in_repo()
        if trace_dir is None:
            pytest.skip(f"SCOUT_TRACE_DIR is {get_settings().trace_dir}: outside the repo")
        probe = str(trace_dir / "run-0123.jsonl")
        assert git("check-ignore", "-q", probe).returncode == 0, (
            f"SCOUT_TRACE_DIR is {trace_dir}, which .gitignore does not cover"
        )

    def test_nothing_under_those_directories_is_tracked(self) -> None:
        """Ignoring a path does not untrack a file committed before the rule."""
        trace_dir = trace_dir_in_repo()
        paths = ["data", "eval/labels"]
        if trace_dir is not None:
            paths.append(str(trace_dir))
        tracked = tracked_files(*paths)
        assert tracked == [], f"corpus or trace files are tracked: {tracked}"

    def test_no_tracked_file_is_corpus_shaped(self) -> None:
        """Catches the copy committed somewhere the ignore rules do not reach."""
        offenders = [
            name for name in tracked_files() if name.lower().endswith(tuple(CORPUS_SUFFIXES))
        ]
        assert offenders == [], f"corpus-shaped files are tracked: {offenders}"

    def test_the_tracked_file_list_was_actually_read(self) -> None:
        """Otherwise the two tests above pass against an empty list."""
        assert "pyproject.toml" in tracked_files()


class TestThePrivacyAffordancesSurviveAnEdit:
    """docs/DATA.md section 6 counts these as enforcement, so they are asserted.

    Both are prose in a template nothing else reads, which is exactly the kind
    of thing a tidy-up deletes.
    """

    def test_the_pull_request_template_asks_about_personal_data(self) -> None:
        template = PR_TEMPLATE.read_text(encoding="utf-8")
        assert "No personal data stored or displayed" in template
        assert "No Inside Airbnb data" in template

    def test_the_bug_report_template_asks_reporters_not_to_paste_names(self) -> None:
        assert "I have not pasted personal data" in BUG_TEMPLATE.read_text(encoding="utf-8")


def test_the_name_scrubber_does_not_exist_yet() -> None:
    """A tripwire, not a wish.

    The other half of this audit is the name scrubber, tested against
    real-shaped review text. It has not been written: it belongs to the
    document-template step, which measures it against loaded review text and
    publishes its error rates, and a second scrubber here would break the
    one-function-one-call-path rule `docs/DATA.md` section 4 sets.

    Deferring that is correct. Deferring it silently is how `docs/DATA.md`
    section 6 came to claim a test that did not exist. So this fails the moment
    the scrubber lands, and whoever lands it moves the scrubber claim into the
    enforced half of section 6 and replaces this with the real test.
    """
    # Whether the module exists, not whether it imports: a template module that
    # imports the embedder -- an extra CI does not install -- would raise
    # ModuleNotFoundError for its dependency and leave this tripwire disarmed.
    assert importlib.util.find_spec("scout.data.document") is None, (
        "scout/data/document.py exists: replace this tripwire with the scrubber's real "
        "test, and move the scrubber claim into the enforced half of docs/DATA.md section 6"
    )
