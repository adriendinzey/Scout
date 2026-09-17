# Scout — Coding Standards

Conventions for all code in this repository. They are enforced in review.

## Comments & documentation

- **Write self-documenting code first.** Clear names, small focused functions,
  obvious control flow. Reach for a better name before reaching for a comment.
- **Comment only when necessary, and explain _why_, not _what_.** Justified: a
  non-obvious invariant, a constraint imposed by Brindle or the data, a
  deliberate trade-off, a link to a source. Don't narrate code that already
  states what it does.
- **Public functions get docstrings** covering behavior, raised errors, units,
  and edge cases — especially "returns at most N" and "excludes NULLs".

## No internal tracking in production code

Planning lives outside the shipped code and must never leak into it:

- **No task IDs in source or comments** (no `T-021`, `T-0xx`).
- **No references to internal-only files** (`tasks/`, `CLAUDE.md`) in `src/`,
  docstrings, user-facing docs, or commit messages.
- Use plain `TODO:` / `FIXME:` describing the actual work — not a ticket number.
  Write `# TODO: read the fan-out cap from settings`, never `# TODO(T-021)`.

## Error handling

- **No silent excepts.** Never `except Exception: pass`. Every caught exception is
  either handled meaningfully or re-raised with context (`raise ... from exc`).
- **Catch specifically.** `except psycopg.Error`, not `except Exception`.
- Errors from the LLM, the database, and the embedding model surface with enough
  context to say *which* call failed and *with what input*.
- A failure must never be reported as an empty result. "No listings matched" and
  "the query errored" look identical to the Check node and would send it relaxing
  filters to fix a broken connection.

## Types

- `mypy --strict` passes on `src/`. No `Any` without a comment justifying it.
- Pydantic models validate every LLM output and every external input before use.
- Use `Literal` for closed sets (backends, relaxation strategies) so an invalid
  value fails at load, not at the third node.

## SQL

- **Never build SQL by string concatenation.** Parameterized queries only.
  Identifiers that must be dynamic are composed with `psycopg.sql.Identifier`.
- The `ORDER BY` operator must match the index's operator class. Cosine
  (`<=>`) with normalized embeddings; a mismatch silently returns wrong neighbors.
- A query whose plan must use the index has a test asserting `Index Cond` in
  `EXPLAIN`. Plans regress quietly.

## The LLM boundary

- Nodes never import `anthropic` directly — they take a backend protocol.
- Model IDs come from settings. **Never hardcode a model ID in a node.**
- Every call records token usage so it can be priced.
- **No test calls the real API.** Not in CI, not locally by default.

## Privacy

`docs/DATA.md` is binding. Personal fields are dropped at parse time; review text
passes through the one scrubbing function before it is embedded or displayed.

## Testing

- Pure logic (filter→SQL, parsing, the document template, relaxation rules) is
  unit-tested exhaustively — it is cheap and it is where the bugs are.
- Integration tests run against the Compose database on a **small synthetic
  fixture**, never real listings.
- Quality claims are backed by an assertion or a reproducible number, not prose.
- A test that would pass if the behavior regressed is not a test.

## Formatting

`ruff check`, `ruff format`, and `mypy` are clean. CI enforces all three.
