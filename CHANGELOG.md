# Changelog

Notable changes to Scout. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Project skeleton: `uv` project targeting Python 3.12, ruff, mypy (strict on
  `src/`), and pytest with unit/integration separation.
- Docker Compose PostgreSQL 17 image carrying **Brindle** (pinned to
  `025b1315`) and **pgvector 0.8.0** in one database, so retrieval baselines
  compare identical rows. The container records its Brindle commit at
  `/etc/brindle-commit.txt`.
- GitHub Actions CI: lint + types, unit tests, and an integration job that builds
  the database image and runs the database-backed tests.
- Typed settings object covering the LLM backend, database, embeddings,
  retrieval, and the agent loop. Defaults to the **fake** LLM backend so an
  unconfigured checkout cannot spend money, and rejects a candidate pool larger
  than `ef_search` at startup rather than letting a ranked scan silently
  under-return.
- Cost accounting with published per-model rates, prompt-cache and Batch-API
  discounts, so every run can report dollars alongside tokens.
- `scout doctor`, which verifies the database and both extensions.
- `scout data migrate`, which applies the numbered SQL migrations in
  `migrations/` and records them in `schema_migrations`. Running it again is
  a no-op; a migration edited after it was applied, or recorded with no file
  left in the repository, is an error rather than a silent divergence.
- The schema Scout searches: `listings`, `reviews`, and the neighbourhood,
  room-type, property-type and amenity lookup tables. Filterable columns are
  typed for Brindle pushdown, personal fields are absent by construction
  rather than nulled, and the index key budget leaves room for 15 amenity
  booleans alongside the vector and the 16 other filters.
- Integration tests asserting the project's premise: a predicate is pushed into
  the Brindle index as an `Index Cond` rather than applied as a post-scan
  `Filter`, across every supported predicate shape; returned rows satisfy the
  predicate; a ranked scan is bounded by `ef_search`; and NULL attributes satisfy
  no comparison.
- Parallel-development tooling: `scripts/worktree.sh` gives each task its own
  worktree, virtualenv, and Postgres container on a free port;
  `scripts/db.sh` wraps Compose with the sandbox's settings.
- The filter vocabulary and the mapping from it to SQL: a validated `Filters`
  model where `None` means unconstrained, and a builder that composes one
  parameterized filtered-vector query per disjunction branch. Fan-out is capped;
  past the cap, and for an amenity with no indexed column, the plan applies the
  condition after the scan and **records that it did**, because that weakens
  recall and a silent fallback would corrupt the evaluation. Filters that
  exclude NULLs are reported back, since a comparison against NULL is never
  true and the answer has to say so.
- Documentation: architecture, data licensing and privacy rules, development
  setup, evaluation method, coding standards, and roadmap.

### Notes

Nothing is claimed as working beyond what the tests above assert. The evaluation
section of the README is deliberately empty until it can be filled from a single
real run.
