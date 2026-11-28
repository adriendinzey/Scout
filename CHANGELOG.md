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
- The LLM boundary every node uses to reach Claude: one multi-turn call shape that
  carries `tool_use` and `tool_result` blocks and reports its stop reason, so the
  agent loop and the single-turn Parse call are the same call rather than two
  interfaces. Behind it sit the real Anthropic client — which marks the stable
  prefix cacheable, tool definitions and grounding block together, while the
  query stays outside it — and a deterministic fake that scripts a whole
  tool-use conversation with no network. The fake is the default, so an
  unconfigured checkout still cannot spend money, and no test reaches the
  network: the unit suite has its sockets taken away rather than being trusted
  not to use them. Structured output is requested as a JSON schema; prose is
  never parsed. **No failure is reported as an empty result** — an API error, a
  refusal, a body truncated at `max_tokens`, and output that does not match the
  schema each raise their own error, and the ones that follow a billed call
  carry their token counts, so a run's cost is not understated by exactly the
  calls that went wrong. Token and dollar totals accumulate per run, broken down
  by node and by model.
- `scout data load`, which streams the two Inside Airbnb CSVs into PostgreSQL.
  Host and reviewer identity is dropped while parsing, by typed rows that have
  no field to put it in, so nothing downstream has to remember to remove it.
  Reviews are capped at the most recent few per listing, chosen with a window
  function over the staged rows rather than tracked per listing in memory: peak
  memory over the 2.2 million row review file is under 60 MB. A reload replaces
  rather than duplicates and keeps the generated listing ids stable, because
  reviews and later relevance labels reference them, and it drops `doc_text`
  and the embedding of any listing whose document inputs changed rather than
  leaving an embedding of text that no longer exists. A row that cannot be
  stored — a missing `minimum_nights`, a value no column can hold — is counted
  and reported by column rather than taking the load down with it, and a file
  that yields no usable listing at all is an error rather than an empty
  database. The snapshot's release date is required and recorded beside the
  data, since listing ids are not stable between releases.
- Amenity grouping and the **13 indexed amenity booleans**, in migration
  `0003`. One London snapshot writes 8,735 distinct amenity spellings for a few
  hundred actual amenities, so each group matches whole strings rather than
  substrings: "Dryer", "Free dryer – In unit" and "Paid dryer" are one
  appliance, while "Hair dryer" — in 68% of listings — is not, and "Garden
  view" is not outdoor space. Free never merges with paid. Which amenities earn
  a column was chosen from real frequency counts on selectivity against demand,
  not popularity: a predicate applied after the ranked scan keeps roughly its
  prevalence share of the candidate pool, so wifi (97.1% of listings) stays an
  array post-filter while pets allowed (13.3%) gets a column. 13 of the 15
  available slots are spent, for 30 of PostgreSQL's 32 index key columns.
  `scout data amenities` ranks every group against the loaded data — including
  the eight that were considered and rejected, and the most common spellings no
  group claims — and `--backfill` derives the booleans; every load now derives
  them too, in the same transaction as the amenity arrays they come from. Only
  rows whose flags actually move are written, so a second backfill writes
  nothing, and the rows are streamed rather than buffered (43 MB of Python
  objects for 92,633 listings instead of ~320 MB). A schema that has no amenity
  columns is an error even when there are no rows to derive, rather than a clean
  report on a database that cannot hold the answer.
  `amenities.indexed_column` names the column that filters each spelling, and
  `is_indexed` is generated from it so the two cannot disagree; the migration
  refuses to run on a database where the old hand-set flag was in use, because
  it records *that* an amenity was indexed and never which column filtered it.
- Two schema corrections the London file forced, in migration `0002`:
  `price_usd` became `price_gbp`, because prices are quoted in pounds despite
  the dollar sign the source prints, and `instant_bookable` became nullable,
  because the scrape has stopped publishing it and NULL is what "the snapshot
  does not say" means. Storing `false` would have invented an answer for
  every one of the 92,638 listing rows the file ships.
- Documentation: architecture, data licensing and privacy rules, development
  setup, evaluation method, coding standards, and roadmap.

### Fixed

- An empty `ANTHROPIC_API_KEY` now counts as no key at all. `ANTHROPIC_API_KEY=`
  in a `.env` parsed as an empty string rather than as unset, which satisfied
  every `is None` check: `scout doctor` reported "environment looks good" while
  pointed at the real API with an unusable key, and the failure would have
  surfaced as an authentication error on the first billed call instead of at
  startup. `.env.example` also shipped `SCOUT_LLM_BACKEND=anthropic` with a key
  placeholder, so copying it opted a fresh checkout into spending money; it now
  ships the fake backend and an empty key, matching the rule that switching is a
  deliberate act.

### Notes

Nothing is claimed as working beyond what the tests above assert. The evaluation
section of the README is deliberately empty until it can be filled from a single
real run.
