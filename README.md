# Scout

[![CI](https://github.com/adriendinzey/Scout/actions/workflows/ci.yml/badge.svg)](https://github.com/adriendinzey/Scout/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Agentic search over short-term rental listings — a LangGraph workflow that parses natural language into filters, then lets Claude drive the search itself: counting matches, checking what a field actually looks like, and searching again with different filters when the first attempt comes back thin.**

You type what you actually want:

> *"quiet place near the water for two, under $200 a night, good for working remotely, well reviewed"*

Scout turns that into structured filters plus a semantic query, then hands them to an agent with five tools over a filter-aware vector index. The agent decides what to do: count how many listings match before spending a search, look up the median price in the neighbourhood before moving a price ceiling, search, read a listing, pull its reviews. It works inside limits the code enforces — a tool-call budget, a search budget, a cost ceiling, and a floor on group size it is not allowed to cross. Then it writes a short answer citing the listings and guest reviews it actually used, and says plainly what it could not satisfy.

> **Status: the data is loading; the demo does not run yet.** `scout ask` is not implemented — the embedder, the index build, the Parse node, the Retrieve node and the agent loop are the next milestones. What is actually built is listed in [What works today](#what-works-today); nothing else below is claimed as working, and the numbers in [Evaluation](#evaluation) stay empty until they come from a real run. Milestones: [docs/ROADMAP.md](docs/ROADMAP.md).

---

## What works today

State at commit `e9ab31f` (2026-09-18). Anything not ticked here is design, not a claim.

| | | Asserted by |
|---|---|---|
| Postgres 17 in one image with **Brindle** (pinned `025b1315`) and **pgvector 0.8.0** | ✅ | `docker compose up -d --wait`, `uv run scout doctor` |
| **The premise, asserted:** `EXPLAIN` shows `Index Cond`, not a post-scan `Filter`, for every supported predicate shape; `ef_search` bounds a ranked scan; NULL satisfies no comparison | ✅ | `tests/integration/test_brindle_pushdown.py` |
| Schema, lookup tables, and the migration runner — `scout data migrate` (migrations `0001`–`0002`) | ✅ | `tests/integration/test_schema.py`, `tests/unit/test_migrate.py` |
| **`scout data load`** — streams both CSVs into PostgreSQL, drops host and reviewer identity while parsing into rows that have no field to hold it, caps reviews per listing, replaces rather than duplicates on reload, and refuses to run without the snapshot's release date | ✅ | `tests/unit/test_load.py`, `tests/unit/test_parsers.py`, `tests/integration/test_load.py` |
| Typed settings, `scout doctor`, and cost accounting with cache and Batch rates | ✅ | `tests/unit/test_config.py`, `tests/unit/test_pricing.py` |
| `Filters` model and the **filter→SQL mapping**: parameterized filtered-vector queries, disjunction fan-out, the cap, reported post-filters and NULL exclusions | ✅ pure logic, no caller yet | `tests/unit/test_filters_to_sql.py` |
| **LLM boundary:** backend protocol, the Anthropic client with its cacheable prefix, a deterministic fake that scripts a whole tool-use conversation, per-run tokens and dollars | ✅ | `tests/unit/test_backend.py`, `test_fake_backend.py`, `test_anthropic_backend.py`, `test_usage.py` |
| Amenity columns, the document template, `scout data embed`, `scout data index` | ⬜ M1 | — |
| Parse node, Retrieve node, `scout ask` | ⬜ M2 | — |
| The agent tool loop, the five tools, the enforced budgets, `--mode fixed` | ⬜ M3 | — |
| Run store, `scout trace`, `scout runs`, per-run cost | ⬜ M3.5 | — |
| Ranking, review citations, the Answer node | ⬜ M4 | — |
| The evaluation harness and every number in [Evaluation](#evaluation) | ⬜ M5 | — |

**480 tests pass at this commit** — 430 unit, 50 integration against the Compose database — with `ruff` and `mypy --strict` clean. The commands that are not implemented exit non-zero naming the milestone they land in, rather than printing nothing and returning 0.

## Why this exists

The retrieval step runs on [**Brindle**](https://github.com/adriendinzey/Brindle), a PostgreSQL vector-search extension I wrote that keeps recall high when a SQL predicate has to hold at the same time. Scout is Brindle's real workload: filtered vector search over real listings, not synthetic benchmarks.

That makes the interesting question measurable rather than rhetorical — **does pushing the predicate into the index actually beat filtering after the fact?** Scout answers it against pgvector baselines and exact search, on the same rows, and publishes the losses alongside the wins.

## The agent loop

> **Not built yet — this is the design for M3, and the reason the project exists.** Today the budgets below are validated settings that `scout doctor` prints; the loop that enforces them, and the tools it would call, are the work after Parse and Retrieve.

What makes this agentic is not that an LLM is involved — it is that the model chooses the next action and the code decides what is permissible.

```mermaid
flowchart LR
    START([start]) --> parse[Parse<br/>query → filters + semantic query]
    parse --> agent{Agent<br/>which tool next?}
    agent -->|tool call| tools[search_listings · count_matches<br/>field_stats · get_listing · get_reviews]
    tools --> agent
    agent -->|done, or a limit fired| answer[Answer<br/>rank, cite, explain]
    answer --> END([end])
```

Parse stays deterministic on purpose: one grounded, schema-validated call, so the agent starts from validated filters and the expensive grounding block stays cacheable.

**What the model decides:** which tool to call, with which arguments, when to search again with different filters, when it has enough, and how to explain what it changed.

**What the code will refuse to let it do:**

| | |
|---|---|
| **10 tool calls** per query, **4 searches** | the loop terminates on a budget, not on the model's goodwill |
| **A group-size floor** | any filter set that lowers the parsed `min_accommodates` is rejected with a structured error — a place that cannot fit the party is useless however well it scores |
| **No repeated calls** | an identical call with identical arguments returns the cached result, and the repeat is recorded |
| **A per-query cost ceiling** | priced from real token counts as the loop runs |
| **A 60-second wall clock** | |
| **An honest ending** | when the loop stops for any reason other than the model finishing, the answer says what could not be satisfied |

Each limit lands as a pure function with a test that fails if the limit is removed — a limit that lives only in the prompt is a suggestion, and a milestone is not done until those tests exist.

### The baseline it gets measured against

`--mode fixed` is the hardcoded path — search, and if fewer than five results come back, loosen the most restrictive filter by rule and search again — so the question "does letting the model choose tools actually beat a hardcoded path?" is answered with numbers rather than vibes. Both modes share the retrieval code, the guardrails and the trace shape; only the decision differs. If `fixed` wins on quality and costs less, this README will say so.

> A sample trace — the agent calling `count_matches`, then searching again with different filters — goes here once M3 lands and there is a real one to paste.

## What it costs to run

Scout calls Claude for Parse (once), for each turn of the agent loop (capped), and for the Answer. **You supply your own `ANTHROPIC_API_KEY`** — it is read from the environment and never committed, so running Scout costs the person running it, not the author.

Every run will print what it cost, and four things keep that number small — two of them already in:

| | |
|---|---|
| **Prompt caching** ✅ | The stable prefix — grounding block and tool definitions — is byte-identical on every query, so it bills at ~0.1× after the first call. The breakpoint is set at the boundary today, and cache reads and writes are counted and priced |
| **A per-query ceiling** ⬜ | `SCOUT_MAX_QUERY_COST_USD` (default $0.10) is validated configuration today; the mid-loop check that stops the agent and answers with what it has arrives with the loop (M3) |
| **Batch API** ⬜ | Half-price rates are in the cost model; the evaluation harness that submits the batches is M5 |
| **A backend switch** ✅ | `SCOUT_LLM_BACKEND=fake` is the default and returns scripted tool-use conversations with no network, so development and CI cost nothing. The unit suite has its sockets taken away rather than being trusted not to use them; Claude is for runs whose numbers get published |

Per-query cost is **budgeted, not yet measured** — the real figure, per mode, comes from the evaluation run and lands in the table below.

A Claude Pro/Max subscription does **not** grant API access — that is a separate product. Scout needs API credits.

## Quick start

What runs today:

```bash
# 1. Database: Postgres 17 with Brindle (pinned) and pgvector in one image.
#    The first build compiles Brindle from source and takes a few minutes.
docker compose up -d --wait

# 2. Python environment
uv sync --all-extras
cp .env.example .env      # SCOUT_LLM_BACKEND defaults to fake; a key is only
                          # needed for real Claude calls, which nothing makes yet

# 3. Confirm the stack is actually working
uv run scout doctor

# 4. Create the schema
uv run scout data migrate

# 5. Load the snapshot — see "Getting the data"; Scout never downloads it for you.
#    SCOUT_SNAPSHOT_DATE is required: listing ids are not stable between releases.
uv run scout data load

# 6. Run the suite (integration tests need step 1)
uv run pytest
```

The rest is specified and not implemented. These commands exist and exit non-zero naming the milestone they land in:

```bash
uv run scout data embed    # M1
uv run scout data index    # M1

# M2 — the first thing worth showing anyone
uv run scout ask "quiet place near the water for two, under \$200, good for working remotely"

# M3.5 — the step table: what it called, how long each step took, what it cost
uv run scout ask "3-bed in Hackney under \$60 a night, 5-star" --trace

# M3 — the hardcoded baseline instead of the agent
uv run scout ask "..." --mode fixed

# M3.5 — earlier runs, stored and replayable
uv run scout runs --last 20
uv run scout trace <run_id>
```

### Getting the data

Scout uses [Inside Airbnb](https://insideairbnb.com/get-the-data/) listings and reviews for **London**. **You download them yourself, once**, and point `.env` at the local files — nothing in Scout fetches from Inside Airbnb at runtime, by deliberate design. Full instructions, and the privacy rules Scout applies at load time: [docs/DATA.md](docs/DATA.md).

## Evaluation

> Filled in from a single named run once M5 lands. Until then this section is deliberately empty rather than optimistic.

| | |
|---|---|
| City / snapshot | London, _(date TBD)_ |
| Listings indexed | _TBD_ |
| Brindle commit | [`025b131`](https://github.com/adriendinzey/Brindle/commit/025b1315f40a1384695d81ff846bfc8c104c5aea) |
| Embedding model | `all-MiniLM-L6-v2` (384d, normalized) |

**Parse** — per-field filter accuracy, invalid-JSON rate, hallucinated-column rate.
**Retrieval** — recall@10 vs exact search, precision@5 vs hand labels, p50/p95 latency **split cold and warm**, for Brindle / pgvector post-filter / pgvector iterative / exact.
**Agent behaviour** — tool calls and searches per query, which tools actually get used, how runs end (finished / limit / timeout / error), invalid-argument rate, repeat rate.
**Agent vs fixed** — the same eval set in `--mode agent`, `--mode fixed`, and `--mode fixed --no-relax`: share of queries ending with ≥ 5 results, precision@5, latency, tokens, and **dollars per query**.

Method: [docs/EVALUATION.md](docs/EVALUATION.md).

## Limitations

Stated up front rather than discovered by a reader:

- **One city, one snapshot.** Results do not generalize to other markets, and listing IDs are not stable across Inside Airbnb snapshots.
- **Reviews are not searchable.** They are stored for citation and a capped excerpt feeds the listing's embedding, but the unit of search is the listing. A query about something only a reviewer mentioned may miss.
- **`OR` is not pushed down.** Brindle pushes equality and ranges combined with `AND`. Disjunctions run one retrieval per branch and merge, capped at 4 branches, after which Scout post-filters and records that it did.
- **Coordinates are approximate.** Inside Airbnb offsets listing locations for privacy, so "near the water" is a semantic hint and a bounding box, not a real distance.
- **Nulls exclude.** A listing with no rating does not satisfy `rating >= 4.5`. Filtering on quality silently drops new listings, and Scout says so when it does.
- **No availability or pricing intelligence.** Scout cannot answer "is it free in June" — dates are not in scope, and such asks are reported as unsupported rather than quietly ignored.

## Development

Setup, the parallel-worktree workflow, and the daily loop: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Design rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Conventions: [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md).

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -m "not integration"    # unit tests, no database needed
uv run pytest                          # everything, needs docker compose up
```

## Attribution and license

Listing and review data from **[Inside Airbnb](http://insideairbnb.com)**, licensed **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. Inside Airbnb is a mission-driven project reporting on the effect of short-term rentals on housing; Scout is a guest-side search demo and is not affiliated with it. **No Inside Airbnb data is redistributed in this repository** — see [docs/DATA.md](docs/DATA.md).

Scout's own code is [MIT licensed](LICENSE).

Built with AI assistance (Claude Code), the same way [Brindle](https://github.com/adriendinzey/Brindle) was.
