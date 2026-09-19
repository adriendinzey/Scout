# Scout — Architecture

## 1. Shape of the system

```
scout ask "<query>"
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph (src/scout/graph/)                              │
│                                                                       │
│   --mode agent  (default)                                             │
│     parse ──▶ agent ──▶ answer                                        │
│                ▲  │                                                   │
│                │  ▼                                                   │
│                tools:  search_listings   count_matches   field_stats  │
│                        get_listing       get_reviews                  │
│                                                                       │
│   --mode fixed  (the baseline the agent is measured against)          │
│     parse ──▶ retrieve ──▶ check ──┬──▶ answer                        │
│                  ▲                 │                                  │
│                  └───── relax ─────┘                                  │
└───────────────┬───────────────────────────────────────────────────────┘
                │
     ┌──────────┴──────────┐
     ▼                     ▼
 Anthropic API      PostgreSQL 17
 (parse/agent/       ├── brindle  (filtered vector search)
  answer)            └── vector   (pgvector — evaluation baseline)
```

Four layers, kept separate so each is testable on its own:

| Layer | Package | Depends on |
|---|---|---|
| **Pure logic** | `filters`, `sql`, parsing, the document template, the loop limits, relaxation rules | nothing external — unit-testable with no database and no API |
| **Retrieval** | `scout.retrieval` | PostgreSQL |
| **Tools** | `scout.tools` | retrieval + pure logic; **no LLM** |
| **LLM boundary** | `scout.llm` | the Anthropic API, behind an interface |
| **Graph** | `scout.graph` | all of the above, wired by LangGraph |

**Rule:** new logic goes in the pure layer wherever it can. Filter-to-SQL mapping,
the loop limits, relaxation rules, amenity parsing, and name scrubbing are all
pure functions with no I/O — that is why they can be tested exhaustively and
cheaply.

Parse runs before either mode and is deliberately *not* agentic: it is one
grounded, schema-validated call producing `filters`, `semantic_query`, and
`unsupported_constraints`. Keeping it deterministic is what makes the expensive
schema grounding cacheable, and it means the agent starts from a validated filter
set rather than inventing one.

## 2. The agent loop, and what the code refuses to let it do

In `--mode agent`, a single `agent` node runs a tool-use conversation with Claude:
on each turn the model either calls a tool or says it is done, the node executes
the tool, appends the result, and loops. **The model chooses which tool to call
and with what arguments; the code decides what it is allowed to do.**

The loop is written out explicitly. No `create_react_agent`, no prebuilt agent
helper, anywhere in the codebase — the control flow, the stopping conditions, and
the trace are the part of this project worth showing, and a prebuilt agent hides
exactly those. The same reasoning applies to the graph itself: an explicit
`StateGraph` with named nodes and visible edges, so it can be rendered as a
diagram generated from the compiled graph rather than drawn by hand.

### The tools

Each tool is a plain Python function with a JSON schema, a Pydantic-validated
input model, and its own unit tests. Results are compact JSON, never prose.
Filters are validated against the same model Parse produces, so invalid arguments
come back as a structured error the model can read and correct — and the error is
recorded in the trace, because how often the model gets its own tool arguments
wrong is a number worth publishing.

| Tool | Input | Returns |
|---|---|---|
| `search_listings` | filters, semantic query, limit | ranked listings: id, name, distance, and the filter-relevant fields |
| `count_matches` | filters | one count — how strict a filter set is, *before* spending a search on it |
| `field_stats` | field name, filters | min, max, median, non-null count under those filters |
| `get_listing` | listing id | the stored fields for one listing |
| `get_reviews` | listing id, limit | stored review excerpts, personal names removed |

There is no `relax` tool. Relaxation is what it looks like when the agent calls
`count_matches` or `search_listings` with different filters — which is the point:
the interesting version of relaxation is a reasoned decision ("only 3 listings
match at £150; the median in Hackney is £180") rather than a blind 15% bump.

`field_stats` takes a column name chosen by the model, so it maps that name
through an allowlist of filterable columns to a `psycopg.sql.Identifier`. An
unknown field is a structured error, never interpolated text.

### The limits

Every one of these is enforced in code, has a test proving it fires, and has a
test proving the run still produces an answer when it does:

| Limit | Default | Why |
|---|---|---|
| Tool calls per query | 10 | The loop must terminate on a budget, not on the model's goodwill |
| `search_listings` calls per query | 4 | Searches are the expensive call — embedding plus a ranked scan |
| Guest count | — | A filter set that lowers `min_accommodates` below the parsed value is **rejected** with a structured error. A place that cannot fit the group is useless however well it scores |
| Repeated calls | — | An identical call with identical arguments is not executed twice: the cached result is returned and the repeat is noted in the trace |
| Cost | $0.10/query | Priced from real token counts as the loop runs, not estimated |
| Wall clock | 60 s | |

When the loop stops for any reason other than Claude finishing, the Answer node
**must say what could not be satisfied**. A truncated search that reads like a
complete one is the worst output this system can produce.

Before its final turn the model states, in a sentence or two, what it searched,
what it changed and why, and what it could not satisfy. That text goes into the
trace and into the answer's transparency section.

### Why `--mode fixed` exists

`--mode fixed` is the original hardcoded path: search, and if fewer than
`thin_result_threshold` rows come back, loosen one filter by **rule** — the most
restrictive constraint, measured by how many rows it excludes — and search again,
at most `max_relaxations` times. It shares the retrieval code, the relaxation
vocabulary, the guest-count rule, and the trace shape with agent mode; only the
*decision* differs.

It is not dead code, and it is not a fallback. It is the comparison that makes the
evaluation mean something: does letting the model choose tools actually beat a
hardcoded path, on result rate, precision, latency, and dollars? If it does not,
that is a real finding and the README says so. `--no-relax` narrows it further to
a single search, for the "does the loop help at all" control.

## 3. The LLM boundary

Nodes never import `anthropic` directly. They call a protocol with two
implementations:

- **`AnthropicBackend`** — the real API. Caches the stable prefix (system prompt,
  grounding block, tool definitions) and reports token usage so every call has a
  dollar figure attached.
- **`FakeBackend`** — deterministic canned responses, including scripted tool-use
  turns, so a whole agent loop can be exercised with no network. This is what
  tests and CI use, and it is the default so an unconfigured checkout cannot spend
  money.

The protocol is multi-turn: it takes a message list that may contain `tool_use`
and `tool_result` blocks, returns the assistant turn plus its stop reason, and
reports `Usage` per call. A single-shot "prompt in, JSON out" interface cannot
express a tool loop, and Parse is just the one-turn case of the same call.

Every LLM output is validated with Pydantic **before** it reaches the database or
the answer — a parse into `Filters`, a tool call into that tool's input model. An
invalid parse is retried once with the validation error fed back, then fails
loudly. An invalid tool call is returned to the model as a structured error and
counts against the tool budget.

Model IDs come from settings (`model_parse`, `model_agent`, `model_answer`), never
from a node.

## 4. Retrieval

### Why the filters go *into* the index

The whole point. Given `WHERE price < 200 AND accommodates >= 2 ORDER BY embedding
<=> $1 LIMIT 10`, there are three ways to run it:

1. **Post-filter** — vector search first, filter the results. Cheap, but if only
   2% of listings match the predicate, the top-100 nearest may contain almost no
   matches. Recall collapses.
2. **Pre-filter** — filter first, then scan the matches exactly. Correct, but
   throws away the index.
3. **Pushdown** — evaluate the predicate *during* graph traversal, so the search
   budget is spent on nodes that can actually be returned. This is Brindle.

Scout measures all three, plus exact search as ground truth. If pushdown does not
win on real queries, the report says so.

### The constraints Brindle imposes

Verified against Brindle at the pinned commit. These are not guesses — they shape
the schema.

| Constraint | Consequence for Scout |
|---|---|
| Filterable columns must be `bool`, `int2`, `int4`, `int8`, `float4`, `float8` | Neighbourhood, room type, and property type are **integer foreign keys**, not text. Timestamps are refused |
| Filter columns must be **key columns after the vector**, never `INCLUDE` | `EXPLAIN` must show `Index Cond`, not `Filter`. An integration test asserts this |
| Only `=`, `<`, `<=`, `>`, `>=`, `BETWEEN` joined by `AND` are pushed down | `OR` and `NOT` are not. Disjunctions fan out into one query per branch and merge (§5) |
| PostgreSQL allows ≤ 32 key columns per index | The vector plus every filter column must fit in 32. The amenity boolean budget is what is left over |
| A ranked scan returns at most `brindle.ef_search` rows | `SET brindle.ef_search` ≥ the candidate pool. Config rejects a pool larger than `ef_search` at startup rather than letting `LIMIT` silently under-return |
| A NULL vector is skipped by ranked scans | Every indexed listing must have an embedding. The index build asserts this |
| Each connection decodes its own copy of the index | **~1.9 KB/node at 384 dims** (measured: 887 B/node at 128 dims, of which 512 B is the vector). London ≈ 172 MB **per backend**. Pool size is a memory multiplier — default 4 |
| A new connection pays a one-time decode cost | Use a long-lived pool. Report **cold and warm latency separately**, never just warm |
| Storage is a whole-index blob; every write rewrites it | Load all rows, *then* `CREATE INDEX`. Never insert row-by-row into an indexed table |
| Build is single-threaded | Index builds take minutes. Record the time in the eval report |

### Disjunctions

Brindle does not push down `OR`, so `filters.neighbourhoods = ["Hackney", "Islington"]`
becomes two retrievals merged by distance and deduplicated by listing id. Fan-out
is capped (default 4). Past the cap, Scout falls back to a post-filter and
**records in the attempt that it did**, because that weakens recall and a silent
fallback would corrupt the evaluation.

## 5. Schema

`listings` holds one row per searchable unit. Its filterable columns are typed for
Brindle; its text columns are for display and for building `doc_text`.

Nullability is load-bearing: **a NULL satisfies no comparison**, so `rating >= 4.5`
silently excludes every listing with no rating. Parse and Check both have to know
this, and the Answer node says so when a quality filter was applied.

Full column list: `migrations/0001_initial.sql`, which is the authority on the
column types, on which columns are nullable and why, and on how much of the
32-key-column index budget is left for amenity booleans. `0002` amends it with
what the snapshot turned out to contain: `price_usd` became `price_gbp`, because
London's prices are quoted in pounds, and `instant_bookable` became nullable,
because the scrape no longer publishes it.

## 6. What gets embedded

One document per listing: name, description, neighbourhood name, room and property
type, the amenity list, and a capped amount of recent review text (~150 words).
Host and reviewer names are removed **before** embedding, not just before display.

The exact embedded string is stored in `doc_text` so embeddings can be regenerated
and debugged. The template is **one tested function** — if the document shape
drifts between what was indexed and what a query builds, retrieval quality decays
in a way that is very hard to spot.

## 7. State, the trace, and cost

Every run gets a `run_id` and a series of step records — one per node and one per
tool call:

| Field | Notes |
|---|---|
| `run_id`, `step_index` | |
| `kind` | `node` or `tool_call` |
| `name` | `parse`, `search_listings`, … |
| `input_summary`, `output_summary` | truncated; review text scrubbed **before** it is written, not before it is displayed |
| `latency_ms` | |
| `input_tokens`, `output_tokens` | from the API response usage, never estimated |
| `cost_usd` | token counts × the price table |
| `error` | structured, when a step fails |

Runs and steps are stored in Postgres (`runs`, `run_steps`) and written as one
JSONL file per run under a gitignored directory, so a trace survives a database
that was torn down and rebuilt.

That record is a first-class output, not debug logging. `scout ask` prints the
human trace and the run's total cost; `--trace` adds the step table; `scout trace
<run_id>` replays a stored run; `scout runs --last 20` lists recent ones. The
evaluation reads the same records to report tool calls per query, searches per
query, how runs ended, invalid-argument rate, repeat rate, and dollars per query.

Prices live in a table that can be overridden by a file (`SCOUT_PRICE_TABLE`),
because published prices change and a stale number in a report is worse than no
number. A model with no known rate raises rather than costing zero.

If `LANGSMITH_API_KEY` is set, traces are also sent to LangSmith. It is strictly
optional: everything above works without it, no test depends on it, and it is the
one path that sends query and listing text off the machine — so it stays off
unless the key is present deliberately.
