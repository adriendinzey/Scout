# Scout — Architecture

## 1. Shape of the system

```
scout ask "<query>"
        │
        ▼
┌───────────────────────────────────────────────┐
│  LangGraph StateGraph (src/scout/graph/)      │
│                                               │
│   parse ──▶ retrieve ──▶ check ──┬──▶ answer  │
│               ▲                  │            │
│               └──── relax ───────┘            │
└───────────────┬───────────────────────────────┘
                │
     ┌──────────┴──────────┐
     ▼                     ▼
 Anthropic API      PostgreSQL 17
 (parse/check/       ├── brindle  (filtered vector search)
  answer)            └── vector   (pgvector — evaluation baseline)
```

Four layers, kept separate so each is testable on its own:

| Layer | Package | Depends on |
|---|---|---|
| **Pure logic** | `filters`, `sql`, parsing, the document template, relaxation rules | nothing external — unit-testable with no database and no API |
| **Retrieval** | `scout.retrieval` | PostgreSQL |
| **LLM boundary** | `scout.llm` | the Anthropic API, behind an interface |
| **Graph** | `scout.graph` | all three, wired by LangGraph |

**Rule:** new logic goes in the pure layer wherever it can. Filter-to-SQL mapping,
relaxation rule enforcement, amenity parsing, and name scrubbing are all pure
functions with no I/O — that is why they can be tested exhaustively and cheaply.

## 2. Why the graph is hand-written

No `create_react_agent` or other prebuilt helper. The graph is an explicit
`StateGraph` with named nodes and a visible conditional edge, because the point of
the project is to show the control flow, explain why the loop terminates, and
render it as a diagram. A prebuilt agent hides exactly the part worth showing.

## 3. The LLM boundary

Nodes never import `anthropic` directly. They call a protocol with two
implementations:

- **`AnthropicBackend`** — the real API. Uses prompt caching on the stable prefix
  and reports token usage so every call has a dollar figure attached.
- **`FakeBackend`** — deterministic canned responses keyed by input. This is what
  tests and CI use, and it is the default so an unconfigured checkout cannot spend
  money.

Every LLM output is validated with Pydantic **before** it reaches the database or
the answer. An invalid parse is retried once with the validation error fed back,
then fails loudly.

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

Full column list: `SCOUT_SPEC.md` § 5.2 and the migration that creates it.

## 6. What gets embedded

One document per listing: name, description, neighbourhood name, room and property
type, the amenity list, and a capped amount of recent review text (~150 words).
Host and reviewer names are removed **before** embedding, not just before display.

The exact embedded string is stored in `doc_text` so embeddings can be regenerated
and debugged. The template is **one tested function** — if the document shape
drifts between what was indexed and what a query builds, retrieval quality decays
in a way that is very hard to spot.

## 7. State and the trace

Every attempt (filters used, SQL, row count, latency) and every relaxation (what
was loosened, and the model's one-sentence reason) is recorded in the graph state.
That record is what `scout ask` prints as a trace, what `--json` emits, and what
the evaluation measures the loop with. It is a first-class output, not debug
logging.
