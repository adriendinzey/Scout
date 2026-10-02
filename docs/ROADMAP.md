# Scout Roadmap

The public view of where the project is. Status here reflects what actually
works at the current commit, not what is planned.

**Legend:** ✅ done · 🚧 in progress · ⬜ not started

## M0: Project skeleton ✅
Repo layout, the `uv` project, ruff, mypy, pytest, CI, and a Docker Compose
Postgres 17 image carrying Brindle (pinned) and pgvector.
*Exit: `docker compose up` works, both extensions load, CI green.* **Met.** Both
extensions load in one database, and the pushdown contract tests assert that
`EXPLAIN` shows `Index Cond` rather than a post-scan `Filter` for every supported
predicate shape.

## M1: Data loaded and indexed 🚧
Load, embed, and index the London Inside Airbnb snapshot, with the privacy rules
enforced and tested. **In:** the schema, the migration runner (`scout data
migrate`), and `scout data load`, which drops host and reviewer identity while
parsing instead of nulling it afterwards. **Not in:** the amenity columns, the
document template, the embedder, and the index build.
*Exit: every listing is embedded, the Brindle index builds, `EXPLAIN` shows
`Index Cond`, and cold and warm latency and backend memory are measured.*

## M2: Parse and Retrieve 🚧 *(priority)*
Natural language becomes validated filters, and those filters become a filtered
vector search with a trace you can read. **In:** the LLM boundary, which holds the
backend protocol, the Anthropic client with its cacheable prefix, the
deterministic fake, and usage and cost accounting, together with the
filter-to-SQL mapping. **Not in:** the Parse node, the Retrieve node, and
`scout ask`.
*Exit: `scout ask` returns filtered results from Brindle for a real query.*

## M3: Agent tool loop ⬜ *(priority)*
This is the agentic part. Claude is given five tools, which are search, count
matches, field stats, listing detail and reviews, and it chooses which to call
next, including whether to search again with different filters. The limits that
fence it are enforced in code rather than stated in the prompt. The hardcoded
search-then-relax path stays behind `--mode fixed` as the baseline it gets
measured against.
*Exit: an over-constrained query shows the agent checking counts, changing its
filters for a stated reason, and ending with results. `--mode fixed` still runs.*

## M3.5: Tracing and cost ⬜
Every run records each step with its latency, tokens, and dollars, stored in
Postgres and as a JSONL file. `scout trace <run_id>` and `scout runs` read them
back, and `scout ask` prints what the query cost.
*Exit: traces are stored and replayable, and a test proves the per-query cost
ceiling stops a loop.*

## M4: Answer ⬜
Ranking, review citations, and transparency about what was relaxed and what could
not be satisfied.
*Exit: answers cite only fetched listings and reviews, and contain no personal
names.*

## M5: Evaluation ⬜
The numbers. Parse accuracy, retrieval quality against pgvector and exact search,
how the agent actually behaves, including tool calls per query and how runs end,
and whether letting the model choose tools beats the hardcoded path on quality,
latency and dollars.
*Exit: one command produces the full report from a single run, in both modes.*

## M6: Stretch ⬜
A web UI, a hosted demo behind hard spending guardrails, and a learned reranker
evaluated against plain vector ranking. None of it starts before M5 is done.

---

Scope, including what is deliberately **out** of scope, is fixed in the build
spec. The notable exclusions are that no prebuilt agent helper is used, because
the tool loop is written out explicitly, that reviews are not indexed as their own
rows in v1, and that nothing in this repository changes Brindle.
