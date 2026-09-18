# Scout — Roadmap

The public view of where the project is. Status here reflects what actually
works at the current commit, not what is planned.

**Legend:** ✅ done · 🚧 in progress · ⬜ not started

## M0 — Project skeleton ⬜
Repo layout, `uv` project, ruff/mypy/pytest, CI, and a Docker Compose Postgres 17
image carrying Brindle (pinned) and pgvector.
*Exit: `docker compose up` works, both extensions load, CI green.*

## M1 — Data loaded and indexed ⬜
Load, embed, and index the London Inside Airbnb snapshot. Privacy rules enforced
and tested.
*Exit: every listing embedded, Brindle index builds, `EXPLAIN` shows `Index Cond`,
cold/warm latency and backend memory measured.*

## M2 — Parse and Retrieve ⬜ *(priority)*
Natural language → validated filters → filtered vector search, with a trace.
*Exit: `scout ask` returns filtered results from Brindle for a real query.*

## M3 — Agent tool loop ⬜ *(priority)*
The agentic part: Claude is given five tools — search, count matches, field
stats, listing detail, reviews — and chooses which to call next, including
whether to search again with different filters. Fenced by limits the code
enforces, not the prompt. The hardcoded search-then-relax path stays behind
`--mode fixed` as the baseline it gets measured against.
*Exit: an over-constrained query shows the agent checking counts, changing its
filters for a stated reason, and ending with results — and `--mode fixed` still
runs.*

## M3.5 — Tracing and cost ⬜
Every run records each step with its latency, tokens, and dollars, stored in
Postgres and as a JSONL file. `scout trace <run_id>` and `scout runs` read them
back; `scout ask` prints what the query cost.
*Exit: traces are stored and replayable, and a test proves the per-query cost
ceiling stops a loop.*

## M4 — Answer ⬜
Ranking, review citations, and transparency about what was relaxed and what could
not be satisfied.
*Exit: answers cite only fetched listings and reviews, and contain no personal names.*

## M5 — Evaluation ⬜
The numbers: parse accuracy, retrieval quality against pgvector and exact search,
how the agent actually behaves (tool calls per query, how runs end), and whether
letting the model choose tools beats the hardcoded path — on quality, latency and
dollars.
*Exit: one command produces the full report from a single run, in both modes.*

## M6 — Stretch ⬜
Web UI, a hosted demo behind hard spending guardrails, and a learned reranker
evaluated against plain vector ranking. Only after M5.

---

Scope, including what is deliberately **out** of scope, is fixed in the build
spec. Notable exclusions: no prebuilt agent helpers — the tool loop is written
out explicitly — reviews are not indexed as their own rows in v1, and no changes
to Brindle from this repo.
