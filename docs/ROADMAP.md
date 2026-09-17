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

## M3 — Check loop ⬜ *(priority)*
The agentic part: the model decides which constraint to loosen when results are
thin, fenced by rules the code enforces. Plus a rule-based relaxer to compare
against.
*Exit: an over-constrained query shows a reasoned relaxation and ends with results.*

## M4 — Answer ⬜
Ranking, review citations, and transparency about what was relaxed and what could
not be satisfied.
*Exit: answers cite only fetched listings and reviews, and contain no personal names.*

## M5 — Evaluation ⬜
The numbers: parse accuracy, retrieval quality against pgvector and exact search,
and whether the retry loop actually helps.
*Exit: one command produces the full report from a single run.*

## M6 — Stretch ⬜
Web UI, a hosted demo behind hard spending guardrails, and a learned reranker
evaluated against plain vector ranking. Only after M5.

---

Scope, including what is deliberately **out** of scope, is fixed in the build
spec. Notable exclusions: no prebuilt LangGraph agents, reviews are not indexed
as their own rows in v1, and no changes to Brindle from this repo.
