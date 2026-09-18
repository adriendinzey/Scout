# Scout — Evaluation

The project is not done until this produces numbers. This document is the method;
the results live in the README, drawn from one named run.

---

## 1. Rules that make the numbers mean something

- **One run set.** Every figure in a report comes from a **single named run**,
  recorded with: the Brindle commit, pgvector version, embedding model, city and
  snapshot date, machine, and the full settings object.
- **Reproducible.** One command runs the whole eval and writes the report.
- **Honest.** Losses are published alongside wins. If Brindle does not beat
  pgvector on real queries, the report says so in the same typeface.
- **No moving the goalposts.** The eval set is written before the numbers are
  seen. Queries are not removed because they score badly — a hard query that
  Scout fails is a finding.

## 2. The eval set

50–100 hand-written queries covering different filter combinations and
difficulties. Each is labeled with:

- **expected parsed filters** — for parsing accuracy;
- **a small set of relevant listing IDs** — for retrieval quality.

Labeling is assisted: a helper shows candidates from exact search so relevance can
be judged against what is actually in the corpus, rather than from imagination.

**Tricky cases are included deliberately**, not avoided:

| Kind | Why it is in the set |
|---|---|
| Disjunctions ("Hackney or Islington") | Exercises the fan-out path Brindle cannot push down |
| Unsupported constraints ("free in June") | Must be reported, not silently dropped |
| Contradictory requests ("cheap luxury penthouse under $40") | The loop should give up gracefully and say why |
| Very few true matches | The case the loop exists for — does the agent notice, and what does it do about it? |
| Quality filters over sparse data ("well reviewed") | Exposes the NULL-excludes-everything trap |

**Storage:** queries and expected filters are committed — they are original text.
**Relevance labels are gitignored**: they reference listing IDs from one snapshot,
so they are neither portable nor ours to publish.

## 3. Metrics

### Parse

- Per-field accuracy against the labeled filters.
- Rate of invalid JSON **before** the retry.
- Rate of hallucinated columns or values — inventing a neighbourhood that does not
  exist is a different and worse failure than getting a price bound wrong.

### Retrieval

Run with the **parsed filters held fixed**, so every method faces the same
predicate. Comparing methods on different predicates measures nothing.

| Method | What it is |
|---|---|
| **Brindle** | Predicate pushed into the index |
| **pgvector post-filter** | HNSW `ORDER BY ... LIMIT k`, filter applied after |
| **pgvector iterative** | `hnsw.iterative_scan = relaxed_order` |
| **Exact** | Sequential scan, same `WHERE` and `ORDER BY` — **ground truth for recall** |

Reported: **recall@10** against exact, **precision@5** against the labels, and
**p50/p95 latency split into cold and warm**.

> Cold and warm are reported separately because a Brindle connection pays a
> one-time index-decode cost. Quoting only warm latency would flatter Scout;
> quoting only cold would flatter the baselines. Both, always.

### Agent behaviour

Read from the stored run steps, so these are counts of what happened rather than
impressions of it:

- **Tool calls per query**, and **`search_listings` calls per query** — mean and
  distribution.
- **Which tools get used at all.** A tool the model never calls is either badly
  described or not worth its schema tokens.
- **How runs end:** Claude finished · tool-call limit · search limit · cost
  ceiling · timeout · error. A healthy distribution is mostly "finished"; a large
  limit share means the limits are doing the model's job for it.
- **Invalid tool arguments** — rate of calls rejected by the input model or by the
  guest-count rule, before and after the model's self-correction.
- **Repeated identical calls** — rate of calls served from the repeat cache.

### Agent versus fixed

The whole eval set is run in **both modes**, plus `--mode fixed --no-relax` as the
"no loop at all" control. Same queries, same parsed filters, same corpus:

| | share ending with ≥ 5 results | precision@5 vs labels | p50 / p95 latency | tokens / query | **USD / query** |
|---|---|---|---|---|---|
| `--mode agent` | | | | | |
| `--mode fixed` | | | | | |
| `--mode fixed --no-relax` | | | | | |

That last column is the point of the comparison. An agent that buys two points of
precision at triple the cost and double the latency is a trade-off to state, not
an unambiguous win — **and if `fixed` wins on quality and costs less, the README
says exactly that**, in the same typeface as anything favourable.

### Prompt regression (runs in CI)

A small set of queries with their expected parsed filters, checked against
**recorded** Claude responses with the backend faked — no network, no key, no
cost. A change to the Parse prompt, the grounding block, or the filter schema that
breaks the mapping fails the build rather than being discovered in an eval run
weeks later.

The same recorded-response approach covers the agent loop's shape: a scripted
sequence of tool-use turns asserts that the loop executes them, enforces its
limits, and still reaches an answer.

## 4. Keeping it cheap

The eval is the most expensive thing Scout does — 50–100 queries × several
variants, repeated whenever something changes. Three things keep that affordable:

1. **Batch API** — the eval is asynchronous by nature, so it runs at **half price**.
2. **Prompt caching** — the Parse grounding block is byte-identical across every
   query in a run.
3. **A recorded run set** — reported figures come from a saved run, so writing up
   results does not mean paying to re-run.

The harness estimates projected spend **before** it starts and refuses to exceed
`SCOUT_MAX_RUN_COST_USD`. Within a run, each query is separately capped by
`SCOUT_MAX_QUERY_COST_USD`, priced from real token counts as the loop runs — so a
single pathological query cannot eat the run's budget, and the report shows how
many queries hit the ceiling.

## 5. What the report contains

- The run manifest (§1) — without it the numbers are not reproducible.
- Each metric in §3, with the sample size beside it.
- **Total cost of the run, and cost per query**, per mode.
- Index build times and the measured per-backend memory.
- A plain-language summary including anything that went **against** the
  hypothesis.
