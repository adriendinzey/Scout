# Data: source, terms, and the privacy rules Scout enforces

Scout searches [Inside Airbnb](https://insideairbnb.com/get-the-data/) data. This
document is the contract for how that data is obtained, what Scout keeps, and what
it deliberately throws away. **These are constraints, not preferences** — a change
here needs a deliberate decision, not a convenient refactor.

---

## 1. Getting the data

Scout ships **no data**. You download it yourself, once.

1. Go to <https://insideairbnb.com/get-the-data/> and find **London**.
2. Download two files:
   - **Detailed Listings data** — `listings.csv.gz`
   - **Detailed Review Data** — `reviews.csv.gz`
   (The smaller "summary" files are not enough: Scout needs `description`,
   `amenities`, and the `review_scores_*` columns.)
3. Put them under `data/london/` and point `.env` at them.
4. Note the **snapshot date** shown on the page into `SCOUT_SNAPSHOT_DATE`. It
   goes into the evaluation report, because listing IDs are not stable between
   snapshots and a relevance label without a date is meaningless.

The load script reads **local files only**. Nothing in Scout fetches from Inside
Airbnb at runtime, and nothing should be changed to do so — their community
guidelines ask users to download once rather than re-fetch, and to not scrape the
site.

Verify the column names against the file and the
[data dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4)
before relying on them. Inside Airbnb changes its schema between snapshots; a
column this repo assumes may have been renamed.

## 2. License and attribution

Inside Airbnb data is licensed **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**.
Attribution is required, and Scout carries it in three places: the README, the
`LICENSE` file, and any UI or demo that displays results. If you add a fourth
surface that shows listings, it needs the attribution too.

Inside Airbnb describes itself as a mission-driven housing-advocacy project.
Scout is a **guest-side search demo**. It does not build features aimed at hosts
or at optimizing listings for revenue, and it should not start.

## 3. Never commit the data

`data/` and `eval/labels/` are gitignored from the first commit. That covers:

- the raw CSVs,
- anything derived in bulk from them (database dumps, extracted tables, parquet),
- **the embeddings** — a 384-dimensional vector per listing is derived data and
  redistributing it is still redistributing the corpus,
- relevance labels, which reference listing IDs from one specific snapshot.

Individual listings and short review excerpts **may** appear in answers, the
README, and the demo recording, with attribution. The line is between
*illustrating* results and *republishing* the corpus.

## 4. Personal data: dropped at load time

These fields exist in the source files and **never reach Scout's database**. They
are dropped during parsing, before the first `INSERT` — not nulled afterwards,
not stored "just in case".

| Source column | Why it is dropped |
|---|---|
| `host_id` | Identifies a person |
| `host_name` | A personal name |
| `host_about`, `host_thumbnail_url`, `host_picture_url` | Personal profile content |
| `host_url`, `listing_url` | Resolve back to a profile |
| `host_profile_id`, `host_profile_url` | Added by a later snapshot; resolve to a profile |
| `reviewer_id` | Identifies a person |
| `reviewer_name` | A personal name |

`host_is_superhost` **is** kept. It is a property of the listing's service level,
not a personal detail, and it is a filter guests genuinely use.

### Review text

Review bodies are kept, because citing a real guest is the point of the Answer
node. But guests write names into reviews constantly ("Maria was a wonderful
host"). So:

- A best-effort scrubbing pass removes probable personal names from review text
  **both** before embedding and before display.
- It is **best-effort and must be described that way**. It is to be tested
  against a fixture of real-shaped cases, with its limitations stated in the
  README rather than quietly hoped over. Neither exists yet: see § 6.
- The scrubber is to be one tested function with one call path. If you find a
  second place review text reaches a user, route it through the same function.

### Coordinates

`latitude` and `longitude` are **already offset by Inside Airbnb** — up to ~150m —
specifically so listings cannot be pinpointed. Scout uses them for coarse
bounding-box filtering only. Do not present them as a real address, and do not
add a feature whose value depends on them being exact.

## 5. What Scout stores

See `docs/ARCHITECTURE.md` § Schema for the full table definitions. In summary:

- **`listings`** — the searchable unit: text, typed attributes, amenity booleans,
  the embedded `doc_text`, and the vector. Two columns carry a caveat the load
  found in the file rather than in the data dictionary: **`price_gbp` is in
  pounds**, the currency London's snapshot quotes despite printing a dollar
  sign, and **`instant_bookable` is NULL for every row**, because the scrape has
  stopped publishing the field. NULL is the honest value — a NULL satisfies no
  comparison, so a filter on it returns nothing rather than something invented.
- **`reviews`** — `listing_id`, `date`, `comments`. Capped at
  `SCOUT_MAX_REVIEWS_PER_LISTING` (default 5, most recent). **No reviewer name,
  no reviewer ID.**
- **Lookup tables** — neighbourhoods, room types, property types, amenities.
- **`runs` and `run_steps`** (not built yet; see § 6) — what each query did: the
  query text, the mode, the tools called, truncated input and output summaries, timings, tokens, and cost.
  A step summary can quote listing and review text, so **the scrubber runs before
  a summary is written**, not before it is displayed. The same records are written
  as one JSONL file per run under `SCOUT_TRACE_DIR`, which is gitignored — a
  trace is a local artifact and is never committed.
- **No secrets in the stored settings snapshot.** The run record keeps the
  configuration so a report can be reproduced; the API key is to be excluded, and
  a test is to assert it once the run store exists (see § 6).

### Data that leaves the machine

Two paths, both deliberate:

1. **The Anthropic API.** The query, the grounding block, tool definitions, and
   tool results — which include listing fields and scrubbed review excerpts — are
   sent to Claude. That is inherent to the product.
2. **LangSmith, only if `LANGSMITH_API_KEY` is set.** Optional in the strict
   sense: unset, nothing is imported and nothing is sent. When it is set, it
   receives the same scrubbed, truncated summaries the local trace holds — never
   more. It is a convenience view, never the record a published number is drawn
   from.

Nothing else sends data anywhere. Scout does not fetch from Inside Airbnb at
runtime, and it has no telemetry.

## 6. Enforcement

Both constraints above fail *silently*. A stray `host_name` column returns
perfectly correct rows; a committed CSV passes every other test in the suite. So
the rules are checked rather than trusted — and this section says which checks
exist **at this commit**, because an enforcement claim nothing asserts is worse
than no claim at all.

### Enforced now

`tests/unit/test_privacy_audit.py` is the audit, and it is a unit test on
purpose: it reads text rather than a database, so it runs in well under a second
with no Postgres, alongside the schema-level check below, which needs one.

- No column whose name matches the dropped-fields list exists in any Scout
  table. Asserted twice: against `information_schema` once the migrations have
  been applied (`tests/integration/test_schema.py`), and against the migration
  files themselves without a database.
- No `reviewer_*` column exists at all, whatever it is suffixed with.
- No dropped-field name appears in any string literal under `src/`, except
  docstrings and the one list that names them in order to refuse them — so SQL that selects or
  copies a personal column fails here rather than at the first `COPY`.
- The columns the loader writes contain no personal field, and
  `ListingRow`/`ReviewRow` have no field one could sit in: the parsers' own
  tests assert that, which is the load-bearing check, because a field that does
  not exist cannot be written.
- The dropped-fields table in § 4 and the list the code enforces name exactly
  the same columns, so the prose and the code cannot drift apart.
- Each of those scans is pointed at a planted violation and asserted to fire. A
  scanner that cannot fail still reads like coverage.
- `data/`, `eval/labels/` and `SCOUT_TRACE_DIR` are ignored by git — asked of
  git itself, not of a reimplementation of its rules — nothing under them is
  tracked, and no tracked file is corpus-shaped.
- The PR template keeps its privacy checkbox, and the bug-report template keeps
  its box asking reporters not to paste personal data.

### Not enforced yet

Each of these needs the thing it would test, and none of them exists yet. The
audit carries a tripwire that fails the moment the scrubber lands, so the claim
moves up into the list above in the same change rather than quietly becoming
true-but-untested.

- **The name scrubber removes names from a fixture of real-shaped review text**,
  and its false-positive and false-negative rates are published. The
  document-template step owns the scrubber, measures it against loaded review
  text, and is the only place it may be written: one function, one call path.
- **No personal name reaches the run store**, and **no API key appears in a
  run's stored settings snapshot**. The run store owns both; there is no
  `runs` table and no stored snapshot to assert against today.

If you find personal data stored anywhere in Scout, that is a **bug with priority
over feature work**, not a cleanup task.
