# Scout — Development

## 1. Where the working tree lives

**Work in WSL2 on the Linux-native filesystem** — `~/code/scout`, not under
`/mnt/c` or `/mnt/d`.

Two reasons, both of which bite hard:

- **Docker.** The Compose Postgres runs in WSL. A bind mount reaching a Windows
  drive goes over the 9P protocol and is dramatically slower, and the `initdb`
  scripts need Unix permissions and LF line endings.
- **Python.** `uv sync` writes thousands of small files into `.venv`. On `/mnt/*`
  that is minutes instead of seconds.

`.gitattributes` forces LF in the repo, which keeps the shell scripts and
`initdb` SQL working regardless of what checks them out.

## 2. First-time setup

```bash
# Toolchain
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv (Python + venv manager)
# Docker: use Docker Desktop with the WSL2 backend, or docker-ce inside WSL.

git clone https://github.com/adriendinzey/scout.git ~/code/scout
cd ~/code/scout

uv sync --all-extras       # --all-extras pulls torch for the embedder (~2 GB)
cp .env.example .env       # then add your ANTHROPIC_API_KEY

docker compose up -d --wait   # first build compiles Brindle; takes a few minutes
uv run scout doctor           # confirms Postgres + both extensions
```

`--all-extras` is only needed for `scout data embed`. For everything else
`uv sync` is enough and much faster — which is why CI does not use it.

## 3. The daily loop

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy                              # strict, on src/
uv run pytest -m "not integration"       # fast; no database needed
uv run pytest                            # everything; needs the database up
uv run scout ask "..."                   # the actual product
```

Green on all four is the bar for opening a PR. CI runs exactly these.

## 4. The database image

`docker/Dockerfile` builds one Postgres 17 image carrying **both** extensions:

- **Brindle**, compiled with `cargo-pgrx` from a **pinned commit** (`BRINDLE_REF`).
- **pgvector**, the evaluation baseline, pinned by tag.

They share one database on purpose: comparing retrieval methods across different
copies of the data would prove nothing.

The running container records which Brindle it carries:

```bash
docker compose exec db cat /etc/brindle-commit.txt
```

### Bumping Brindle

`BRINDLE_REF` is a deliberate decision, not a chore — Dependabot is configured
not to touch it. Changing it invalidates every published number, so it comes with
an eval re-run and a README update. If Scout needs something Brindle cannot do,
**open an issue on Brindle** rather than working around it here; this repo never
modifies Brindle.

## 5. Memory: the constraint that surprises people

Brindle decodes its own copy of the index **per backend connection**, bounded by
`brindle.cache_max_mb` (default 256 MB).

Measured in Brindle: **887 bytes/node at 128 dimensions.** Scout embeds at 384
dimensions, so the vector alone is 1536 bytes, and per node it works out to
roughly **1.9 KB**:

| Listings | Index per backend | 4 connections |
|---|---|---|
| 40,000 | ~76 MB | ~305 MB |
| **90,000 (London)** | **~172 MB** | **~688 MB** |
| 150,000 | ~285 MB | over the default cap — every query re-decodes |

**So pool size is a memory multiplier, not just a concurrency knob.** The default
`SCOUT_DB_POOL_SIZE` is 4 for exactly this reason. Above `cache_max_mb`, the cache
stops helping and every query decodes from scratch — which shows up as latency
that looks like cold-start on every single query.

This is also why the evaluation reports **cold and warm latency separately**: a
fresh connection pays the decode once, and quoting only warm numbers would
flatter Scout.

## 6. Parallel development with git worktrees

Several agents or sessions must **never share one working directory** — they will
stomp on each other's uncommitted files, and two branches cannot be checked out in
one directory anyway.

```bash
scripts/worktree.sh new T-021-filter-sql   # own branch, dir, venv, and Postgres
cd ../scout-wt/T-021-filter-sql
uv sync
scripts/db.sh up
scripts/worktree.sh ls                     # what's active, and on which port
scripts/worktree.sh rm T-021-filter-sql    # after the PR merges
```

Each sandbox gets:

| Isolated | How |
|---|---|
| Directory + branch | `git worktree` |
| Python environment | `uv` creates a `.venv` per directory automatically |
| Postgres + its data | its own Compose **project name** and a free host port |
| Scout's config | a generated `.env` pointing at that port, with `SCOUT_LLM_BACKEND=fake` |

That last one matters: a parallel task defaults to the fake LLM backend, so
running five sandboxes at once cannot quietly run up an API bill.

Run database commands through `scripts/db.sh`, which sources the sandbox's
settings. Plain `uv run ...` needs no wrapper.

> If you drive sub-agents via the Claude Code Agent tool, `isolation: "worktree"`
> sets the worktree up for you — but the database isolation still comes from
> `scripts/worktree.sh` / `scripts/db.sh`.

## 7. Working offline, and not spending money

`SCOUT_LLM_BACKEND=fake` runs the whole graph against deterministic canned
responses. It is the default, so an unconfigured checkout cannot call a paid API
by accident, and **no test calls the real Anthropic API** — in CI or locally.

Switch to `anthropic` deliberately, for runs whose output you intend to publish.
`SCOUT_MAX_RUN_COST_USD` refuses to start a run whose projected spend exceeds it.

## 8. Branch protection

`main` is protected: changes land only via PR, required checks must pass, the
branch must be up to date, history stays linear (rebase, not merge), and
force-pushes and deletion are blocked. The maintainer merges.
