# Contributing to Scout

Scout is an early-stage project built in the open — issues and pull requests are
welcome.

## Building and testing

Scout is a Python 3.12 project managed with [`uv`](https://docs.astral.sh/uv/),
searching a PostgreSQL 17 database that carries both
[Brindle](https://github.com/adriendinzey/Brindle) and
[pgvector](https://github.com/pgvector/pgvector). Develop on **Linux, WSL2, or
macOS**. On Windows, work inside WSL2 on the Linux-native filesystem (`~/code/scout`,
not `/mnt/*`); full setup and the parallel-worktree workflow are in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

```bash
docker compose up -d --wait          # Postgres 17 + Brindle + pgvector
uv sync                              # add --all-extras for the embedder (torch)
cp .env.example .env
uv run scout doctor                  # is the stack actually working?
```

The daily loop — CI runs exactly these:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy                          # strict, on src/
uv run pytest -m "not integration"   # fast; no database needed
uv run pytest                        # everything; needs the database up
```

## Project layout

Pure logic — filter-to-SQL mapping, amenity and price parsing, the document
template, the relaxation rules — lives in functions with no I/O, so it is
unit-testable without a database or an API key. The LangGraph nodes wire those
together; the LLM sits behind an interface with a fake implementation. Please keep
new logic in the pure layer wherever it can live there.

Design rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Conventions: [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md).

## Three rules with no exceptions

**1. Never commit Inside Airbnb data.** Not the CSVs, not a database dump, not
derived tables, **not the embeddings**. `data/` and `eval/labels/` are gitignored
from the first commit. See [docs/DATA.md](docs/DATA.md).

**2. Never store or display personal data.** Host names and IDs, reviewer names
and IDs, and host profile text are dropped at load time — before the first
`INSERT`, not nulled afterwards. Review text passes through the scrubbing function
before it is embedded or shown. If you add a second path that displays review
text, route it through that same function.

**3. No test calls the real Anthropic API.** Not in CI, not by default locally.
`SCOUT_LLM_BACKEND=fake` is the default for exactly this reason. Use recorded
fixtures or the fake backend.

## Contribution flow

1. **Open (or pick) an issue** describing the bug or feature.
2. **Branch off `main`** and keep the change focused: one logical change per PR.
3. **Add tests** for new behavior and run the loop above locally.
4. **Open a pull request** and fill in the template. The *files touched* list
   matters — it is how overlap between in-flight PRs gets spotted early. Paste the
   actual verification output; "should work" is not verification.
5. **CI must be green.** `main` is protected: changes land only via PR, required
   checks must pass, the branch must be up to date (rebase rather than merge —
   linear history is enforced), and force-pushes and deletion of `main` are
   blocked. The maintainer does the merging.

## Claims and honesty

Scout's value is that its numbers are trustworthy. So:

- A performance or quality claim needs a reproducible number behind it, from a
  single named run, not an impression.
- Losses get published next to wins. If a change makes something worse, the PR
  says so.
- If the eval set needs to change, change it **before** seeing the new numbers,
  and say why in the PR.

## Changing Brindle's pin

`BRINDLE_REF` in `docker/Dockerfile` is pinned deliberately, and Dependabot is
configured not to touch it. Bumping it invalidates every published number, so it
comes with an eval re-run and a README update in the same PR.

If Scout needs something Brindle cannot do, **open an issue on
[Brindle](https://github.com/adriendinzey/Brindle/issues)** describing the gap.
This repository never modifies Brindle.

## Commit messages

`type(scope): summary`, where `type` is one of
`feat | fix | test | docs | refactor | chore | perf | ci` — for example
`feat(graph): explicit tool loop with enforced budgets`. Add a body when
the change needs context.

## Changelog

User-visible changes get a line under **Unreleased** in
[CHANGELOG.md](CHANGELOG.md).

## License

Scout is released under the [MIT License](LICENSE). By contributing, you agree
that your contributions are licensed under it.
