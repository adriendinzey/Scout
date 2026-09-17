# Postgres 17 carrying both extensions Scout measures against each other:
#   - Brindle  (filter-aware vector index, built from a pinned commit)
#   - pgvector (the baseline: HNSW post-filter and iterative scan)
#
# Both live in ONE database so the retrieval comparison in docs/EVALUATION.md
# runs over identical rows — a baseline on a different copy of the data proves
# nothing.
#
# Builder and runtime share the same postgres:17-bookworm base so the compiled
# .so matches the server ABI exactly.

# ---------------------------------------------------------------- builder ---
FROM postgres:18-bookworm AS builder

# Pinned deliberately. Scout's README reports which Brindle commit produced its
# numbers; bumping this is a decision with an eval re-run attached, not a chore.
ARG BRINDLE_REPO=https://github.com/adriendinzey/Brindle.git
ARG BRINDLE_REF=025b1315f40a1384695d81ff846bfc8c104c5aea
ARG PGVECTOR_REF=v0.8.0
# Must match the `pgrx = "=x.y.z"` pin in Brindle's Cargo.toml at BRINDLE_REF.
ARG PGRX_VERSION=0.12.9

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_TERM_COLOR=always

RUN apt-get update && apt-get install -y --no-install-recommends \
        postgresql-server-dev-17 \
        build-essential clang libclang-dev pkg-config \
        libssl-dev libreadline-dev zlib1g-dev \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# --- pgvector (plain C, quick) ---
RUN git clone --depth 1 --branch "${PGVECTOR_REF}" \
        https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make OPTFLAGS="" \
    && make install DESTDIR=/out

# --- Rust toolchain for pgrx ---
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

RUN cargo install --locked cargo-pgrx --version "${PGRX_VERSION}"

# Point pgrx at the image's own Postgres rather than letting it compile one.
RUN cargo pgrx init --pg17 /usr/lib/postgresql/17/bin/pg_config

# --- Brindle ---
# A full clone, because a pinned SHA is generally not a branch tip and cannot be
# reached by `clone --depth 1 --branch`.
RUN git clone "${BRINDLE_REPO}" /tmp/brindle \
    && cd /tmp/brindle \
    && git checkout --quiet "${BRINDLE_REF}" \
    && git rev-parse HEAD > /brindle-commit.txt

WORKDIR /tmp/brindle
RUN cargo pgrx package --pg-config /usr/lib/postgresql/17/bin/pg_config

# `cargo pgrx package` emits a tree rooted at / under target/release/brindle-pg17.
# Fold it into the same /out prefix pgvector used so the runtime has one COPY.
RUN cp -a target/release/brindle-pg17/. /out/ \
    && ls -la /out/usr/lib/postgresql/17/lib/ \
    && ls -la /out/usr/share/postgresql/17/extension/

# ---------------------------------------------------------------- runtime ---
FROM postgres:18-bookworm

COPY --from=builder /out/usr/lib/postgresql/17/lib/     /usr/lib/postgresql/17/lib/
COPY --from=builder /out/usr/share/postgresql/17/extension/ /usr/share/postgresql/17/extension/
# Recorded so a running container can state exactly which Brindle it carries —
# the README's pinned-commit claim is verifiable from the artifact itself.
COPY --from=builder /brindle-commit.txt /etc/brindle-commit.txt

# Runs once, on an empty data directory, as part of the official image's initdb.
COPY initdb/ /docker-entrypoint-initdb.d/
