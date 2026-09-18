"""Typed settings, loaded once from the environment.

Every knob Scout has lives here. Nodes read settings rather than os.environ, so
a run's configuration is one inspectable object that the evaluation report can
record verbatim.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmBackend = Literal["anthropic", "fake"]

# `agent` lets Claude choose which tool to call next; `fixed` runs the hardcoded
# retrieve-then-relax path. The second exists to be measured against the first.
SearchMode = Literal["agent", "fixed"]


class Settings(BaseSettings):
    """Scout's configuration. Reads SCOUT_*, plus ANTHROPIC_API_KEY unprefixed."""

    model_config = SettingsConfigDict(
        env_prefix="SCOUT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    # Defaults to `fake` so an unconfigured checkout (and CI) can never spend
    # money by accident. Choosing `anthropic` is an explicit act.
    llm_backend: LlmBackend = "fake"
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    model_parse: str = "claude-haiku-4-5"
    model_agent: str = "claude-haiku-4-5"
    model_answer: str = "claude-sonnet-5"

    # Ceiling on a whole evaluation run, checked against projected spend before
    # the run starts. The per-query ceiling below is the one that fires mid-loop.
    max_run_cost_usd: float = Field(default=5.00, gt=0)

    # --- Database ----------------------------------------------------------
    database_url: str = "postgresql://scout:scout@localhost:5433/scout"
    # Brindle decodes a private copy of the index per backend, so pool size is a
    # memory multiplier, not just a concurrency knob. See docs/DEVELOPMENT.md.
    db_pool_size: int = Field(default=4, ge=1, le=32)

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dims: int = Field(default=384, gt=0)

    # --- Retrieval ---------------------------------------------------------
    ef_search: int = Field(default=100, ge=1, le=10_000)
    candidate_pool: int = Field(default=100, ge=1)

    # --- Agent loop --------------------------------------------------------
    # The model decides what to do; these decide what it is allowed to do. Each
    # one is enforced in code and has a test proving it fires, because a limit
    # that lives only in a prompt is a suggestion.
    mode: SearchMode = "agent"
    max_tool_calls: int = Field(default=10, ge=1)
    max_searches: int = Field(default=4, ge=1)
    max_query_cost_usd: float = Field(default=0.10, gt=0)
    query_timeout_s: float = Field(default=60.0, gt=0)

    # --- Fixed mode --------------------------------------------------------
    # The hardcoded baseline the agent is compared against: search, and if fewer
    # than `thin_result_threshold` rows come back, loosen one filter by rule and
    # search again, at most `max_relaxations` times.
    thin_result_threshold: int = Field(default=5, ge=1)
    max_relaxations: int = Field(default=3, ge=0)

    disjunction_fanout_cap: int = Field(default=4, ge=1)

    # --- Tracing -----------------------------------------------------------
    # Every run writes a JSONL file here as well as rows in the database, so a
    # trace survives a database that has been torn down and rebuilt.
    trace_dir: Path = Path("runs")
    # Published token prices change. The built-in table is the default; point
    # this at a TOML file to correct prices without editing code.
    price_table: Path | None = None
    # Optional, off unless the key is present: Scout is fully usable, and every
    # test passes, without it.
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")

    # --- Data --------------------------------------------------------------
    listings_csv: Path = Path("data/london/listings.csv.gz")
    reviews_csv: Path = Path("data/london/reviews.csv.gz")
    city: str = "London"
    snapshot_date: str | None = None
    max_reviews_per_listing: int = Field(default=5, ge=0)

    @field_validator("candidate_pool")
    @classmethod
    def _pool_must_fit_ef_search(cls, pool: int, info: object) -> int:
        """A ranked Brindle scan returns at most ef_search rows.

        Asking for a larger candidate pool does not error — it silently returns
        fewer rows than requested, which would look like a thin-result problem
        and send the loop chasing a phantom. Catch it at startup instead.
        """
        data = getattr(info, "data", {})
        ef = data.get("ef_search")
        if isinstance(ef, int) and pool > ef:
            raise ValueError(
                f"candidate_pool ({pool}) exceeds ef_search ({ef}); a ranked "
                f"Brindle scan returns at most ef_search rows, so the extra "
                f"would be silently dropped. Raise SCOUT_EF_SEARCH."
            )
        return pool

    @field_validator("max_searches")
    @classmethod
    def _searches_must_fit_tool_budget(cls, searches: int, info: object) -> int:
        """The search cap has to bind before the tool-call cap does.

        A search budget larger than the total tool budget is dead configuration:
        the loop would always stop on tool calls first, and the eval's
        "searches per query" column would be measuring the wrong limit.
        """
        data = getattr(info, "data", {})
        calls = data.get("max_tool_calls")
        if isinstance(calls, int) and searches > calls:
            raise ValueError(
                f"max_searches ({searches}) exceeds max_tool_calls ({calls}); "
                f"the search limit could never fire. Raise SCOUT_MAX_TOOL_CALLS."
            )
        return searches

    def require_anthropic_key(self) -> SecretStr:
        """Return the API key, or explain precisely what is missing."""
        if self.anthropic_api_key is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set but SCOUT_LLM_BACKEND=anthropic. "
                "Set the key, or use SCOUT_LLM_BACKEND=fake for offline work."
            )
        return self.anthropic_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached so one run reports one configuration."""
    return Settings()
