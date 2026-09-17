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
    model_check: str = "claude-haiku-4-5"
    model_answer: str = "claude-sonnet-5"

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
    thin_result_threshold: int = Field(default=5, ge=1)
    max_relaxations: int = Field(default=3, ge=0)
    disjunction_fanout_cap: int = Field(default=4, ge=1)

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
        and send the Check node chasing a phantom. Catch it at startup instead.
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
