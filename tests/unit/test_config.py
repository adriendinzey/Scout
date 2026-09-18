"""Settings behave, and the footguns the spec warns about are caught at startup."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scout.config import Settings


def test_defaults_do_not_spend_money() -> None:
    """An unconfigured checkout must not be able to call a paid API."""
    settings = Settings(_env_file=None)
    assert settings.llm_backend == "fake"


def test_candidate_pool_above_ef_search_is_rejected() -> None:
    """Brindle returns at most ef_search rows from a ranked scan.

    Asking for more does not error at the database — it quietly returns fewer
    rows, which the agent would misread as a genuinely thin result.
    """
    with pytest.raises(ValidationError, match="exceeds ef_search"):
        Settings(_env_file=None, ef_search=64, candidate_pool=100)


def test_candidate_pool_equal_to_ef_search_is_allowed() -> None:
    settings = Settings(_env_file=None, ef_search=100, candidate_pool=100)
    assert settings.candidate_pool == 100


def test_missing_api_key_explains_itself() -> None:
    settings = Settings(_env_file=None, llm_backend="anthropic", anthropic_api_key=None)
    with pytest.raises(RuntimeError, match="SCOUT_LLM_BACKEND=fake"):
        settings.require_anthropic_key()


def test_ef_search_is_bounded_by_what_brindle_accepts() -> None:
    """brindle.ef_search has a hard range of 1..10000 in the extension."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ef_search=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ef_search=20_000)
