"""The boundary's contract: its shape, its layering rule, and how it is chosen."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import SecretStr

from scout.config import Settings
from scout.llm import build_backend
from scout.llm.backend import (
    FINISHED_STOP_REASONS,
    LlmBackend,
    LlmEmptyResponseError,
    LlmError,
    LlmRefusalError,
    LlmRequest,
    LlmResponse,
    LlmResponseError,
    LlmSchemaError,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    decode_json_output,
    structured_output,
)
from scout.llm.fake_backend import FakeBackend
from scout.llm.pricing import Usage

SRC = Path(__file__).resolve().parents[2] / "src" / "scout"

#: The one module allowed to know the SDK exists.
SDK_FACING = {"anthropic_backend.py"}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_only_one_module_imports_the_sdk() -> None:
    """Nodes take a protocol. The SDK stops at the boundary.

    Walks the whole package rather than a hand-kept list, so a future node that
    reaches for `anthropic` directly fails here instead of in review.
    """
    offenders = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.name not in SDK_FACING and "anthropic" in imported_modules(path)
    )

    assert offenders == []


def test_the_fake_path_does_not_pull_in_the_sdk() -> None:
    """CI installs the SDK, but the default path must not need to import it."""
    for module in ("backend.py", "fake_backend.py", "usage.py", "pricing.py"):
        assert "anthropic" not in imported_modules(SRC / "llm" / module)


# --- Choosing a backend ----------------------------------------------------


def test_the_default_backend_cannot_spend_money() -> None:
    backend = build_backend(Settings(_env_file=None))

    assert isinstance(backend, FakeBackend)


def test_the_real_backend_is_chosen_deliberately() -> None:
    """Constructing the client sends nothing; the key is never used here."""
    settings = Settings(
        _env_file=None, llm_backend="anthropic", anthropic_api_key=SecretStr("sk-ant-not-real")
    )

    backend = build_backend(settings)

    assert type(backend).__name__ == "AnthropicBackend"


def test_asking_for_the_real_backend_without_a_key_explains_itself() -> None:
    settings = Settings(_env_file=None, llm_backend="anthropic", anthropic_api_key=None)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_backend(settings)


def test_both_backends_satisfy_the_protocol() -> None:
    settings = Settings(
        _env_file=None, llm_backend="anthropic", anthropic_api_key=SecretStr("sk-ant-not-real")
    )

    assert isinstance(build_backend(Settings(_env_file=None)), LlmBackend)
    assert isinstance(build_backend(settings), LlmBackend)


# --- Request and response shapes -------------------------------------------


def test_a_request_with_the_same_prefix_compares_equal() -> None:
    """How a test proves the cached prefix did not drift."""
    first = LlmRequest(model="m", system="grounding", messages=(Message.user("a"),))
    second = LlmRequest(model="m", system="grounding", messages=(Message.user("b"),))

    assert first.system == second.system
    assert first != second


def test_tool_results_go_back_as_one_user_turn() -> None:
    message = Message.tool_results(
        ToolResultBlock("toolu_1", "{}"), ToolResultBlock("toolu_2", "{}")
    )

    assert message.role == "user"
    assert len(message.content) == 2


def test_a_response_separates_prose_from_tool_calls() -> None:
    response = LlmResponse(
        model="m",
        content=(
            TextBlock("Checking the count first."),
            ToolUseBlock("toolu_1", "count_matches", {"max_price": 150}),
        ),
        stop_reason="tool_use",
        usage=Usage(),
    )

    assert response.text == "Checking the count first."
    assert [call.name for call in response.tool_uses] == ["count_matches"]
    assert response.wants_tools
    assert not response.finished


def test_a_response_becomes_the_next_turns_assistant_message() -> None:
    blocks = (ThinkingBlock("why", "sig"), ToolUseBlock("toolu_1", "field_stats", {}))
    response = LlmResponse(model="m", content=blocks, stop_reason="tool_use", usage=Usage())

    assert response.as_message() == Message(role="assistant", content=blocks)


def test_only_a_natural_stop_counts_as_finished() -> None:
    """Everything else has to be reported in the answer, so it must not read as done."""
    assert set(FINISHED_STOP_REASONS) == {"end_turn", "stop_sequence"}
    for stop_reason in ("tool_use", "max_tokens", "pause_turn", "model_context_window_exceeded"):
        response = LlmResponse(
            model="m",
            content=(TextBlock("partial"),),
            stop_reason=stop_reason,  # type: ignore[arg-type]
            usage=Usage(),
        )
        assert not response.finished


# --- Failures are never an empty result ------------------------------------


def test_every_boundary_failure_shares_one_catchable_base() -> None:
    for error in (LlmSchemaError, LlmRefusalError, LlmEmptyResponseError, LlmResponseError):
        assert issubclass(error, LlmError)


def test_a_schema_error_carries_what_the_model_actually_said() -> None:
    with pytest.raises(LlmSchemaError) as caught:
        decode_json_output("about two hundred pounds")

    assert caught.value.raw_output == "about two hundred pounds"


def test_an_empty_body_is_a_schema_error_not_an_empty_dict() -> None:
    with pytest.raises(LlmSchemaError, match="empty body"):
        decode_json_output("   ")


def test_a_valid_body_decodes() -> None:
    assert decode_json_output('{"max_price": 200}') == {"max_price": 200}


def test_a_truncated_body_is_not_treated_as_a_schema_mistake() -> None:
    """The retry-once path feeds the error back and asks again; that would
    truncate in exactly the same place."""
    with pytest.raises(LlmResponseError, match="max_tokens") as caught:
        structured_output(
            '{"max_price": 2',
            stop_reason="max_tokens",
            model="claude-haiku-4-5",
            usage=Usage(input_tokens=10, output_tokens=4096),
        )

    assert not isinstance(caught.value, LlmSchemaError)


def test_a_failed_response_still_reports_what_it_cost() -> None:
    """A refused or unusable turn is billed like any other."""
    usage = Usage(input_tokens=900, output_tokens=12)
    with pytest.raises(LlmSchemaError) as caught:
        structured_output(
            "sorry, no", stop_reason="end_turn", model="claude-haiku-4-5", usage=usage
        )

    assert caught.value.usage == usage
    assert caught.value.model == "claude-haiku-4-5"
