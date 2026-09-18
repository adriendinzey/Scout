"""A deterministic stand-in for Claude, with no network and no key.

This is the default backend, so an unconfigured checkout cannot spend money, and
it is what every test and CI run uses. It exists to make the *graph* testable:
because it can script a sequence of tool-use turns, a whole agent loop — the
model asking for ``count_matches``, reading the count, asking for
``search_listings`` with looser filters, then answering — runs offline and
identically every time.

Two ways to control it, both deterministic:

* **A script** — a queue of turns handed back in order. This is the common case
  and reads like the conversation it stands for.
* **A router** — a function from the request to a turn, for tests that need to
  branch on what the model was actually asked.

Running off the end of a script raises. Handing back a canned "I'm done" turn
instead would quietly convert a test-authoring mistake into a passing test of the
wrong thing, which is the same failure this whole layer exists to prevent.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from scout.llm.backend import (
    ContentBlock,
    LlmEmptyResponseError,
    LlmError,
    LlmRefusalError,
    LlmRequest,
    LlmResponse,
    StopReason,
    TextBlock,
    ToolUseBlock,
    structured_output,
)
from scout.llm.pricing import Usage

#: Non-zero on purpose. A fake that reported no tokens would let a broken cost
#: accumulator pass every test, and the cost ceiling is a limit Scout claims to
#: enforce. Tests that care about a specific figure pass their own.
DEFAULT_FAKE_USAGE = Usage(input_tokens=100, output_tokens=25)

#: A scripted tool call may leave its id blank; the backend fills in a stable,
#: per-instance one so a test need not invent ids it does not care about.
UNASSIGNED_TOOL_USE_ID = ""


class ScriptExhaustedError(RuntimeError):
    """The loop asked for one more turn than the test scripted.

    Deliberately *not* an :class:`~scout.llm.backend.LlmError`: this is a bug in
    the test, and a node's ``except LlmError`` must not swallow it.
    """


@dataclass(frozen=True, slots=True)
class FakeTurn:
    """One scripted assistant turn."""

    content: tuple[ContentBlock, ...]
    stop_reason: StopReason = "end_turn"
    usage: Usage = DEFAULT_FAKE_USAGE
    #: Set for a refusal so the backend can report the category the API would.
    refusal_category: str | None = None
    #: Raised instead of answering. Scripts a transport failure so the retry
    #: paths the boundary promises can be driven offline like any other turn.
    error: LlmError | None = None


def text_turn(text: str, *, usage: Usage = DEFAULT_FAKE_USAGE) -> FakeTurn:
    """The model answering in prose and stopping."""
    return FakeTurn(content=(TextBlock(text),), stop_reason="end_turn", usage=usage)


def json_turn(payload: Mapping[str, Any] | str, *, usage: Usage = DEFAULT_FAKE_USAGE) -> FakeTurn:
    """The model answering a schema-constrained request.

    A mapping is serialised with sorted keys so the same payload always produces
    the same bytes. Pass a string to script malformed output on purpose.
    """
    body = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
    return FakeTurn(content=(TextBlock(body),), stop_reason="end_turn", usage=usage)


def tool_use_turn(
    name: str,
    arguments: Mapping[str, Any],
    *,
    tool_use_id: str = UNASSIGNED_TOOL_USE_ID,
    text: str | None = None,
    usage: Usage = DEFAULT_FAKE_USAGE,
) -> FakeTurn:
    """The model asking for one tool.

    ``text`` scripts the sentence the model says alongside the call, which the
    trace records.
    """
    blocks: list[ContentBlock] = []
    if text is not None:
        blocks.append(TextBlock(text))
    blocks.append(ToolUseBlock(id=tool_use_id, name=name, arguments=dict(arguments)))
    return FakeTurn(content=tuple(blocks), stop_reason="tool_use", usage=usage)


def refusal_turn(*, category: str | None = None, usage: Usage = DEFAULT_FAKE_USAGE) -> FakeTurn:
    """The model declining. Raises :class:`LlmRefusalError` when consumed."""
    return FakeTurn(
        content=(TextBlock(""),),
        stop_reason="refusal",
        usage=usage,
        refusal_category=category,
    )


def empty_turn(*, usage: Usage = DEFAULT_FAKE_USAGE) -> FakeTurn:
    """A turn with no content. Raises :class:`LlmEmptyResponseError` when consumed."""
    return FakeTurn(content=(), stop_reason="end_turn", usage=usage)


def error_turn(error: LlmError) -> FakeTurn:
    """A call that fails instead of answering.

    Usually an :class:`~scout.llm.backend.LlmApiError` — the transport failure a
    caller is meant to retry. Nothing is billed, so no usage is reported.
    """
    return FakeTurn(content=(), usage=Usage(), error=error)


class FakeBackend:
    """A scripted :class:`~scout.llm.backend.LlmBackend`.

    Every request is recorded in :attr:`calls`, which is how a test asserts that
    the cacheable prefix did not move and that tool results went back with the
    right ids.
    """

    def __init__(
        self,
        turns: Iterable[FakeTurn] = (),
        *,
        router: Callable[[LlmRequest], FakeTurn] | None = None,
    ) -> None:
        scripted = list(turns)
        if scripted and router is not None:
            raise ValueError("pass a script or a router, not both: they would disagree")
        self.turns: list[FakeTurn] = scripted
        self.router = router
        self.calls: list[LlmRequest] = []
        self._next_tool_use_id = 0

    def script(self, *turns: FakeTurn) -> None:
        """Append turns to the script."""
        self.turns.extend(turns)

    @property
    def pending(self) -> int:
        """Scripted turns not yet consumed. Zero at the end of a well-drawn test."""
        return len(self.turns)

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Hand back the next scripted turn, applying the protocol's contract."""
        self.calls.append(request)
        turn = self._next_turn(request)
        if turn.error is not None:
            raise turn.error
        content = tuple(self._assign_ids(block) for block in turn.content)

        if turn.stop_reason == "refusal":
            raise LlmRefusalError(
                "the model declined the request",
                category=turn.refusal_category,
                model=request.model,
                usage=turn.usage,
            )
        if not content:
            raise LlmEmptyResponseError(
                "the model returned no content blocks",
                model=request.model,
                usage=turn.usage,
            )

        json_output = None
        if request.response_schema is not None:
            json_output = structured_output(
                "\n".join(b.text for b in content if isinstance(b, TextBlock)),
                stop_reason=turn.stop_reason,
                model=request.model,
                usage=turn.usage,
            )

        return LlmResponse(
            model=request.model,
            content=content,
            stop_reason=turn.stop_reason,
            usage=turn.usage,
            json_output=json_output,
        )

    def _next_turn(self, request: LlmRequest) -> FakeTurn:
        if self.router is not None:
            return self.router(request)
        if not self.turns:
            raise ScriptExhaustedError(
                f"call {len(self.calls)} to model {request.model!r} has no scripted turn; "
                f"script one more, or the code under test is looping further than expected"
            )
        return self.turns.pop(0)

    def _assign_ids(self, block: ContentBlock) -> ContentBlock:
        if not isinstance(block, ToolUseBlock) or block.id != UNASSIGNED_TOOL_USE_ID:
            return block
        self._next_tool_use_id += 1
        return replace(block, id=f"toolu_fake_{self._next_tool_use_id:04d}")
