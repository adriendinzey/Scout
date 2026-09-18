"""The one interface every node uses to reach Claude.

Nodes never ``import anthropic``. They take an :class:`LlmBackend` and call
:meth:`LlmBackend.complete`, so the same graph runs against the real API and
against :class:`~scout.llm.fake_backend.FakeBackend`. That is what keeps CI free
and makes a whole agent loop reproducible offline.

**One call shape, not two.** ``complete`` takes a message list that may already
contain ``tool_use`` and ``tool_result`` blocks and returns the assistant turn
plus its stop reason. Parse is the one-turn case of the same call — it passes no
tools. A single-shot "prompt in, text out" interface would have to be thrown away
the moment the agent node lands.

**Structured output is requested with the API's JSON-schema response format**
(``output_config.format``), not by asking for JSON in prose and not by forcing a
single-tool call. Prose is never parsed: when a caller passes a
``response_schema``, the backend hands back :attr:`LlmResponse.json_output`
already decoded, or raises :class:`LlmSchemaError`. A forced tool call would also
work, but the response format is enforced by the schema compiler rather than by
the model's goodwill, and it leaves ``tool_choice`` free for the agent loop.

**No failure here is reported as an empty result.** An API error, a refusal, and
a response that does not match the requested schema each raise a distinct,
catchable exception. The agent cannot tell "no listings matched" from "the call
failed", and would start relaxing filters to fix a broken network.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from scout.llm.pricing import Usage

# Generous enough for a parse, a tool call, or a short cited answer without
# truncating mid-JSON; a caller that needs more passes its own. Note that on a
# model with thinking on, reasoning is drawn from the same budget — a node using
# one should raise this rather than discover it as a `max_tokens` stop reason.
DEFAULT_MAX_TOKENS = 4096

# Mirrors the API's own set. Scout does not collapse these into a bool, because
# "the model finished" and "the model ran out of room" must be told apart: a
# truncated search that reads like a complete one is the worst output Scout can
# produce, and the Answer node has to say which happened.
StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "pause_turn",
    "refusal",
    "model_context_window_exceeded",
]

#: Stop reasons that mean the model said what it wanted to say.
FINISHED_STOP_REASONS: frozenset[str] = frozenset({"end_turn", "stop_sequence"})


# --- Errors ----------------------------------------------------------------


class LlmError(Exception):
    """Base for every failure at the LLM boundary."""


class LlmApiError(LlmError):
    """The call itself failed: network, timeout, rate limit, 5xx, bad key.

    ``retryable`` distinguishes "try again" (429, 5xx, connection reset) from
    "this request will never work" (400, 401, 404), so a caller can back off
    without inspecting status codes it should not have to know about.
    """

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class LlmResponseError(LlmError):
    """The call succeeded but the model returned something Scout cannot use.

    Carries what the call cost. A refused or unusable response is billed like any
    other, and dropping its tokens here would understate every run and let the
    per-query cost ceiling be quietly overrun by exactly the calls that went
    wrong most often.
    """

    def __init__(
        self, message: str, *, model: str | None = None, usage: Usage | None = None
    ) -> None:
        super().__init__(message)
        self.model = model
        self.usage = usage


class LlmSchemaError(LlmResponseError):
    """Output did not match the requested schema, or was not valid JSON.

    Carries the raw text so a caller can retry once with the failure fed back to
    the model, and so the trace records what the model actually said.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_output: str,
        model: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        super().__init__(message, model=model, usage=usage)
        self.raw_output = raw_output


class LlmRefusalError(LlmResponseError):
    """The model declined the request.

    A refusal is not an empty result. It ends the run with a stated reason.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str | None = None,
        model: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        super().__init__(message, model=model, usage=usage)
        self.category = category


class LlmEmptyResponseError(LlmResponseError):
    """The assistant turn contained no content blocks at all."""


# --- Content blocks --------------------------------------------------------
#
# Scout models these itself rather than passing the SDK's block objects around,
# because the whole point of this layer is that nothing above it knows the
# Anthropic SDK exists. The translation lives in `anthropic_backend`.


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A span of assistant prose."""

    text: str


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """Reasoning the model returned.

    Kept verbatim — signature included — because a thinking block replayed into a
    later turn of the same conversation must be byte-identical to the one the
    model produced.
    """

    thinking: str
    signature: str


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """The model asking for a tool, with the arguments it chose."""

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """What running that tool produced, on its way back to the model.

    ``is_error`` carries a structured tool error — a bad filter, an unknown
    field — which the model is expected to read and correct. It is not how a
    system failure is reported; those raise.
    """

    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of the conversation."""

    role: Role
    content: tuple[ContentBlock, ...]

    @staticmethod
    def user(text: str) -> Message:
        """A plain user turn."""
        return Message(role="user", content=(TextBlock(text),))

    @staticmethod
    def assistant(*blocks: ContentBlock) -> Message:
        """An assistant turn, usually :meth:`LlmResponse.as_message`."""
        return Message(role="assistant", content=tuple(blocks))

    @staticmethod
    def tool_results(*results: ToolResultBlock) -> Message:
        """Tool results go back as a *user* turn, all of them in one message.

        Splitting results across several messages teaches the model to stop
        asking for tools in parallel.
        """
        return Message(role="user", content=tuple(results))


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool offered to the model.

    ``input_schema`` is generated from the tool's Pydantic input model, so the
    schema the model sees and the validation the code applies cannot drift.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]


# --- The call --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LlmRequest:
    """Everything one call needs, as a single value.

    A request is a value rather than a pile of keyword arguments so that it can
    be recorded, compared, and asserted on. The prompt cache is a *prefix* match:
    any byte change in ``system`` or ``tools`` silently invalidates it and the
    only symptom is a bigger bill, so a test that builds the request twice and
    compares is the cheap way to catch that.
    """

    model: str
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    response_schema: Mapping[str, Any] | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    # The stable prefix — system prompt, grounding block, tool definitions — is
    # cached; the user's query sits after it in `messages` and never is. Turn
    # this off only to measure what caching is worth.
    cache_prefix: bool = True


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """One assistant turn, with what it cost and why it stopped."""

    model: str
    content: tuple[ContentBlock, ...]
    stop_reason: StopReason
    usage: Usage
    #: Decoded response body, present only when the request carried a schema.
    json_output: Mapping[str, Any] | None = None

    @property
    def text(self) -> str:
        """Every text block, joined. Empty when the turn was only a tool call."""
        return "\n".join(block.text for block in self.content if isinstance(block, TextBlock))

    @property
    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        """The tools the model asked for, in the order it asked."""
        return tuple(block for block in self.content if isinstance(block, ToolUseBlock))

    @property
    def wants_tools(self) -> bool:
        """True when the model asked for a tool rather than finishing."""
        return self.stop_reason == "tool_use"

    @property
    def finished(self) -> bool:
        """True when the model stopped because it had said its piece."""
        return self.stop_reason in FINISHED_STOP_REASONS

    def as_message(self) -> Message:
        """This turn, ready to append to the next request's message list."""
        return Message(role="assistant", content=self.content)


@runtime_checkable
class LlmBackend(Protocol):
    """What a node needs from Claude, and nothing else."""

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Run one turn.

        Raises:
            LlmApiError: the call failed; ``retryable`` says whether to try again.
            LlmRefusalError: the model declined.
            LlmEmptyResponseError: the turn came back with no content.
            LlmSchemaError: a schema was requested and the body did not match it.
        """
        ...


def structured_output(
    text: str, *, stop_reason: StopReason, model: str, usage: Usage
) -> Mapping[str, Any]:
    """Decode a schema-constrained body, or raise with the call's cost attached.

    Shared by both backends so the fake cannot be more forgiving than the real
    thing.

    A body cut off at ``max_tokens`` is not the model getting the schema wrong,
    and it gets its own error: the retry-once path feeds the validation failure
    back and asks again, which would truncate at exactly the same place. The
    caller has to raise the ceiling instead.
    """
    if stop_reason == "max_tokens":
        raise LlmResponseError(
            "the model hit max_tokens before finishing a schema-constrained response; "
            "retrying the same request truncates identically, so raise max_tokens instead",
            model=model,
            usage=usage,
        )
    return decode_json_output(text, model=model, usage=usage)


def decode_json_output(
    text: str, *, model: str | None = None, usage: Usage | None = None
) -> Mapping[str, Any]:
    """Decode a JSON response body.

    Raises rather than returning ``None``: a caller that asked for structured
    output and got prose has nothing to fall back on.
    """
    if not text.strip():
        raise LlmSchemaError(
            "model returned an empty body for a schema-constrained request",
            raw_output=text,
            model=model,
            usage=usage,
        )
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LlmSchemaError(
            f"model output was not valid JSON: {exc}", raw_output=text, model=model, usage=usage
        ) from exc
    if not isinstance(decoded, dict):
        raise LlmSchemaError(
            f"expected a JSON object, got {type(decoded).__name__}",
            raw_output=text,
            model=model,
            usage=usage,
        )
    return decoded
