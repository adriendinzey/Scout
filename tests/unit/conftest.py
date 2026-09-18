"""No unit test reaches the network.

Hard constraint 3 says no test calls the real Anthropic API — not in CI, not by
default locally. Observing that it currently holds is not the same as enforcing
it: the boundary's tests construct real SDK objects, and one forgotten stub would
turn a unit test into a billed API call that still passes. The same reasoning the
project applies to the agent's budgets applies here — a rule that lives only in a
convention is a suggestion.

So the socket is taken away for the whole of `tests/unit`. Integration tests are
unaffected: they live in `tests/integration` and legitimately talk to Postgres.
"""

from __future__ import annotations

import socket
from typing import Any, NoReturn

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound connection from a unit test fail loudly."""

    def refuse(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError(
            "a unit test tried to open a network connection. Unit tests never reach the "
            "network, and no test may call the real Anthropic API — stub the client instead."
        )

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
