"""A minimal blackboard: a shared workspace where heterogeneous agents cooperate.

Agents read and write structured entries to a common space rather than messaging
each other directly. A simple controller repeatedly asks each agent whether it
can contribute given the current state, until no agent can add anything new.
This mirrors the classical blackboard architecture, adapted to LLM-style agents.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Entry:
    key: str
    value: Any
    author: str


class Blackboard:
    """Thread-safe shared key/value workspace with a change log.

    All reads and writes are serialised by a single lock, so concurrent agents
    can safely write and read (including the change `log`) without data races.
    """

    def __init__(self):
        self._data: dict[str, Entry] = {}
        self._log: list[Entry] = []
        self._lock = threading.Lock()

    def write(self, key: str, value: Any, author: str) -> None:
        with self._lock:
            entry = Entry(key, value, author)
            self._data[key] = entry
            self._log.append(entry)

    def read(self, key: str, default=None):
        with self._lock:
            e = self._data.get(key)
            return e.value if e else default

    def keys(self) -> set[str]:
        with self._lock:
            return set(self._data)

    @property
    def log(self) -> list[Entry]:
        # Copy under the lock: `write` mutates `self._log` while holding the
        # lock, so reading/copying it without the lock would be a data race
        # (and could observe a list being mutated mid-append).
        with self._lock:
            return list(self._log)


@dataclass
class Agent:
    """An agent that can act on the blackboard when its precondition holds.

    `can_act(bb)` -> bool ; `act(bb)` performs one contribution (writes to bb).
    """
    name: str
    can_act: Callable[[Blackboard], bool]
    act: Callable[[Blackboard], None]


class Controller:
    """Runs agents over a blackboard until quiescence (no agent can act)."""

    def __init__(self, blackboard: Blackboard, agents: list[Agent], max_rounds: int = 50):
        self.bb = blackboard
        self.agents = agents
        self.max_rounds = max_rounds

    def run(self, raise_on_max_rounds: bool = False) -> list[str]:
        """Execute until no agent can contribute. Returns the activation trace.

        Each round, every agent whose ``can_act`` precondition holds is fired
        once (in registration order). The loop stops as soon as a full round
        passes with no agent acting (quiescence).

        **Idempotency / termination contract.** For ``run`` to terminate, agents
        must eventually stop acting: once an agent has made its contribution, its
        ``can_act`` should return ``False`` on the resulting state (i.e. acting is
        effectively idempotent — re-running on an unchanged state must not keep
        producing new work). An agent whose ``can_act`` stays ``True`` forever
        will keep firing every round and never reach quiescence.

        If ``max_rounds`` is reached without quiescence this is treated as a
        likely non-terminating configuration: a :class:`RuntimeWarning` is always
        emitted, and ``RuntimeError`` is raised when ``raise_on_max_rounds`` is
        set.
        """
        trace = []
        reached_quiescence = False
        for _ in range(self.max_rounds):
            progressed = False
            for agent in self.agents:
                if agent.can_act(self.bb):
                    agent.act(self.bb)
                    trace.append(agent.name)
                    progressed = True
            if not progressed:
                reached_quiescence = True
                break
        if not reached_quiescence:
            msg = (
                f"Controller.run hit max_rounds={self.max_rounds} without reaching "
                "quiescence; an agent's can_act may never become False (check the "
                "idempotency/termination contract)."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            if raise_on_max_rounds:
                raise RuntimeError(msg)
        return trace
