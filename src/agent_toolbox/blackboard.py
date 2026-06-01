"""A minimal blackboard: a shared workspace where heterogeneous agents cooperate.

Agents read and write structured entries to a common space rather than messaging
each other directly. A simple controller repeatedly asks each agent whether it
can contribute given the current state, until no agent can add anything new.
This mirrors the classical blackboard architecture, adapted to LLM-style agents.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Entry:
    key: str
    value: Any
    author: str


class Blackboard:
    """Thread-safe shared key/value workspace with a change log."""

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

    def run(self) -> list[str]:
        """Execute until no agent can contribute. Returns the activation trace."""
        trace = []
        for _ in range(self.max_rounds):
            progressed = False
            for agent in self.agents:
                if agent.can_act(self.bb):
                    agent.act(self.bb)
                    trace.append(agent.name)
                    progressed = True
            if not progressed:
                break
        return trace
