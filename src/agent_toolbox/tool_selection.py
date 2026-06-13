"""Semantic tool selection for LLM agents.

When an agent has access to many tools, their descriptions do not all fit in the
context window. This module embeds each tool's description once, embeds the
incoming request at query time, and returns the top-k tools by cosine
similarity — so the LLM is only shown the tools it is likely to need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import get_embedder


@dataclass
class Tool:
    name: str
    description: str


class ToolSelector:
    """Embed a tool registry once, then retrieve the most relevant tools per query."""

    def __init__(self, tools: list[Tool], embedder=None):
        if not tools:
            raise ValueError("tools must be a non-empty list")
        self.tools = tools
        self.embedder = embedder or get_embedder()
        corpus = [f"{t.name}. {t.description}" for t in tools]
        self._matrix = self.embedder.encode(corpus)  # (n_tools, dim), L2-normalised

    def select(self, query: str, k: int = 3) -> list[tuple[Tool, float]]:
        """Return the top-k (tool, similarity) for a query, highest first."""
        q = self.embedder.encode([query])[0]
        sims = self._matrix @ q                       # cosine (rows are normalised)
        # Stable, deterministic ordering: argsort on -sims with a stable kind so
        # ties are broken by original tool order (registration order) rather than
        # the arbitrary order that reversing an ascending argsort produces.
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self.tools[i], float(sims[i])) for i in order]

    def select_names(self, query: str, k: int = 3) -> list[str]:
        return [t.name for t, _ in self.select(query, k)]
