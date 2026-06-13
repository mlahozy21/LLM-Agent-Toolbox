import numpy as np

from agent_toolbox.demo_data import TOOLS, LABELLED
from agent_toolbox.tool_selection import Tool, ToolSelector


def test_selector_basic_and_topk():
    sel = ToolSelector(TOOLS)
    res = sel.select("weather in Madrid", k=3)
    assert len(res) == 3
    names = [t.name for t, _ in res]
    assert "weather" in names              # obvious tool retrieved
    # similarities are sorted descending
    sims = [s for _, s in res]
    assert sims == sorted(sims, reverse=True)


def test_recall_at_3_reasonable():
    sel = ToolSelector(TOOLS)
    hits = sum(len(set(sel.select_names(q, 3)) & gold) > 0 for q, gold in LABELLED)
    assert hits / len(LABELLED) >= 0.6      # even the offline fallback clears this


class _ConstEmbedder:
    """Embedder that returns identical unit vectors, so every tool ties on score.

    Used to verify deterministic, registration-order tie-breaking."""

    def encode(self, texts):
        v = np.ones((len(texts), 4), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v


def test_tie_break_is_stable_registration_order():
    tools = [Tool(name=f"t{i}", description="same") for i in range(5)]
    sel = ToolSelector(tools, embedder=_ConstEmbedder())
    # all sims equal -> stable argsort must return the original order
    names = sel.select_names("anything", k=3)
    assert names == ["t0", "t1", "t2"]
    # deterministic across repeated calls
    assert sel.select_names("anything", k=3) == names
