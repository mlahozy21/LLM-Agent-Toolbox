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
