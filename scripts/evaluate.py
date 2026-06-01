"""Evaluate semantic tool selection: recall@k on the labelled query set."""
import sys
sys.path.insert(0, "src")
from agent_toolbox.demo_data import TOOLS, LABELLED
from agent_toolbox.tool_selection import ToolSelector


def recall_at_k(selector, labelled, k):
    hits = 0
    for q, gold in labelled:
        got = set(selector.select_names(q, k=k))
        hits += len(got & gold) > 0
    return hits / len(labelled)


if __name__ == "__main__":
    sel = ToolSelector(TOOLS)
    for k in (1, 2, 3):
        print(f"recall@{k}: {recall_at_k(sel, LABELLED, k):.2f}")
    print("\nExample — query: 'What's the weather in Paris tomorrow?'")
    for t, s in sel.select("What's the weather in Paris tomorrow?", k=3):
        print(f"  {t.name:12s} sim={s:.3f}")
