"""agent_toolbox: building blocks for LLM multi-agent systems.

Two components:
  - tool_selection: pick the relevant tools for a query by semantic similarity
    (so an LLM only sees the tools it needs, under a limited context budget).
  - blackboard: a shared-workspace coordination primitive for heterogeneous agents.
"""
__version__ = "0.1.0"
