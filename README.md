# LLM Agent Toolbox

Two reusable building blocks for **LLM multi-agent systems**, implemented from
scratch and fully tested — the core mechanisms behind how heterogeneous agents
decide *which tool to use* and *how to cooperate*.

## 1. Semantic tool selection

When an agent has access to many tools, their descriptions don't all fit in the
model's context window. `ToolSelector` embeds every tool description **once**,
embeds the incoming request at query time, and returns the **top-k tools by
cosine similarity** — so the LLM is only shown the tools it is likely to need.

```python
from agent_toolbox.tool_selection import Tool, ToolSelector

tools = [Tool("weather", "Get the weather for a city."),
         Tool("calculator", "Evaluate maths expressions."), ...]
sel = ToolSelector(tools)
sel.select_names("What's the weather in Paris?", k=3)   # -> ['weather', ...]
```

Embeddings come from **sentence-transformers** if installed (`pip install -e ".[neural]"`),
otherwise a deterministic, offline hashing fallback is used so everything runs
without downloads.

**Evaluation** (`scripts/evaluate.py`) reports recall@k on a labelled query set.
With the offline fallback embedder the toolbox already reaches **recall@1 = 0.70,
recall@3 = 0.80**; a neural encoder pushes this higher.

## 2. Blackboard coordination

`Blackboard` is a thread-safe shared workspace; agents read/write structured
entries instead of messaging each other directly. All operations — including
reading the change `log` — are serialised by a single lock, so concurrent agents
never race (covered by a multi-threaded test). A `Controller` runs the agents
to **quiescence** (until none can contribute), which naturally resolves
dependencies between them.

**Termination contract.** The controller fires every ready agent once per round
and stops when a full round produces no new work. For this to terminate, acting
must be effectively *idempotent*: once an agent has contributed, its `can_act`
should return `False` on the resulting state. An agent whose `can_act` never
becomes `False` would loop forever, so the controller caps the loop at
`max_rounds` (default 50) and emits a `RuntimeWarning` (or raises, with
`run(raise_on_max_rounds=True)`) if quiescence is not reached.

`scripts/blackboard_demo.py` shows a trip-planning workflow where a weather agent,
a packing agent (depends on weather) and a summary agent (depends on both)
cooperate purely through the shared state — the controller fires them in the
correct order even though they are registered unordered.

## Run

```bash
pip install -e ".[dev]"          # add ".[neural]" for real embeddings

python scripts/evaluate.py        # recall@k of semantic tool selection
python scripts/blackboard_demo.py # multi-agent blackboard demo
pytest                            # test suite
```

## Layout

```
.
├── src/agent_toolbox/
│   ├── embeddings.py       # sentence-transformers backend + offline fallback
│   ├── tool_selection.py   # ToolSelector: top-k tools by semantic similarity
│   ├── blackboard.py       # Blackboard + Agent + Controller (run to quiescence)
│   └── demo_data.py        # small tool registry + labelled queries
├── scripts/                # evaluate.py, blackboard_demo.py
└── tests/                  # tool-selection and blackboard tests
```

## Context

Built around the questions I study in my M2 internship on LLM multi-agent
systems: inter-agent communication (blackboard / shared workspaces) and tool
engagement/invocation under a limited context budget.

## License

Released under the MIT License — see `LICENSE`.
