import sys; sys.path.insert(0, "scripts")
from agent_toolbox.blackboard import Blackboard, Agent, Controller

def test_blackboard_rw_and_log():
    bb = Blackboard()
    bb.write("a", 1, "u")
    assert bb.read("a") == 1
    assert bb.read("missing", default=0) == 0
    assert bb.log[-1].author == "u"

def test_controller_runs_dependencies_in_order():
    from blackboard_demo import build
    bb, agents = build()
    trace = Controller(bb, agents).run()
    # weather must precede packing which must precede summary
    assert trace.index("weather_agent") < trace.index("packing_agent") < trace.index("summary_agent")
    assert "Paris" in bb.read("summary")
