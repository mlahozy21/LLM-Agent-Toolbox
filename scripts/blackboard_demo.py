"""Demo: heterogeneous agents cooperating through a blackboard.

A tiny 'trip planning' workflow: a weather agent, a packing agent (depends on
weather) and a summary agent (depends on both) contribute to a shared workspace
without messaging each other directly. The controller runs them to quiescence.
"""
import sys
sys.path.insert(0, "src")
from agent_toolbox.blackboard import Agent, Blackboard, Controller

def build():
    bb = Blackboard()
    bb.write("destination", "Paris", "user")

    def weather_act(bb): bb.write("weather", "rainy, 12C", "weather_agent")
    weather = Agent("weather_agent",
                    can_act=lambda bb: "destination" in bb.keys() and "weather" not in bb.keys(),
                    act=weather_act)

    def packing_act(bb):
        w = bb.read("weather")
        items = ["umbrella", "jacket"] if "rain" in w else ["sunglasses"]
        bb.write("packing", items, "packing_agent")
    packing = Agent("packing_agent",
                    can_act=lambda bb: "weather" in bb.keys() and "packing" not in bb.keys(),
                    act=packing_act)

    def summary_act(bb):
        bb.write("summary",
                 f"Trip to {bb.read('destination')}: weather {bb.read('weather')}, "
                 f"pack {bb.read('packing')}.", "summary_agent")
    summary = Agent("summary_agent",
                    can_act=lambda bb: {"weather", "packing"} <= bb.keys() and "summary" not in bb.keys(),
                    act=summary_act)
    return bb, [summary, packing, weather]  # deliberately unordered

if __name__ == "__main__":
    bb, agents = build()
    trace = Controller(bb, agents).run()
    print("activation order:", " -> ".join(trace))
    print("final summary:", bb.read("summary"))
