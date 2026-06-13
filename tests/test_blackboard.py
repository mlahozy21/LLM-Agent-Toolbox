import sys; sys.path.insert(0, "scripts")
import threading
import warnings

import pytest

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


def test_concurrent_writes_and_log_reads_are_safe():
    """Many threads writing while others read the log: backs the thread-safe
    claim. Asserts no exception is raised and the final log length is exact."""
    bb = Blackboard()
    n_writers, per_writer = 8, 200
    errors = []

    def writer(wid):
        try:
            for i in range(per_writer):
                bb.write(f"k{wid}-{i}", i, author=f"w{wid}")
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    def reader():
        try:
            for _ in range(per_writer):
                # copying the log concurrently must never raise / observe a torn list
                _ = len(bb.log)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread-safety violated: {errors[:3]}"
    assert len(bb.log) == n_writers * per_writer


def test_controller_terminates_on_quiescence():
    """A well-behaved (idempotent) agent set reaches quiescence with no warning."""
    bb = Blackboard()
    once = Agent(
        name="once",
        can_act=lambda b: b.read("done") is None,
        act=lambda b: b.write("done", True, "once"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        trace = Controller(bb, [once], max_rounds=5).run()
    assert trace == ["once"]


def test_controller_warns_on_max_rounds():
    """An agent whose can_act never becomes False must trip the max_rounds guard."""
    bb = Blackboard()
    forever = Agent(
        name="forever",
        can_act=lambda b: True,                      # never self-disables
        act=lambda b: b.write("x", 1, "forever"),
    )
    ctrl = Controller(bb, [forever], max_rounds=3)
    with pytest.warns(RuntimeWarning):
        trace = ctrl.run()
    assert len(trace) == 3                            # one activation per round

    # and it can be made strict (it also warns before raising; suppress that
    # expected warning so it does not show up in the suite summary)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(RuntimeError):
            Controller(bb, [forever], max_rounds=3).run(raise_on_max_rounds=True)
