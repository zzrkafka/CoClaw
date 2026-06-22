"""Smoke: §1.3 plan-as-first-class-field, end to end, NO LLM / NO LKH (pure plumbing).

Checks: the field exists + defaults; materialize_skill threads `plan`; the library snapshots and
RELOADS it; the solver's plans-block filters to plan-bearing skills; and the CodeAct prompt shows
the STRATEGY PLANS section ONLY when a non-empty plans_block is injected (so the baseline solve,
which passes ""  unless harness.inject_plans, is byte-for-byte unchanged)."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evolve import materialize_skill                       # noqa: E402
from agent.solver import _plans_block                            # noqa: E402
from analysis.lesion import load_library_from_run                # noqa: E402
from llm.prompt_loader import render_messages                    # noqa: E402
from skills.library import SkillLibrary                          # noqa: E402
from skills.schema import Contract, Skill                        # noqa: E402

PLAN = "Cluster by cost; solve the small cluster TSP; build in that order; ONE bounded 2-opt."


def mk(sid, kind, plan=""):
    return Skill(id=sid, kind=kind, version=1, name=sid, description_nl=sid, contract=Contract(),
                 code="def f(n, x, C, skills, prims):\n    return list(range(n))\n",
                 parent_ids=[], created_round=0, status="active", plan=plan)


# 1) field exists + defaults blank
assert mk("op1", "order").plan == "", "plan must default to blank"
strat = mk("strat1", "strategy", PLAN)
assert strat.plan == PLAN

# 2) materialize_skill threads plan from the planner op
m = materialize_skill({"kind": "strategy", "plan": PLAN, "code": "def f():\n    pass\n"}, round_idx=0)
assert m.plan == PLAN, "materialize_skill dropped plan"
m0 = materialize_skill({"kind": "order", "code": "def f():\n    pass\n"}, round_idx=0)
assert m0.plan == "", "operator op should carry no plan"

# 3) snapshot + reload round-trip (skill_states -> skill_snapshot.jsonl -> load_library_from_run)
with tempfile.TemporaryDirectory() as d:
    run = Path(d) / "r"
    (run / "skills").mkdir(parents=True)
    lib = SkillLibrary(store_dir=str(run / "skills"))
    lib.add(strat)
    states = lib.skill_states()
    assert states[0]["plan"] == PLAN, "skill_states omitted plan"
    # emulate RunRecorder.skill_snapshot
    import json
    with (run / "skill_snapshot.jsonl").open("w") as f:
        for st in states:
            f.write(json.dumps({"library_version": lib.version, **st}) + "\n")
    reloaded = load_library_from_run(run)
    assert reloaded.active()[0].plan == PLAN, "plan lost on reload"

# 4) _plans_block filters to plan-bearing skills only
block = _plans_block([strat, mk("op2", "order")])
assert "strat1" in block and "op2" not in block, f"plans_block wrong: {block!r}"
assert _plans_block([mk("op2", "order")]) == "", "no plans -> empty block"

# 5) prompt shows STRATEGY PLANS only when plans_block is non-empty
def render(pb):
    return "\n".join(c for _, c in render_messages(
        "codeact_system", n=60, c_stats="x", x_preview="[]", skills_block="(none)",
        plans_block=pb, max_steps=8, eval_budget=None))

assert "STRATEGY PLANS" not in render(""), "empty plans_block must NOT add the section (baseline)"
on = render("- [strat1 (strategy)] " + PLAN)
assert "STRATEGY PLANS" in on and PLAN in on, "non-empty plans_block must show the section"

print("OK §1.3: plan field defaults/threads/snapshots/reloads; plans_block filters; prompt gated.")
