"""Smoke: §2.2 typed-hole authoring + §2.4 two-level insufficiency / red line. No LLM / no LKH
(fill_holes uses an injected mock induction fn; the rest is pure logic)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.growth import (STRATEGY_HOLES, fill_holes, hole_intent,    # noqa: E402
                          slot_fill_ratio, strategy_redline, unfilled_holes, weak_holes)
from agent.lessons import lint_skill                                  # noqa: E402
from agent.seeds import SEED_STRATEGY_CODE                            # noqa: E402
from skills.library import SkillLibrary                               # noqa: E402
from skills.schema import Contract, Skill                             # noqa: E402


def op(sid, kind, mv=0.0):
    s = Skill(id=sid, kind=kind, version=1, name=sid, description_nl=sid, contract=Contract(),
              code="def f(*a, **k):\n    return []\n", parent_ids=[], created_round=0, status="active")
    s.reuse.marginal_value = mv
    return s


# §2.2 unfilled / weak holes
lib = SkillLibrary()
lib.add(op("d", "diagnose"))
assert unfilled_holes(lib) == ["order", "construct", "local_search"], unfilled_holes(lib)
lib.add(op("o", "order", mv=0.0))            # an order operator exists but carries no value
assert "order" in weak_holes(lib, {"curation": {"prune_mv_eps": 0.001}}), "0-marginal order is weak"
assert unfilled_holes(lib) == ["construct", "local_search"]
assert all(hole_intent(k) for k in STRATEGY_HOLES)
print("OK §2.2: unfilled/weak holes identify exactly what to author.")

# §2.4 L2 red line
ok, _ = strategy_redline(SEED_STRATEGY_CODE)
assert ok, "the seed strategy is a thin dispatcher -> allowed"
bad_noskills = "def f(n, x, C, skills, prims):\n    return list(range(n))\n"   # inlines, ignores skills
ok2, why2 = strategy_redline(bad_noskills)
assert not ok2 and "DISPATCH" in why2, "a non-dispatching strategy must be rejected"
fat = "def f(n, x, C, skills, prims):\n    skills\n" + "\n".join(f"    a{i} = {i}" for i in range(90))
ok3, _ = strategy_redline(fat)
assert not ok3, "an over-long strategy must be rejected (no inline monolith)"
hard, _ = lint_skill("strategy", bad_noskills)
assert any("DISPATCH" in h for h in hard), "lint_skill must hard-block a non-dispatching strategy"
assert not lint_skill("strategy", SEED_STRATEGY_CODE)[0], "the seed strategy must pass lint"
print("OK §2.4: red line blocks non-dispatch / inline-monolith strategies; thin dispatcher passes.")

# §2.2 fill_holes with a mock induction fn (authors exactly the unfilled holes)
lib2 = SkillLibrary()
lib2.add(op("d", "diagnose"))


def mock_induce(o, lib, dev, cfg, llm, lblock, rnd, lpath):
    return (f"def f(*a, **k):\n    return []  # {o['kind']}\n", {"feasible": True}, True)


off = fill_holes(lib2, [], {"library": {"fill_holes": False}}, None, None, 0, induce=mock_induce)
assert off == [], "fill_holes must be a no-op when disabled"
recs = fill_holes(lib2, [], {"library": {"fill_holes": True}}, None, None, 0, induce=mock_induce)
filled = {r["hole"] for r in recs if r["accepted"]}
assert filled == {"order", "construct", "local_search"}, f"should author the 3 unfilled holes, got {filled}"
kinds = {s.kind for s in lib2.active()}
assert {"diagnose", "order", "construct", "local_search"} <= kinds, kinds
print("OK §2.2: fill_holes authors exactly the missing operators; no-op when disabled.")

# §2.5 self-bootstrap signal: fraction of slots filled by a VALUABLE operator
CF = {"curation": {"prune_mv_eps": 0.001}}
assert slot_fill_ratio(SkillLibrary(), CF) == 0.0
lib3 = SkillLibrary()
lib3.add(op("o", "order", mv=0.08)); lib3.add(op("c", "construct", mv=0.07))
lib3.add(op("d2", "diagnose", mv=0.0)); lib3.add(op("l", "local_search", mv=0.0))
assert slot_fill_ratio(lib3, CF) == 0.5, slot_fill_ratio(lib3, CF)   # 2 of 4 slots valuable
# leave-one-in value also counts a slot as filled
lib3.active()[2].reuse.marginal_add = 0.05    # the diagnose now carries in-context value
assert slot_fill_ratio(lib3, CF) == 0.75, slot_fill_ratio(lib3, CF)
print("OK §2.5: slot-fill ratio tracks slots filled by valuable operators (rising = converging).")
