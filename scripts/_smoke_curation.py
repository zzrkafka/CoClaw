"""Smoke: §3.1 two-axis criterion + §3.2 tenure + §3.3 prune hook. Pure logic, no LLM/LKH."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.curation import (INVESTIGATE, KEEP, MERGE, PRUNE,    # noqa: E402
                               apply_curation, curation_decision)
from skills.library import SkillLibrary                            # noqa: E402
from skills.schema import Contract, Skill                          # noqa: E402

CFG = {"curation": {"prune": True, "tenure_k": 3, "prune_mv_eps": 0.001, "prune_reuse_max": 0.34}}


def mk(sid, mv, reuse, created):
    s = Skill(id=sid, kind="order", version=1, name=sid, description_nl=sid, contract=Contract(),
              code="def f(struct, C, prims):\n    return []\n", parent_ids=[], created_round=created,
              status="active")
    s.reuse.marginal_value = mv
    s.reuse.reuse_rate = reuse
    s.reuse.created_round = created
    return s


R = 5  # current round; tenure_k=3 so created<=2 is tenure-satisfied, created>2 is young
cases = [
    # (mv,    reuse, created) -> expected verdict
    ((0.05,   1.0,   0), KEEP),          # valuable + reused
    ((0.0,    1.0,   0), MERGE),         # reused but no marginal -> redundant
    ((0.0,    0.0,   0), PRUNE),         # dead weight, tenure satisfied
    ((0.0,    0.0,   5), INVESTIGATE),   # dead-looking but YOUNG -> shielded (§3.2)
    ((0.05,   0.0,   0), INVESTIGATE),   # valuable but not reused -> wire it up, never prune
    ((-0.05,  1.0,   0), PRUNE),         # harmful -> prune even though reused
    ((-0.05,  0.0,   5), PRUNE),         # harmful -> prune even within tenure
]
for (mv, reuse, created), want in cases:
    got = curation_decision(mk("s", mv, reuse, created), R, CFG)
    assert got == want, f"mv={mv} reuse={reuse} created={created}: got {got}, want {want}"
print("OK §3.1/§3.2: 2-axis verdicts + tenure shield + harmful-immediate all correct.")

# §3.3 hook: prunes only PRUNE verdicts; off by default
lib = SkillLibrary()
for sid, (mv, reuse, created) in {
        "keep": (0.05, 1.0, 0), "dead": (0.0, 0.0, 0), "harm": (-0.05, 1.0, 0),
        "young": (0.0, 0.0, 5), "valuable_idle": (0.05, 0.0, 0)}.items():
    lib.add(mk(sid, mv, reuse, created))
res = apply_curation(lib, R, CFG)
active_ids = {s.id for s in lib.active()}
assert set(res["pruned"]) == {"dead", "harm"}, f"pruned wrong set: {res['pruned']}"
assert active_ids == {"keep", "young", "valuable_idle"}, f"survivors wrong: {active_ids}"
assert res["counts"][PRUNE] == 2 and res["counts"][KEEP] == 1

off = apply_curation(SkillLibrary(), R, {"curation": {"prune": False}})
assert off == {"enabled": False, "decisions": {}, "pruned": []}, "must be a no-op when disabled"
print("OK §3.3: prune hook removes dead/harmful, keeps the rest, no-op when disabled.")

# §3.4: a dead-LOOKING (mv~0), un-reused, tenure-SATISFIED skill that has positive leave-one-in
# value must be SPARED when leave_one_in is on, but pruned when it is off.
combo = mk("combo", mv=0.0, reuse=0.0, created=0)       # would be PRUNE on the 2 axes alone
combo.reuse.marginal_add = 0.05
assert curation_decision(combo, R, {"curation": {"prune": True, "tenure_k": 3}}) == PRUNE, \
    "without leave_one_in, a dead-looking tenure-satisfied skill is pruned"
assert curation_decision(combo, R, {"curation": {"prune": True, "tenure_k": 3,
                                                  "leave_one_in": True}}) == INVESTIGATE, \
    "with leave_one_in, positive marginal_add must spare it (weak-alone/strong-combo)"
print("OK §3.4: leave-one-in combination value protects weak-alone/strong-combo from pruning.")
