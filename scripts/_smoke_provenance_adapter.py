"""Smoke: §6 provenance (origin_family threads + round-trips), §5.4a adapter faithfulness, §6.1 V(t).
Needs LKH for the adapter gap check + V(t)."""
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evolve import materialize_skill                         # noqa: E402
from analysis.lesion import build_holdout, library_value, load_library_from_run  # noqa: E402
from problems.adapter import ProblemAdapter, get_adapter, register_adapter        # noqa: E402
from sandbox.primitives import is_valid_tour, tour_length          # noqa: E402
from skills.library import SkillLibrary                            # noqa: E402
from skills.schema import KIND, Contract, Skill                    # noqa: E402
from solvers.reference import gap as direct_gap                    # noqa: E402

# --- §6 provenance: materialize threads origin_family; snapshot + reload preserve it ---
m = materialize_skill({"kind": "order", "origin_family": "F",
                       "code": "def f(struct, C, prims):\n    return []\n"}, 0)
assert m.origin_family == "F"
with tempfile.TemporaryDirectory() as d:
    run = Path(d) / "r"; (run / "skills").mkdir(parents=True)
    lib = SkillLibrary(store_dir=str(run / "skills")); lib.add(m)
    st = lib.skill_states()[0]
    assert st["origin_family"] == "F", "skill_states must carry origin_family"
    (run / "skill_snapshot.jsonl").write_text(json.dumps({"library_version": lib.version, **st}) + "\n")
    assert load_library_from_run(run).active()[0].origin_family == "F", "reload must preserve it"
print("OK §6 provenance: origin_family threads through materialize, snapshot, reload.")

# --- §5.4a adapter: the routing adapter wraps the existing fns FAITHFULLY (zero-change) ---
cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
inst = build_holdout(cfg, k=1)[0]
ad = get_adapter("routing")
assert tuple(sorted(ad.typed_vocab)) == tuple(sorted(KIND)), "typed_vocab must match the schema KINDs"
tour = list(range(inst.n))
assert ad.feasible(tour, inst) == is_valid_tour(tour, inst.n)
assert ad.objective(tour, inst) == tour_length(tour, inst.C)
assert ad.gap(tour, inst) == direct_gap(tour, inst), "adapter gap must equal the direct gap"
register_adapter("dummy", lambda: ProblemAdapter("dummy", ad.feasible, ad.objective, ad.gap,
                                                 ad.fill_reference))
assert get_adapter("dummy").name == "dummy", "registry must accept a second family"
print("OK §5.4a adapter: routing adapter faithful to direct fns; registry extensible.")

# --- §6.1 V(t): a good library beats the empty (NN-floor) library by a lot ---
GOOD = r'''
def f(n, x, C, prims, struct=None, order=None):
    import numpy as np, itertools
    labels = prims.cluster_by_cost(C)
    cm = {}
    for i, lab in enumerate(labels): cm.setdefault(lab, []).append(i)
    cl = list(cm); K = len(cl)
    IC = np.full((K, K), np.inf)
    for a in range(K):
        for b in range(K):
            if a != b: IC[a, b] = min(C[i, j] for i in cm[cl[a]] for j in cm[cl[b]])
    best, bc = None, np.inf
    for p in itertools.permutations(range(K)):
        c = sum(IC[p[t], p[t+1]] for t in range(K-1))
        if c < bc: bc, best = c, p
    t = []
    for idx in best:
        nd = cm[cl[idx]]
        s = min(nd, key=lambda z: C[t[-1], z]) if t else nd[0]
        un = set(nd); un.discard(s); cur = s; ct = [s]
        while un:
            nx = min(un, key=lambda z: C[cur, z]); ct.append(nx); un.remove(nx); cur = nx
        t.extend(ct)
    if not prims.is_valid_tour(t, n): t.extend([i for i in range(n) if i not in t])
    return t
'''
lib2 = SkillLibrary()
lib2.add(Skill(id="good", kind="construct", version=1, name="good", description_nl="good",
               contract=Contract(), code=GOOD, parent_ids=[], created_round=0, status="active"))
V = library_value(lib2, build_holdout(cfg, k=2), cfg)
assert V is not None and V > 0.10, f"V(t) should be large positive (library beats NN floor), got {V}"
print(f"OK §6.1 V(t): library_value = {V:.1%} (>0 = compounding over the NN floor).")
