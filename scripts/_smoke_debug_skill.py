"""Smoke: §1.4 debug skill -- kind registered + grounded by repair-delta from a STUCK base.
A real recovery (re-cluster -> solve cluster order -> rebuild) must score a LARGE improvement
(NN base ~28-31% -> ~2%); a no-op debug (`return tour`) must score ~0. Needs LKH (gap)."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evolve import eval_debug                       # noqa: E402
from analysis.lesion import build_holdout                 # noqa: E402
from skills.library import SkillLibrary                   # noqa: E402
from skills.schema import KIND, SIGNATURES, Contract, Skill   # noqa: E402

assert "debug" in KIND and SIGNATURES["debug"] == "def f(tour, C, prims) -> list[int]"

GOOD = r'''
def f(tour, C, prims):
    import numpy as np, itertools
    n = len(tour)
    labels = prims.cluster_by_cost(C)
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)
    cl = list(clusters.keys()); K = len(cl)
    IC = np.full((K, K), np.inf)
    for a in range(K):
        for b in range(K):
            if a != b:
                IC[a, b] = min(C[i, j] for i in clusters[cl[a]] for j in clusters[cl[b]])
    best, bc = None, np.inf
    for perm in itertools.permutations(range(K)):
        c = sum(IC[perm[t], perm[t + 1]] for t in range(K - 1))
        if c < bc:
            bc, best = c, perm
    out = []
    for idx in best:
        nodes = clusters[cl[idx]]
        start = min(nodes, key=lambda nd: C[out[-1], nd]) if out else nodes[0]
        un = set(nodes); un.discard(start); cur = start; ct = [start]
        while un:
            nx = min(un, key=lambda nd: C[cur, nd]); ct.append(nx); un.remove(nx); cur = nx
        out.extend(ct)
    if not prims.is_valid_tour(out, n):
        out.extend([i for i in range(n) if i not in out])
    return out
'''
NOOP = "def f(tour, C, prims):\n    return list(tour)\n"


def mk(name, code):
    return Skill(id=name, kind="debug", version=1, name=name, description_nl=name,
                 contract=Contract(), code=code, parent_ids=[], created_round=0, status="active")


cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
dev = build_holdout(cfg, k=cfg["budgets"]["dev_eval_k"])
lib = SkillLibrary()

good = eval_debug(mk("good_recover", GOOD), dev, lib, cfg)
noop = eval_debug(mk("noop_recover", NOOP), dev, lib, cfg)
print(f"good debug: delta={good['delta']:+.1%} base={good['base_gap']:.1%} -> cand={good['cand_gap']:.1%} "
      f"regressed={good['regressed']}")
print(f"noop debug: delta={noop['delta']:+.1%} base={noop['base_gap']:.1%} -> cand={noop['cand_gap']:.1%} "
      f"regressed={noop['regressed']}")

assert good["feasible"] and good["delta"] < -0.10, f"good debug should recover a lot, got {good['delta']:+.1%}"
assert not good["regressed"], "good debug must not regress"
assert abs(noop["delta"]) < 0.01, f"noop debug should be ~0, got {noop['delta']:+.1%}"
print("OK §1.4: debug kind registered; repair-delta credits real recovery, zeroes a no-op.")
