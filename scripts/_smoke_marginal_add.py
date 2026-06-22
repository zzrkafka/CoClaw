"""Smoke: §3.4 leave-one-in marginal_add credits an operator in the strategy context (LKH).
With detect+order+build present, the `order` skill -- which siblings/leave-one-out can mask -- must
show a LARGE positive leave-one-in value (base detect+build uses order=None -> bad; adding order ->
near-optimal). detect ~0 (the default struct already covers it). Mirrors the hard_split finding."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.lesion import build_holdout, lesion_marginal_add   # noqa: E402
from skills.library import SkillLibrary                          # noqa: E402
from skills.schema import Contract, Skill                        # noqa: E402

DETECT = '''
def f(tour, x, C, prims):
    lab = prims.cluster_by_cost(C)
    return {"labels": [int(v) for v in lab], "k": len(set(int(v) for v in lab))}
'''
ORDER = '''
def f(struct, C, prims):
    import numpy as np, itertools
    labels = struct["labels"]; cl = sorted(set(labels)); K = len(cl)
    clusters = {c: [i for i, l in enumerate(labels) if l == c] for c in cl}
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
    return [cl[i] for i in best]
'''
BUILD = '''
def f(n, x, C, prims, struct=None, order=None):
    labels = struct["labels"] if struct else [0] * n
    if order is None:
        order = sorted(set(labels))
    clusters = {}
    for i, l in enumerate(labels):
        clusters.setdefault(l, []).append(i)
    tour = []
    for c in order:
        nodes = clusters.get(c, [])
        if not nodes:
            continue
        start = min(nodes, key=lambda nd: C[tour[-1], nd]) if tour else nodes[0]
        un = set(nodes); un.discard(start); cur = start; ct = [start]
        while un:
            nx = min(un, key=lambda nd: C[cur, nd]); ct.append(nx); un.remove(nx); cur = nx
        tour.extend(ct)
    if not prims.is_valid_tour(tour, n):
        tour.extend([i for i in range(n) if i not in tour])
    return tour
'''


def mk(sid, kind, code):
    return Skill(id=sid, kind=kind, version=1, name=sid, description_nl=sid, contract=Contract(),
                 code=code, parent_ids=[], created_round=0, status="active")


cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
lib = SkillLibrary()
lib.add(mk("detect", "diagnose", DETECT))
lib.add(mk("order", "order", ORDER))
lib.add(mk("build", "construct", BUILD))

holdout = build_holdout(cfg, k=cfg["budgets"]["dev_eval_k"])
add = lesion_marginal_add(lib, holdout, cfg)
print("marginal_add:", {k: f"{v:+.1%}" for k, v in add.items()})

# write-back onto the skills
back = {s.id: s.reuse.marginal_add for s in lib.active()}
assert back == add, "marginal_add not written back to the library"
assert add["order"] > 0.05, f"order should carry large in-context value, got {add['order']:+.1%}"
assert add["build"] > 0.05, f"build should carry large in-context value, got {add['build']:+.1%}"
print("OK §3.4: leave-one-in credits order/build in context; written back to the library.")
