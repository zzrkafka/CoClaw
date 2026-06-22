"""Offline smoke test (no LLM) for the new step-wise grounded judge + supervisor.

Checks: (1) lint flags the asymmetric-2-opt local_search, not the clean construct;
(2) eval_on_dev ACCEPTS the clean cluster-order construct (big negative delta);
(3) eval_local_search DETECTS the bad 2-opt as a regression (rejected)."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evolve import eval_local_search, eval_on_dev, materialize_skill
from agent.lessons import lint_skill
from lcar.generator import Family
from skills.library import SkillLibrary
from solvers.reference import fill_reference

GOOD_CONSTRUCT = r'''
def f(n, x, C, prims):
    from itertools import permutations
    labels = prims.cluster_by_cost(C)
    clusters = sorted(set(int(l) for l in labels))
    K = len(clusters)
    idx = {c: [i for i in range(n) if int(labels[i]) == c] for c in clusters}
    IC = [[0.0]*K for _ in range(K)]
    for ai,a in enumerate(clusters):
        for bi,b in enumerate(clusters):
            if ai==bi: continue
            best=None
            for u in idx[a]:
                for v in idx[b]:
                    cc=C[u,v]
                    if best is None or cc<best: best=cc
            IC[ai][bi]=best
    if K<=8:
        bo,bc=list(range(K)),None
        for p in permutations(range(1,K)):
            order=[0]+list(p)
            cost=sum(IC[order[i]][order[(i+1)%K]] for i in range(K))
            if bc is None or cost<bc: bc,bo=cost,order
    else: bo=list(range(K))
    co=[clusters[i] for i in bo]
    tour=[]
    for c in co:
        nodes=idx[c]
        if not nodes: continue
        sub,rem=[nodes[0]],set(nodes[1:])
        while rem:
            last=sub[-1]; nxt=min(rem,key=lambda u:C[last,u]); sub.append(nxt); rem.discard(nxt)
        tour+=sub
    if not prims.is_valid_tour(tour,n): tour=list(range(n))
    return tour
'''

BAD_2OPT = r'''
def f(tour, C, prims):
    tour=list(tour); m=len(tour)
    improved=True
    while improved:
        improved=False
        for i in range(1,m-1):
            for j in range(i+1,m):
                a,b=tour[i-1],tour[i]; c,d=tour[j],tour[(j+1)%m]
                if a==d: continue
                if C[a,c]+C[b,d] < C[a,b]+C[c,d]:
                    tour[i:j+1]=tour[i:j+1][::-1]; improved=True
    return tour
'''

cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
cfg["budgets"]["dev_eval_k"] = 3
fam = Family.from_config(cfg["lcar"], seed=1234, name="F_hard")
dev = [fam.sample_instance(n=60, seed=1234 + i) for i in range(3)]
for inst in dev:
    fill_reference(inst)

print("(1) LINT")
print("  good construct:", lint_skill("construct", GOOD_CONSTRUCT))
print("  bad 2-opt     :", lint_skill("local_search", BAD_2OPT))

print("\n(2) eval_on_dev on the clean construct (empty library) -- expect big negative delta -> accept")
lib = SkillLibrary()
cand = materialize_skill({"op": "add", "kind": "construct", "name": "clu", "code": GOOD_CONSTRUCT}, 0)
res = eval_on_dev(cand, dev, lib, cfg)
print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})

print("\n(3) eval_local_search on the BAD 2-opt (library = the good construct) -- expect regression -> reject")
lib2 = SkillLibrary()
lib2.add(materialize_skill({"op": "add", "kind": "construct", "name": "clu", "code": GOOD_CONSTRUCT}, 0), 0)
badls = materialize_skill({"op": "add", "kind": "local_search", "name": "bad2opt", "code": BAD_2OPT}, 0)
res2 = eval_local_search(badls, dev, lib2, cfg)
print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res2.items()})

thr = cfg["budgets"]["accept_threshold"]
print("\nVERDICT:")
print("  construct accepted? ", res["feasible"] and res["delta"] <= -thr)
print("  bad-2opt accepted?  ", res2["feasible"] and res2["delta"] <= -thr, "(should be False)")
print("  bad-2opt regressed? ", res2["regressed"], "(should be True)")
