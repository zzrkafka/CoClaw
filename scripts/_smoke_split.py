"""Smoke: validate the atomized construct (detect -> order -> build) + per-atom lesion credit.

(1) the deterministic pipeline must COMPOSE a diagnose(detect)+order+construct(build) trio into a
    near-oracle tour on F_hard;
(2) lesion must attribute the credit correctly: `order` HIGH (the must-induce cheap visit order),
    `construct`(build) HIGH (assembly), `diagnose`(detect) ~0 (a free prim wrapper -- removing it
    falls back to the default struct, no loss).
Offline (no LLM). Also doubles as an import/syntax check on the edited agent/evolve.py."""
import sys
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evolve import _run_pipeline, _skill_dict          # noqa: E402  (also checks the edit imports)
from analysis.lesion import build_holdout, lesion_marginal_values  # noqa: E402
from skills.library import SkillLibrary                       # noqa: E402
from skills.schema import Contract, Skill                     # noqa: E402
from solvers.reference import gap                             # noqa: E402

DETECT = '''def f(tour, x, C, prims):
    labels = prims.cluster_by_cost(C)
    return {"labels": [int(v) for v in labels], "k": len(set(int(v) for v in labels))}
'''

ORDER = '''def f(struct, C, prims):
    from itertools import permutations
    labels = list(struct["labels"]); n = len(labels)
    clusters = sorted(set(labels)); K = len(clusters)
    idx = {c: [i for i in range(n) if labels[i] == c] for c in clusters}
    IC = [[0.0] * K for _ in range(K)]
    for ai, a in enumerate(clusters):
        for bi, b in enumerate(clusters):
            if ai == bi:
                continue
            best = None
            for u in idx[a]:
                for v in idx[b]:
                    cc = C[u, v]
                    if best is None or cc < best:
                        best = cc
            IC[ai][bi] = best
    best_order, best_cost = list(range(K)), None
    for p in permutations(range(1, K)):
        order = [0] + list(p)
        cost = sum(IC[order[i]][order[(i + 1) % K]] for i in range(K))
        if best_cost is None or cost < best_cost:
            best_cost, best_order = cost, order
    return [clusters[i] for i in best_order]
'''

BUILD = '''def f(n, x, C, prims, struct=None, order=None):
    if struct is None:
        labels = [int(v) for v in prims.cluster_by_cost(C)]
    else:
        labels = list(struct["labels"])
    clusters = sorted(set(labels))
    idx = {c: [i for i in range(n) if labels[i] == c] for c in clusters}
    if order is None:
        order = clusters                      # arbitrary label order -> bad (why `order` matters)
    tour = []
    for c in order:
        nodes = idx.get(c, [])
        if not nodes:
            continue
        if tour:
            last = tour[-1]; start = min(nodes, key=lambda u: C[last, u])
        else:
            start = nodes[0]
        rem = set(nodes); rem.discard(start); sub = [start]; cur = start
        while rem:
            nxt = min(rem, key=lambda u: C[cur, u]); sub.append(nxt); rem.discard(nxt); cur = nxt
        tour += sub
    if not prims.is_valid_tour(tour, n):
        tour = tour + [i for i in range(n) if i not in set(tour)]
    return tour
'''


def mk(kind, name, code):
    return Skill(id=name, kind=kind, version=1, name=name, description_nl=name,
                 contract=Contract(), code=code, parent_ids=[], created_round=0, status="active")


cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
lib = SkillLibrary()
lib.add(mk("diagnose", "detect_clusters", DETECT))
lib.add(mk("order", "cheap_order", ORDER))
lib.add(mk("construct", "build_from_order", BUILD))

holdout = build_holdout(cfg, k=4)
print(f"[smoke] holdout = {len(holdout)} F_hard instances")

dicts = [_skill_dict(s) for s in lib.active()]
gaps = []
for inst in holdout:
    t = _run_pipeline(inst, dicts, None, apply_local=True)
    gaps.append(gap(t, inst) if t is not None else float("nan"))
trio = mean(g for g in gaps if g == g)
print(f"detect->order->build trio pipeline mean gap = {trio:.1%}   (expect near-oracle, < 10%)")

marginal = lesion_marginal_values(lib, holdout, cfg)
by_kind = {s.kind: marginal[s.id] for s in lib.active()}
print("lesion marginal by atom:", {k: f"{v:+.1%}" for k, v in by_kind.items()})

assert trio < 0.10, f"trio should compose to near-oracle, got {trio:.1%}"
assert by_kind["order"] > 0.10, f"order should be HIGH marginal, got {by_kind['order']:+.1%}"
assert by_kind["construct"] > 0.10, f"build should be HIGH marginal, got {by_kind['construct']:+.1%}"
assert by_kind["diagnose"] < 0.05, f"detect should be ~0 (prim wrapper), got {by_kind['diagnose']:+.1%}"
print("OK: split composes to near-oracle; lesion credits order & build, not the free detect wrapper.")
