"""Smoke: validate analysis.lesion offline (no LLM).

A correct cluster-order construct must score a LARGE positive marginal value (removing it drops
the library back to cost-NN ~28-31%); a no-op construct must score ~0 (the pipeline's built-in
NN already dominates it). This checks the leave-one-out credit logic end to end on F_hard."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.lesion import build_holdout, lesion_marginal_values   # noqa: E402
from skills.library import SkillLibrary                             # noqa: E402
from skills.schema import Contract, Skill                           # noqa: E402

# Validated near-optimal cluster-order construct (cluster -> brute-force directed cluster TSP ->
# within-cluster NN). Inlined here so this smoke test is self-contained (no scripts/ dependency).
GOOD = r'''
def f(n, x, C, prims):
    import numpy as np
    import itertools
    labels = prims.cluster_by_cost(C)
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)
    clust_list = list(clusters.keys())
    K = len(clust_list)
    IC = np.full((K, K), np.inf)
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            IC[a, b] = min(C[i, j] for i in clusters[clust_list[a]] for j in clusters[clust_list[b]])
    best_order, best_cost = None, np.inf
    for perm in itertools.permutations(range(K)):
        cost = sum(IC[perm[t], perm[t + 1]] for t in range(K - 1))
        if cost < best_cost:
            best_cost, best_order = cost, perm
    tour = []
    for idx in best_order:
        nodes = clusters[clust_list[idx]]
        if not nodes:
            continue
        start = min(nodes, key=lambda nd: C[tour[-1], nd]) if tour else nodes[0]
        unvisited = set(nodes); unvisited.discard(start)
        current, cluster_tour = start, [start]
        while unvisited:
            nxt = min(unvisited, key=lambda nd: C[current, nd])
            cluster_tour.append(nxt); unvisited.remove(nxt); current = nxt
        tour.extend(cluster_tour)
    if not prims.is_valid_tour(tour, n):
        tour.extend([i for i in range(n) if i not in tour])
    return tour
'''
NOOP = "def f(n, x, C, prims):\n    return list(range(n))\n"   # NN already beats identity


def mk(name, code):
    return Skill(id=name, kind="construct", version=1, name=name, description_nl=name,
                 contract=Contract(), code=code, parent_ids=[], created_round=0, status="active")


cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
lib = SkillLibrary()
lib.add(mk("good_cluster_order", GOOD))
lib.add(mk("noop_identity", NOOP))

holdout = build_holdout(cfg, k=4)
print(f"[smoke] holdout = {len(holdout)} F_hard instances (seed band 7000)")
marginal = lesion_marginal_values(lib, holdout, cfg, mode="pipeline")
print("marginal values:", {k: f"{v:+.1%}" for k, v in marginal.items()})

good = next(v for k, v in marginal.items() if "good" in k)
noop = next(v for k, v in marginal.items() if "noop" in k)
print(f"good construct marginal = {good:+.1%}   (expect large +, ~25%+)")
print(f"noop construct marginal = {noop:+.1%}   (expect ~0)")

# also confirm write-back into the library object
back = {s.id: s.reuse.marginal_value for s in lib.active()}
assert back == marginal, f"write-back mismatch: {back} vs {marginal}"

assert good > 0.10, f"good construct should have large positive marginal, got {good:+.1%}"
assert abs(noop) < 0.03, f"noop should be ~0 marginal, got {noop:+.1%}"
print("OK: lesion distinguishes a valuable skill from dead weight, and writes marginal_value back.")
