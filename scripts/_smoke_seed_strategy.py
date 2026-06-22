"""Smoke: §2.1 seed strategy + §2.3 graceful degradation.
- gating (pure logic): seed_library is a no-op off / when a strategy already exists; seeds otherwise.
- degradation (LKH): the seed strategy with NO operators yields a LEGAL tour (NN fallback).
- composition (LKH): with detect+order+build operators it yields a NEAR-OPTIMAL tour (holes filled)."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.seeds import seed_library, seed_strategy_skill          # noqa: E402
from analysis.lesion import build_holdout                          # noqa: E402
from sandbox.executor import Limits, run_in_sandbox                # noqa: E402
from sandbox.primitives import is_valid_tour                       # noqa: E402
from skills.library import SkillLibrary                            # noqa: E402
from skills.schema import Contract, Skill                          # noqa: E402
from solvers.reference import gap                                  # noqa: E402
import numpy as np                                                 # noqa: E402

# --- gating (no LKH) ---
lib = SkillLibrary()
assert seed_library(lib, {"library": {"seed_strategy": False}}) is None and not lib.active()
sid = seed_library(lib, {"library": {"seed_strategy": True}})
assert sid and len([s for s in lib.active() if s.kind == "strategy"]) == 1
assert seed_library(lib, {"library": {"seed_strategy": True}}) is None, "must not double-seed"
print("OK §2.1 gating: seeds once, no-op when off or already present.")

# --- operators to fill the holes ---
DETECT = ('def f(tour, x, C, prims):\n'
          '    lab = prims.cluster_by_cost(C)\n'
          '    return {"labels": [int(v) for v in lab], "k": len(set(int(v) for v in lab))}\n')
ORDER = ('def f(struct, C, prims):\n'
         '    import numpy as np, itertools\n'
         '    labels = struct["labels"]; cl = sorted(set(labels)); K = len(cl)\n'
         '    cm = {c: [i for i, l in enumerate(labels) if l == c] for c in cl}\n'
         '    IC = np.full((K, K), np.inf)\n'
         '    for a in range(K):\n'
         '        for b in range(K):\n'
         '            if a != b: IC[a, b] = min(C[i, j] for i in cm[cl[a]] for j in cm[cl[b]])\n'
         '    best, bc = None, np.inf\n'
         '    for p in itertools.permutations(range(K)):\n'
         '        c = sum(IC[p[t], p[t+1]] for t in range(K-1))\n'
         '        if c < bc: bc, best = c, p\n'
         '    return [cl[i] for i in best]\n')
BUILD = ('def f(n, x, C, prims, struct=None, order=None):\n'
         '    labels = struct["labels"] if struct else [0]*n\n'
         '    if order is None: order = sorted(set(labels))\n'
         '    cm = {}\n'
         '    for i, l in enumerate(labels): cm.setdefault(l, []).append(i)\n'
         '    tour = []\n'
         '    for c in order:\n'
         '        nodes = cm.get(c, [])\n'
         '        if not nodes: continue\n'
         '        start = min(nodes, key=lambda nd: C[tour[-1], nd]) if tour else nodes[0]\n'
         '        un = set(nodes); un.discard(start); cur = start; ct = [start]\n'
         '        while un:\n'
         '            nx = min(un, key=lambda nd: C[cur, nd]); ct.append(nx); un.remove(nx); cur = nx\n'
         '        tour.extend(ct)\n'
         '    if not prims.is_valid_tour(tour, n): tour.extend([i for i in range(n) if i not in tour])\n'
         '    return tour\n')


def d(sid, kind, code):
    return {"id": sid, "kind": kind, "code": code}


def run_seed(inst, op_dicts):
    skills = [d("seed_strategy", "strategy", seed_strategy_skill().code)] + op_dicts
    ctx = {"n": inst.n, "x": np.asarray(inst.x), "C": np.asarray(inst.C), "skills": skills}
    code = "FINAL_TOUR = skills['seed_strategy'](n, x, C, skills, prims)"
    return run_in_sandbox(code, ctx, Limits(cpu_seconds=20, wall_timeout=8.0))


cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
inst = build_holdout(cfg, k=1)[0]

bare = run_seed(inst, [])                                   # no operators -> degrade to NN
assert bare.feasible and is_valid_tour(bare.tour, inst.n), "degraded scaffold must be a legal tour"
g_bare = gap(bare.tour, inst)

full = run_seed(inst, [d("det", "diagnose", DETECT), d("ord", "order", ORDER),
                       d("bld", "construct", BUILD)])        # holes filled -> compose
assert full.feasible and is_valid_tour(full.tour, inst.n)
g_full = gap(full.tour, inst)

print(f"seed-strategy gap: bare(NN fallback)={g_bare:.1%}  filled(detect+order+build)={g_full:.1%}")
assert g_full < g_bare - 0.10, f"filling holes must help a lot ({g_bare:.1%} -> {g_full:.1%})"
print("OK §2.1/§2.3: scaffold degrades to a legal tour and composes operators when present.")
