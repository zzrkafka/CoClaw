"""Smoke: §4 discriminative eval set. Pure-logic selection (anchor guardrail + top-discrimination),
then the real gap-matrix + discrimination scores over a small library/pool (LKH)."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.discriminative import (discrimination_scores, gap_matrix,     # noqa: E402
                                     select_discriminative, discriminative_dev)
from analysis.lesion import build_holdout                                   # noqa: E402
from skills.library import SkillLibrary                                     # noqa: E402
from skills.schema import Contract, Skill                                   # noqa: E402

# --- pure logic: selection reserves an anchor + fills with the most discriminative ---
insts = list("ABCDE")
scores = [0.1, 0.9, 0.2, 0.8, 0.05]              # B and D are the most discriminative
sel = select_discriminative(insts, scores, k=3, anchor_frac=0.34)   # 1 anchor + 2 discriminative
assert len(sel) == 3 and "B" in sel and "D" in sel, sel
assert "A" in sel, "the lowest-index instance is the reserved anchor (anti-Goodhart)"
assert select_discriminative(insts, scores, k=9, anchor_frac=0.34) == insts, "k>=n returns all"
print("OK §4 selection: top-discrimination + reserved anchor; returns all when k>=n.")

# --- real gap matrix + discrimination (LKH) ---
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


def mk(sid, kind, code):
    return Skill(id=sid, kind=kind, version=1, name=sid, description_nl=sid, contract=Contract(),
                 code=code, parent_ids=[], created_round=0, status="active")


cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
lib = SkillLibrary()
lib.add(mk("det", "diagnose", DETECT)); lib.add(mk("ord", "order", ORDER))
lib.add(mk("bld", "construct", BUILD))           # order makes per-instance value VARY -> discriminative
pool = build_holdout(cfg, k=4)                    # reuse holdout instances as a dev pool

gm = gap_matrix(lib, pool, cfg)
assert len(gm["base"]) == 4 and set(gm["deltas"]) == {"det", "ord", "bld"}
assert all(len(v) == 4 for v in gm["deltas"].values()), "each skill row spans the pool"
scores = discrimination_scores(gm)
assert len(scores) == 4 and all(s >= 0 for s in scores), scores
assert any(s > 0 for s in scores), "with an order skill, some instances must discriminate skills"
print("gap-matrix base gaps:", [f"{g:.1%}" if g is not None else "NA" for g in gm["base"]])
print("discrimination scores:", [f"{s:.3f}" for s in scores])

sel, _, _ = discriminative_dev(lib, pool, cfg)
assert len(sel) == cfg["budgets"]["dev_eval_k"], f"select dev_eval_k instances, got {len(sel)}"
print(f"OK §4: gap matrix persisted-shape, discrimination scored, selected {len(sel)} dev instances.")
