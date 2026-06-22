"""Seed strategy scaffold (improvement-plan §2.1 mixed-C generation + §2.3 graceful degradation).

Generation is neither pure-seed nor pure-emergent: we seed ONE THIN strategy skeleton that gives
DIRECTION (detect -> order -> build -> bounded-improve), with each stage a TYPED HOLE the agent fills
by inducing/replacing operators of that kind. Every stage DEGRADES GRACEFULLY (default struct /
order=None / NN fallback), so a missing or broken operator still yields a LEGAL tour whose exact gap
is measurable -- that drives hole-filling instead of crashing (the monolithic self-induction that
mostly failed in practice). The scaffold is the `code`; the `plan` is the declarative steer (§1.3).

The skeleton dispatches by KIND via skills.of_kind / skills.first_of_kind (sandbox SkillDict), so it
composes whatever operators the library currently holds without hard-coding ids -- as the library
fills, the same scaffold gets stronger. Seeded at warm-start ONLY when library.seed_strategy is on.
"""
from __future__ import annotations

from skills.schema import Contract, Skill

SEED_STRATEGY_PLAN = (
    "Diagnose what drives cost, solve the cheap visit ORDER over the few groups, BUILD in that "
    "order, then ONE bounded improvement. Use a library operator for each step if present, else "
    "degrade safely. Trust the measured tour cost, not appearance; never run search to convergence."
)

SEED_STRATEGY_CODE = r'''
def f(n, x, C, skills, prims):
    # THIN scaffold with typed holes; every stage degrades gracefully so the tour is always legal.
    _first = getattr(skills, "first_of_kind", lambda k: None)
    # --- detect (hole: diagnose) -> structure; fall back to cost clustering, then trivial ---
    struct = None
    det = _first("diagnose")
    if det is not None:
        try:
            d = det(list(range(n)), x, C, prims)
            if isinstance(d, dict) and "labels" in d:
                struct = d
        except Exception:
            struct = None
    if struct is None:
        try:
            lab = prims.cluster_by_cost(C)
            struct = {"labels": [int(v) for v in lab], "k": len(set(int(v) for v in lab))}
        except Exception:
            struct = {"labels": [0] * n, "k": 1}
    # --- order (hole: order) -> cheap directed visit order; None is a valid fallback ---
    order = None
    od = _first("order")
    if od is not None:
        try:
            order = list(od(struct, C, prims))
        except Exception:
            order = None
    # --- build (hole: construct) -> tour; else nearest-neighbour (always legal) ---
    tour = None
    bd = _first("construct")
    if bd is not None:
        try:
            tour = prims.sanitize_tour(bd(n, x, C, prims, struct, order), n, C)
        except TypeError:
            try:
                tour = prims.sanitize_tour(bd(n, x, C, prims), n, C)
            except Exception:
                tour = None
        except Exception:
            tour = None
    if tour is None:
        visited = [False] * n; tour = [0]; visited[0] = True
        for _ in range(n - 1):
            last = tour[-1]; best = -1; bc = None
            for j in range(n):
                if not visited[j]:
                    c = C[last, j]
                    if bc is None or c < bc:
                        bc = c; best = j
            tour.append(best); visited[best] = True
    # --- bounded improve (hole: local_search) -> keep ONLY if it strictly lowers the full cost ---
    ls = _first("local_search")
    if ls is not None:
        try:
            nt = prims.sanitize_tour(ls(list(tour), C, prims), n, C)
            if prims.tour_length(nt, C) < prims.tour_length(tour, C):
                tour = nt
        except Exception:
            pass
    if not prims.is_valid_tour(tour, n):
        tour = prims.sanitize_tour(tour, n, C)
    return tour
'''


def seed_strategy_skill(round_idx: int = -1) -> Skill:
    return Skill(
        id="seed_strategy", kind="strategy", version=1, name="seed_strategy",
        description_nl="Thin detect->order->build->improve scaffold; composes library operators by "
                       "kind with graceful fallback at each typed hole.",
        contract=Contract(), code=SEED_STRATEGY_CODE, parent_ids=[], created_round=round_idx,
        status="active", plan=SEED_STRATEGY_PLAN, meta={"seed": True},
    )


def seed_library(lib, cfg: dict, round_idx: int = -1) -> str | None:
    """Add the thin seed strategy iff library.seed_strategy is on and no strategy exists yet.
    Returns the seeded skill id, or None (no-op). Call at warm-start, before set_seed."""
    if not (cfg or {}).get("library", {}).get("seed_strategy", False):
        return None
    if any(s.kind == "strategy" for s in lib.active()):
        return None
    return lib.add(seed_strategy_skill(round_idx), by_round=round_idx)
