"""Directed library growth (improvement-plan §2.2 typed-hole authoring + §2.4 two-level insufficiency).

The seed strategy (agent/seeds.py) has TYPED HOLES -- one per operator kind it dispatches on. §2.2:
each unfilled hole PRECISELY specifies which operator is missing, so we can author exactly that one
(focused author -> grounded verify -> into the library, reusable) instead of waiting for the
trajectory planner to rediscover the need. This complements the planner: planner = discovery from
solves, fill_holes = a guarantee the scaffold's holes get filled.

§2.4 two levels of "insufficient":
  L1 (cheap, common): a hole is missing / weak -> AUTHOR an operator of that kind (this module).
  L2 (rare, guarded): the PLAN STRUCTURE itself is wrong -> a BOUNDED modify of the strategy. The
     red line (INV-1/2, EoH): a strategy must stay a THIN DISPATCHER over typed operators, never
     inline a monolithic solver. `strategy_redline` enforces that so L2 can't bloat into program
     synthesis. The planner is told to prefer L1 and keep any L2 modify thin (reflect_induce prompt).
"""
from __future__ import annotations

# The seed strategy's typed holes, in dependency order (detect -> order -> build -> improve).
STRATEGY_HOLES = ("diagnose", "order", "construct", "local_search")

_HOLE_INTENT = {
    "diagnose": "DETECT the cost structure: return {'labels': per-node group id, 'k': K} so order/"
                "build can exploit it.",
    "order": "Solve the cheap DIRECTED visit order over the few groups (brute-force the small "
             "inter-group cost matrix); return the group ids in that order.",
    "construct": "BUILD a tour from the given struct + order (visit groups in order, nearest within "
                 "each); return a permutation of range(n).",
    "local_search": "ONE bounded improvement over the built tour; accept a move only if the FULL "
                    "prims.tour_length drops (asymmetric C); never run to convergence.",
}


def hole_intent(kind: str) -> str:
    return _HOLE_INTENT.get(kind, "(lower future gap)")


def unfilled_holes(lib, holes=STRATEGY_HOLES) -> list[str]:
    """Hole kinds with NO active operator -- the precise, typed list of what to author next (§2.2)."""
    present = {s.kind for s in lib.active()}
    return [k for k in holes if k not in present]


def weak_holes(lib, cfg, holes=STRATEGY_HOLES) -> list[str]:
    """Holes whose best operator carries little marginal value (≈0) -> candidates to RE-author a
    stronger one. Uses whatever marginal_value lesion has populated (0 if not yet measured)."""
    eps = (cfg or {}).get("curation", {}).get("prune_mv_eps", 0.001)
    out = []
    for k in holes:
        ops = [s for s in lib.active() if s.kind == k]
        if ops and max(s.reuse.marginal_value for s in ops) <= eps:
            out.append(k)
    return out


def slot_fill_ratio(lib, cfg, holes=STRATEGY_HOLES) -> float:
    """§2.5 self-bootstrap signal: the fraction of strategy slots (holes) filled by a VALUABLE
    operator -- one whose best leave-one-out OR leave-one-in marginal exceeds eps. Rising over rounds
    = the library is converging onto the scaffold's slots = compounding (as opposed to accreting
    dead variants). Returns a value in [0, 1]; 0 if there are no holes."""
    eps = (cfg or {}).get("curation", {}).get("prune_mv_eps", 0.001)
    if not holes:
        return 0.0
    filled = 0
    for k in holes:
        ops = [s for s in lib.active() if s.kind == k]
        if ops and max(max(s.reuse.marginal_value, s.reuse.marginal_add) for s in ops) > eps:
            filled += 1
    return filled / len(holes)


def strategy_redline(code: str, max_lines: int = 80) -> tuple[bool, str]:
    """§2.4 L2 guard: a strategy must DISPATCH to operators and stay thin. Returns (ok, reason).
    ok=False => reject the strategy add/modify (it crosses the inline-solver red line). Counts only
    code lines (comments excluded); the threshold blocks a real inline monolith (the one that
    motivated this was 121 lines) while allowing a dispatcher plus its safe fallbacks."""
    code = code or ""
    loc = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    dispatches = ("skills[" in code) or ("of_kind" in code)   # skills["id"](...) or skills.of_kind(...)
    if not dispatches:
        return False, ("a strategy must DISPATCH to typed operators (skills.of_kind / skills['id']), "
                       "not inline its own solver -- move the logic into operators (red line)")
    if len(loc) > max_lines:
        return False, (f"strategy is {len(loc)} code lines (> {max_lines}); keep it a THIN dispatcher "
                       "and push logic into typed operators (no inline monolith)")
    return True, ""


def fill_holes(lib, dev_set, cfg, llm, lessons_path, round_idx, *, induce, lblock="",
               holes=STRATEGY_HOLES) -> list[dict]:
    """§2.2/§2.4-L1: author exactly the operators the seed strategy's holes need. `induce` is the
    induction fn (evolve.induce_skill) injected so this is testable; it returns (code, res, accepted).
    Adds each accepted operator to the library. No-op unless library.fill_holes is on. Dependency
    order is honored by STRATEGY_HOLES, so an `order` exists before `construct` is judged."""
    if not (cfg or {}).get("library", {}).get("fill_holes", False):
        return []
    from agent.evolve import materialize_skill
    fam = (cfg.get("problem") or {}).get("family", "F")
    records = []
    for kind in unfilled_holes(lib, holes):             # snapshot of holes before we start filling
        op = {"op": "add", "kind": kind, "description_nl": hole_intent(kind)}
        code, res, accepted = induce(op, lib, dev_set, cfg, llm, lblock, round_idx, lessons_path)
        sid = None
        if accepted and code:
            sid = lib.add(materialize_skill(dict(op, code=code, origin_family=fam), round_idx), round_idx)
        records.append({"hole": kind, "skill_id": sid, "accepted": bool(sid)})
    return records
