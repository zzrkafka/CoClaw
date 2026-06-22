"""Two-axis library curation (improvement-plan §3.1 + §3.2 + §3.3) -- the credit-assignment core.

Diagnosis (from the hard_split run): order/diagnose were admitted on probation (validity only),
lesion measured marginal value but NOTHING acted on it, and the planner rarely merged -> the library
bloated (19 active, only `order` carried value, gap crept back up after r6). The fix is to actually
DECIDE on two measured axes per skill and act:

  reuse_rate   (is it CALLED?)            x   marginal_value (does REMOVING it raise holdout gap?)

      |                 marginal ~0 (eps)        marginal > 0 (valuable)      marginal < 0
  ----+-------------------------------------------------------------------------------------
  low | PRUNE (dead weight)                       INVESTIGATE (valuable but        PRUNE
  reuse|  -- unless within tenure                  not retrieved/orchestrated;      (harmful,
      |                                            NEVER prune -- wire it up)       immediate)
  ----+-------------------------------------------------------------------------------------
  high| MERGE (redundant: called, but              KEEP                            PRUNE
  reuse| removing it costs nothing)                                                (harmful)

TENURE (§3.2): a skill younger than `tenure_k` rounds is shielded -- a brand-new operator can look
"dead" only because retrieval/orchestration has not picked it up yet. Within tenure we downgrade a
would-be PRUNE to INVESTIGATE. A genuinely HARMFUL skill (marginal < -eps) is pruned regardless of
tenure: it actively hurts.

The §3.3 hook (`apply_curation`) acts on PRUNE verdicts only -- the lowest-cost, highest-confidence
first step (zero extra LLM cost; reuses the marginal values lesion already computed each round).
MERGE/INVESTIGATE are recorded as labels; merging is handled by the QD pass (§3.5).
"""
from __future__ import annotations

KEEP, MERGE, PRUNE, INVESTIGATE = "keep", "merge", "prune", "investigate"


def _params(cfg: dict) -> tuple[float, float, int]:
    cur = (cfg or {}).get("curation", {})
    return (cur.get("prune_mv_eps", 0.001),
            cur.get("prune_reuse_max", 0.34),
            cur.get("tenure_k", 3))


def curation_decision(skill, round_idx: int, cfg: dict) -> str:
    """One verdict in {keep, merge, prune, investigate} for a skill, from the two measured axes plus
    the tenure shield. Pure function of the skill's ReuseStats + config; does not mutate anything."""
    eps, reuse_max, tenure_k = _params(cfg)
    leave_one_in = (cfg or {}).get("curation", {}).get("leave_one_in", False)
    mv = skill.reuse.marginal_value
    reuse = skill.reuse.reuse_rate
    age = round_idx - skill.reuse.created_round         # rounds since the skill entered the library
    within_tenure = age < tenure_k
    # §3.4: a skill weak alone but valuable in combination (siblings mask its leave-one-out marginal)
    # shows positive LEAVE-ONE-IN value -> protect it from pruning.
    combo_valuable = leave_one_in and skill.reuse.marginal_add > eps

    if mv < -eps:                                       # actively harmful -> prune now, tenure or not
        return PRUNE
    dead = abs(mv) <= eps                               # marginal ~ 0
    low_reuse = reuse <= reuse_max
    if dead and low_reuse:
        if within_tenure or combo_valuable:             # §3.2 young, or §3.4 combo-valuable -> spare
            return INVESTIGATE
        return PRUNE
    if dead and not low_reuse:
        return MERGE                                    # reused yet redundant -> merge candidate
    if (not dead) and low_reuse:
        return INVESTIGATE                              # valuable but not wired up -> NEVER prune
    return KEEP                                         # valuable AND reused


def decide_library(lib, round_idx: int, cfg: dict) -> dict:
    """Verdict for every active skill: {skill_id: decision}. Read-only (for records / inspection)."""
    return {s.id: curation_decision(s, round_idx, cfg) for s in lib.active()}


def apply_curation(lib, round_idx: int, cfg: dict) -> dict:
    """§3.3 hook: prune the PRUNE-verdict skills; return a summary for the run record.

    Call AFTER lesion has populated marginal_value this round. Honors tenure + the harmful-immediate
    rule via curation_decision. Returns counts + the pruned/labeled ids so the metric record can show
    the library staying lean instead of bloating."""
    if not (cfg or {}).get("curation", {}).get("prune", False):
        return {"enabled": False, "decisions": {}, "pruned": []}
    decisions = decide_library(lib, round_idx, cfg)
    pruned = [sid for sid, d in decisions.items() if d == PRUNE]
    for sid in pruned:
        lib.prune(sid, round_idx)
    counts = {v: sum(1 for d in decisions.values() if d == v)
              for v in (KEEP, MERGE, PRUNE, INVESTIGATE)}
    return {"enabled": True, "counts": counts, "pruned": pruned,
            "decisions": decisions, "n_active_after": len(lib.active())}
