"""Top-level controller playbook -- the FIXED harness SOP (improvement-plan §1.5).

The engineer's workflow is the right skeleton, re-weighted for how an LLM fails: force the steps
it skips (probe / measure the TRUE objective / keep a rolling best / bound every loop), hedge its
training reflexes (don't assume the textbook form of the problem -- verify against the actual cost),
and play to its strengths (one focused decision at a time, externalize observations).

DISCIPLINE -- MINIMAL SEED, GROUNDED COMPLETION. This is deliberately a GENERAL method skeleton
with NO problem-specific answers baked in (no "use clusters", no "asymmetric 2-opt delta", no magic
thresholds). Those are the conclusions we want the system to REDISCOVER from grounded failures
(agent/lessons.py self-distillation) and deposit as per-type plans/lessons -- pre-loading them would
bake the result we are trying to measure. cf. the same rule in lessons.SEED_RULES.

The §1.5 split lives here as three artifacts:
  HARNESS_PLAYBOOK       -- the fixed engine's SOP, injected into the solve prompt (this file).
  PER_TYPE_SKILL_TEMPLATE-- what a per-type strategy "knowledge card" must contain (feeds §2.1 seed
                            + the reflect planner): dispatch / how-to-strategize / pitfalls / debug.
  DEBUG_SKILL_SPEC       -- the recovery SOP a debug skill grounds + evolves against (feeds §1.4).
"""
from __future__ import annotations

# The general controller, surfaced to the agent each instance when harness.use_playbook is on.
# Imperative + short. Every line targets a measured LLM failure mode, not a domain fact.
HARNESS_PLAYBOOK = """\
HOW TO APPROACH THIS INSTANCE (general playbook -- the exact objective is the ONLY arbiter):
1. READ THE TRUE OBJECTIVE FIRST. Optimize the cost you are actually scored on, not a convenient-
   looking proxy (e.g. visible coordinates / "obvious" geometry). Confirm what really drives cost
   before you act -- the clean-looking representation may be a decoy.
2. DIAGNOSE CHEAPLY before optimizing: size? symmetric or not? does the triangle inequality hold?
   is there block / group structure? WHICH layer dominates cost -- per-element moves, or a higher-
   level ordering / assignment?
3. ATTACK THE DOMINANT LAYER. If a small sub-problem dominates, solve THAT small core well (exact /
   brute-force over the FEW elements only -- never factorial over all nodes). A local move on the
   wrong layer cannot fix it; that is the classic way to get stuck far from optimal.
4. BASELINE + BOUND. Get a cheap feasible solution early and keep a ROLLING BEST (never lose a good
   answer). Bound every loop to a small fixed number of passes -- spend budget on structure, not on
   brute force; never run to convergence.
5. VERIFY EVERY STEP against the MEASURED objective, not against how clever a move "sounds". A move
   that reads as smart can secretly be a no-op or a regression -- trust the recomputed cost only.
6. ON FAILURE, RECOVER -- do not discard. Repair a near-feasible solution instead of throwing it
   away; if you are stuck at a bad cost, RE-DIAGNOSE (you may be optimizing the wrong layer); else
   fall back to a safe baseline. Submit the best feasible solution you have."""

# What a per-type strategy carries (a "knowledge card"). Used to shape strategy plans (§2.1/§5.3):
# discipline = inherit a shared schema, do NOT become an island; dispatch on STRUCTURAL regime
# (e.g. "order dominates") not a textbook problem name; evolve by MEASURED consequence, not NL
# self-reflection.
PER_TYPE_SKILL_TEMPLATE = {
    "dispatch": "which operators to call, in what order, for THIS structural regime",
    "strategize": "how to set the plan once the regime is diagnosed",
    "pitfalls": "what this regime tempts you to get wrong (filled by distilled lessons, not seeded)",
    "debug": "how to recover when a step regresses or fails here (-> the debug skill)",
}

# The recovery SOP a debug skill evolves against; its grounded signal is the gap BEFORE vs AFTER the
# repair it prescribes (the cleanest credit signal -- §1.4). Complementary to pitfalls: pitfalls
# AVOID a mistake up front, the debug skill makes the best of one after it happens, and a newly
# discovered failure mode distills back into a pitfall lesson.
DEBUG_SKILL_SPEC = {
    "inputs": "a failed/regressed attempt + grounded evidence (status, hang trace, per-dev gap)",
    "output": "ONE root cause + ONE minimal fix directive (diagnose, do not rewrite)",
    "grounded_by": "repair gap delta = gap(before) - gap(after); accept only a real improvement",
}


def harness_playbook(cfg: dict | None = None) -> str:
    """Return the playbook text iff harness.use_playbook is on, else "" (baseline solve unchanged)."""
    if cfg and cfg.get("harness", {}).get("use_playbook", False):
        return HARNESS_PLAYBOOK
    return ""
