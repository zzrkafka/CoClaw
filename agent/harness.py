"""The FIXED engine (improvement-plan §1.2) -- problem-agnostic, does NOT evolve.

Two loops must never be conflated:
  INNER (this engine, FIXED): drive ONE instance to a submitted solution.
  OUTER (agent/evolve.py, LEARNS): grow + curate the skill library across instances.

The engine owns the problem-agnostic driving: observe -> act -> check feasibility -> measure the
TRUE objective -> RECOVER on error -> track the rolling best -> submit; plus budget / sandbox /
grounding / retry. It is code and it is frozen, which does NOT cross the EoH red line: EoH freezes
the solving *heuristic*; we freeze the generic *engine* and let the heuristic (the skill library)
evolve. Everything problem-specific lives behind `prims` / the problem adapter (§5.4a), not here.

This module makes that boundary explicit and gives RECOVERY its own pluggable stage (so the §1.4
debug skill has a seam to plug into) instead of being scattered inline in the solve loop.
"""
from __future__ import annotations

# The fixed inner-loop stages (documentation + lets records/analysis name the engine contract).
STAGES = ("observe", "act", "check_feasible", "measure_objective", "recover",
          "track_rolling_best", "submit")


def recovery_note(*, remaining: int, obs_error: str | None, best_len: int | None,
                  debug_hint: str = "") -> str:
    """The RECOVER stage: build the feedback note appended after each action (progress + recovery
    guidance). `debug_hint` is the seam the §1.4 debug skill fills; empty by default, so with no
    debug skill this reproduces the prior inline notes byte-for-byte."""
    note = f"\n[{remaining} action(s) left.]"
    if obs_error and "budget" in obs_error.lower():
        note += (" Your cost-lookup budget is EXHAUSTED -- you can no longer read C."
                 " Set FINAL_TOUR (your best) and DONE now; that does not read C.")
    if best_len is not None and obs_error == "kernel reset":
        note += f" Your best tour so far had length {best_len}; rebuild and beat it."
    if debug_hint:
        note += " " + debug_hint.strip()
    if remaining <= 1:
        note += (" Final actions -- ensure FINAL_TOUR holds your BEST valid tour"
                 " (a permutation of range(n)).")
    return note
