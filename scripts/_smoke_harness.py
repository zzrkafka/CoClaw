"""Smoke: §1.2 fixed-harness recovery stage reproduces the prior inline notes byte-for-byte
(behavior-preserving refactor), and the debug_hint seam inserts in the right place. No LLM/LKH."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.harness import STAGES, recovery_note   # noqa: E402


def OLD(remaining, obs_error, best_len):
    """The exact note logic that lived inline in solver.solve_instance before the refactor."""
    note = f"\n[{remaining} action(s) left.]"
    if obs_error and "budget" in obs_error.lower():
        note += (" Your cost-lookup budget is EXHAUSTED -- you can no longer read C."
                 " Set FINAL_TOUR (your best) and DONE now; that does not read C.")
    if best_len is not None and obs_error == "kernel reset":
        note += f" Your best tour so far had length {best_len}; rebuild and beat it."
    if remaining <= 1:
        note += (" Final actions -- ensure FINAL_TOUR holds your BEST valid tour"
                 " (a permutation of range(n)).")
    return note


CASES = [
    (5, None, None), (5, "budget exceeded", 100), (0, "kernel reset", 200),
    (1, None, None), (3, "some error", 50), (1, "budget overrun", 7), (0, "kernel reset", None),
]
for rem, err, bl in CASES:
    got = recovery_note(remaining=rem, obs_error=err, best_len=bl)
    want = OLD(rem, err, bl)
    assert got == want, f"recovery_note diverged for {(rem, err, bl)}:\n got={got!r}\nwant={want!r}"

# debug_hint seam: appears, and BEFORE the final-actions clause (so the closing instruction stays last)
hinted = recovery_note(remaining=1, obs_error=None, best_len=None, debug_hint="TRY cheaper insertion.")
assert "TRY cheaper insertion." in hinted
assert hinted.index("TRY cheaper insertion.") < hinted.index("Final actions"), "hint must precede final-actions"
assert recovery_note(remaining=5, obs_error=None, best_len=None, debug_hint="") == OLD(5, None, None)

assert STAGES[0] == "observe" and STAGES[-1] == "submit" and "recover" in STAGES

print("OK §1.2: recovery stage byte-identical to prior inline notes; debug_hint seam placed correctly.")
