"""Lessons + supervisor -- the experience every run deposits (success OR failure).

Motivation: we measured DeepSeek reliably writing a CORRECT cluster-order construction
(~1.4% gap) and then ruining it with a textbook 2-opt using the SYMMETRIC delta on an
ASYMMETRIC cost matrix, run to convergence (-> ~37%). That is a recurring LOW-LEVEL error.
This module gives the system a memory of such errors:

  - SEED_RULES   : hand-written basic rules injected into the induction prompt to prevent the
                   known low-level mistakes up front.
  - lint_skill   : the SUPERVISOR -- a cheap static scan for anti-patterns BEFORE expensive
                   dev evaluation. Returns (hard_errors, warnings); hard_errors reject pre-eval.
  - load/append  : DEPOSITED lessons persisted per run; whenever a proposal trips the lint or
                   regresses on dev, a lesson is appended so the next round avoids it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SEED_RULES = [
    # MINIMAL generic bottom-lines only. The SPECIFIC bug recipes (asymmetric-2-opt mechanics,
    # C[i,j] vs C[i][j], per-cluster NN) are deliberately NOT seeded here -- the system must
    # REDISCOVER them from grounded failures (debugger -> distilled lesson). The lint detectors
    # below still flag them as harness smells, so the signal reaches the debugger without me
    # hand-feeding the answer.
    {"id": "trust-C", "rule":
        "C is the objective: an n x n INTEGER cost matrix that is ASYMMETRIC and may VIOLATE the "
        "triangle inequality. Trust C, not the auxiliary coords x. Minimise the closed-tour sum "
        "of C[t[i], t[i+1]]."},
    {"id": "decompose", "rule":
        "Propose the constructor and EACH local-search as SEPARATE typed skills, never one "
        "monolithic strategy. Each is judged independently, so a buggy step is rejected on its own "
        "and cannot sink your good constructor (partial credit)."},
    {"id": "valid-perm", "rule":
        "A construct/strategy MUST return a permutation of ALL range(n) (each node exactly once). "
        "Verify with prims.is_valid_tour(tour, n) before returning."},
    {"id": "bound-loops", "rule":
        "Bound every loop to a small fixed number of passes; never run to convergence."},
    {"id": "name-f", "rule":
        "The function MUST be named exactly `f` and match its kind's signature exactly."},
]

# heuristics for the static supervisor
_RE_REVERSE = re.compile(r"\[\s*::\s*-\s*1\s*\]|reversed\s*\(")
# leading (?<![A-Za-z_0-9]) so the cost matrix `C` matches but NOT vars ending in C (e.g. IC, dist_C)
_RE_TWO_COSTS = re.compile(r"(?<![A-Za-z_0-9])C\[[^\]\n]+\]\s*(?:\[[^\]\n]+\]\s*)?\+\s*C\[")  # C[a][c]+C[..
# the dangerous convergence idiom (run-to-fixpoint), NOT a draining loop like `while rem:`
_RE_UNBOUNDED = re.compile(r"while\s+(True|improved\b|not\s)")
_RE_ITERCAP = re.compile(r"max_iter|for\s+_?\w*\s+in\s+range\(\s*\d+\s*\)")
# C[i][j] reads a whole ROW (n cost-evals); C[i, j] reads one -- the row form blows the budget.
# leading (?<![A-Za-z_0-9]) so IC[a][b] / dist_C[a][b] (plain matrices) do NOT false-trigger.
_RE_ROWINDEX = re.compile(r"(?<![A-Za-z_0-9])C\s*\[[^\]\n]+\]\s*\[[^\]\n]+\]")


def lint_skill(kind: str, code: str) -> tuple[list[str], list[str]]:
    """Static supervisor. (hard_errors, warnings). hard_errors -> reject before dev eval."""
    hard: list[str] = []
    warn: list[str] = []
    if not code or "def f(" not in code:
        hard.append("function is not named `f` (rule name-f)")
        return hard, warn
    reverses = bool(_RE_REVERSE.search(code))
    two_costs = bool(_RE_TWO_COSTS.search(code))
    if reverses and two_costs:
        warn.append("LIKELY the asymmetric-2-opt bug (rule asym-2opt): you reverse a tour segment "
                    "and score it with a symmetric 2-opt delta. On asymmetric C this DEGRADES the "
                    "tour. Omit/bound the local search, or recompute prims.tour_length on the full "
                    "tour to accept a move.")
    if _RE_UNBOUNDED.search(code) and not _RE_ITERCAP.search(code):
        warn.append("unbounded `while` loop (rule bound-loops): cap the iterations.")
    if _RE_ROWINDEX.search(code):
        warn.append("reads cost as C[i][j] (rule cost-access): C[i] materializes a whole ROW "
                    "(n cost-evals); use C[i, j]. The row form inflates n_evals ~n-fold and can "
                    "blow the eval budget, getting a correct skill rejected as eval-hungry.")
    if kind in ("construct", "strategy") and "is_valid_tour" not in code:
        warn.append("no prims.is_valid_tour check before returning (rule valid-perm).")
    if kind == "strategy":                      # §2.4 L2 red line: a strategy stays a THIN dispatcher
        from agent.growth import strategy_redline
        ok, reason = strategy_redline(code)
        if not ok:
            hard.append(reason)
    return hard, warn


def load_lessons(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def append_lesson(path: str | Path | None, lesson: dict) -> None:
    """Append a lesson, deduped by id (first writer wins)."""
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = {l.get("id") for l in load_lessons(p)}
    if lesson.get("id") in existing:
        return
    lesson.setdefault("votes", 0)
    with p.open("a") as f:
        f.write(json.dumps(lesson) + "\n")


def lessons_block(deposited: list[dict] | None = None) -> str:
    rows = ["RULES (follow these to avoid known low-level errors):"]
    for r in SEED_RULES:
        rows.append(f"- [{r['id']}] {r['rule']}")
    dep = deposited or []
    if dep:
        rows.append("LESSONS deposited from previous runs (do NOT repeat these mistakes):")
        for l in dep[-8:]:
            rows.append(f"- {l.get('rule', '')}")
    return "\n".join(rows)


# --- lesson lifecycle (item 3: ExpeL voting + AutoGuide conditional retrieval) ---
_LESSON_RETIRE = -3   # votes at/below this -> the lesson is dropped (it never actually helps)


def write_lessons(path: str | Path | None, lessons: list[dict]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(l) + "\n" for l in lessons))


def vote_lessons(path: str | Path | None, ids, delta: int) -> None:
    """ExpeL: +1 to a lesson when a skill authored WITH it in context succeeded, -1 when it
    failed. Lessons that fall to _LESSON_RETIRE are dropped (they don't earn their keep)."""
    ids = {i for i in (ids or []) if i}
    if not path or not ids:
        return
    keep, changed = [], False
    for l in load_lessons(path):
        if l.get("id") in ids:
            l["votes"] = l.get("votes", 0) + delta
            changed = True
        if l.get("votes", 0) > _LESSON_RETIRE:
            keep.append(l)
        else:
            changed = True   # retired
    if changed:
        write_lessons(path, keep)


def retrieve_lessons(deposited: list[dict], kind: str | None = None, k: int = 6) -> list[dict]:
    """AutoGuide: return only the lessons whose condition matches the current context (same skill
    kind, or universal), ranked by votes then recency -- instead of dumping every lesson."""
    cands = [l for l in (deposited or [])
             if (l.get("condition") or {}).get("kind") in (None, kind)]
    cands.sort(key=lambda l: (l.get("votes", 0), l.get("round", -1)), reverse=True)
    return cands[:k]
