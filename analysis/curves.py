"""Analysis curves + H2 ablation comparison (spec section 13).

The compounding question: does an accumulating library (mem) beat a reset one (no-mem) on
quality (gap) and/or efficiency (total_evals -- the F4 foresight signal), and is the
library actually reused (skills_called)?

  python -m analysis.curves --mem ablation_mem --nomem ablation_nomem
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def _load(run_id: str, kind: str, root="runs") -> list[dict]:
    p = Path(root) / run_id / f"{kind}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def arm_curve(run_id: str, root="runs") -> list[dict]:
    """Per-instance row: gap, total_evals, #skill-calls by the agent, library size."""
    solves = _load(run_id, "solve_record", root)
    metrics = {m["round_idx"]: m for m in _load(run_id, "metric_record", root)}
    rows = []
    for i, s in enumerate(solves):
        calls = sum(len(a.get("skills_called", [])) for a in s["actions"])
        distinct = len({sid for a in s["actions"] for sid in a.get("skills_called", [])})
        rows.append({
            "round": i, "instance_id": s["instance_id"], "gap": s["best_gap"],
            "total_evals": s["total_evals"], "skill_calls": calls,
            "distinct_skills_used": distinct,
            "n_active_skills": metrics.get(i, {}).get("n_active_skills", 0),
        })
    return rows


def _halves(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    h = max(1, len(vals) // 2)
    return mean(vals[:h]), mean(vals[h:])


def _fmt_gap(g):
    return f"{g:.2%}" if g is not None else "  NA "


def compare(mem_id: str, nomem_id: str, root="runs") -> None:
    mem, nomem = arm_curve(mem_id, root), arm_curve(nomem_id, root)
    n = min(len(mem), len(nomem))
    print("=" * 78)
    print(f"H2 ablation: {mem_id} (library accumulates) vs {nomem_id} (library reset each step)")
    print("=" * 78)
    print(f"{'rnd':>3} | {'MEM gap':>8} {'evals':>9} {'calls':>5} {'lib':>3} | "
          f"{'NOMEM gap':>9} {'evals':>9} {'calls':>5}")
    print("-" * 78)
    for i in range(n):
        m, nm = mem[i], nomem[i]
        print(f"{i:>3} | {_fmt_gap(m['gap']):>8} {m['total_evals']:>9} {m['skill_calls']:>5} "
              f"{m['n_active_skills']:>3} | {_fmt_gap(nm['gap']):>9} {nm['total_evals']:>9} "
              f"{nm['skill_calls']:>5}")
    print("-" * 78)

    for label, key, fmt in (("gap", "gap", lambda v: f"{v:.2%}" if v is not None else "NA"),
                            ("evals", "total_evals", lambda v: f"{v:,.0f}" if v is not None else "NA")):
        m_first, m_last = _halves([r[key] for r in mem])
        n_first, n_last = _halves([r[key] for r in nomem])
        print(f"\n[{label}] first-half -> last-half")
        print(f"   MEM  : {fmt(m_first)} -> {fmt(m_last)}")
        print(f"   NOMEM: {fmt(n_first)} -> {fmt(n_last)}")
        if m_last is not None and n_last is not None:
            better = "MEM better" if m_last < n_last else ("NOMEM better" if n_last < m_last else "tie")
            print(f"   last-half {label}: {better}")

    mem_calls = sum(r["skill_calls"] for r in mem)
    nomem_calls = sum(r["skill_calls"] for r in nomem)
    mem_final_lib = mem[-1]["n_active_skills"] if mem else 0
    print(f"\n[reuse] MEM total skill-calls={mem_calls} (final library {mem_final_lib} skills) | "
          f"NOMEM total skill-calls={nomem_calls}")
    print("\nVERDICT GUIDE: compounding = MEM last-half gap AND/OR evals below NOMEM, with"
          " MEM skill-calls > 0 and rising. Collapse = MEM ~ NOMEM (library is dead weight).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", required=True)
    ap.add_argument("--nomem", required=True)
    ap.add_argument("--root", default="runs")
    args = ap.parse_args()
    compare(args.mem, args.nomem, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
