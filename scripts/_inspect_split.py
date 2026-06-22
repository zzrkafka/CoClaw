"""Inspect a split run: accepted atoms + lesion marginal_value, frac_general, ops, reuse."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1] if len(sys.argv) > 1 else "hard_split_pilot"
d = ROOT / "runs" / run


def load(name):
    p = d / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


final = {}
for s in load("skill_snapshot"):
    final[s["skill_id"]] = s   # last snapshot wins

print(f"=== {run}: final library ===")
print(f"{'skill_id':38} {'kind':10} {'marginal':>9} {'reuse_rt':>8} {'invoc':>6} {'status':>9}")
for sid, s in final.items():
    r = s.get("reuse", {})
    print(f"{sid:38.38} {s['kind']:10} {r.get('marginal_value', 0):>+8.2%} "
          f"{r.get('reuse_rate', 0):>8.2f} {r.get('invocations', 0):>6} {s.get('status'):>9}")

print("\n=== metrics per round ===")
for m in load("metric_record"):
    print(f"round {m['round_idx']}: mean_gap={m.get('mean_gap')} inst_gap={m.get('instance_gap')} "
          f"n_active={m.get('n_active_skills')} frac_general={m.get('frac_general')}")

print("\n=== evolve ops ===")
for e in load("evolve_record"):
    print(f"-- after {e.get('after_instance_id')}:")
    for o in e.get("ops", []):
        j = o.get("judge", {})
        print(f"   {o.get('op')}/{o.get('kind')}: accepted={j.get('accepted')} "
              f"via={j.get('accepted_via')} probation={j.get('probation')} "
              f"cand_gap={j.get('cand_gap')} skill_id={o.get('skill_id')}")

print("\n=== reuse (skills the agent actually called per solve) ===")
for sr in load("solve_record"):
    calls = [sid for a in sr["actions"] for sid in a.get("skills_called", [])]
    g = sr["best_gap"]
    print(f"   {sr['instance_id']}: gap={g:.2%} calls={len(calls)} distinct={len(set(calls))} "
          f"{dict(Counter(calls))}")
