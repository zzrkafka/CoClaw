"""Integration: run the full §3 curation pipeline on the REAL hard_split library (no API).

Demonstrates the plan's core claim -- "fix the criterion and the library stops bloating": load the
19-active-skill hard_split library, compute fresh leave-one-out + leave-one-in credit (exact LKH
gap), then apply the 2-axis prune (tenure satisfied since we pass a late round) + QD dedupe. Reports
what survives. This is the offline half of §3.3; the on-line A/B is a real arm run with v2 config."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.curation import apply_curation, decide_library          # noqa: E402
from analysis.lesion import (build_holdout, lesion_marginal_add,      # noqa: E402
                            lesion_marginal_values, load_library_from_run)
from analysis.qd import qd_dedupe                                     # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "hard_split"
cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
# v2-style curation: prune on, leave-one-in on, QD on. Late round so tenure is satisfied for all.
cfg["curation"] = {"prune": True, "leave_one_in": True, "qd_niche": True,
                   "tenure_k": 3, "prune_mv_eps": 0.001, "prune_reuse_max": 0.34}
LATE = 999

lib = load_library_from_run(ROOT / "runs" / RUN)
before = len(lib.active())
if before == 0:
    print(f"[skip] {RUN}: no active skills to curate.")
    sys.exit(0)
print(f"[{RUN}] loaded {before} active skills")

holdout = build_holdout(cfg, k=cfg["budgets"]["dev_eval_k"])
lesion_marginal_values(lib, holdout, cfg)
lesion_marginal_add(lib, holdout, cfg)

# show the 2-axis picture before acting
decisions = decide_library(lib, LATE, cfg)
from collections import Counter
print("verdicts:", dict(Counter(decisions.values())))
for s in sorted(lib.active(), key=lambda s: s.reuse.marginal_value, reverse=True)[:8]:
    print(f"  {s.id:42.42} {s.kind:12} mv={s.reuse.marginal_value:+.2%} "
          f"add={s.reuse.marginal_add:+.2%} reuse={s.reuse.reuse_rate:.2f} -> {decisions[s.id]}")

cur = apply_curation(lib, LATE, cfg)
qd = qd_dedupe(lib, LATE, holdout, cfg)
after = len(lib.active())
print(f"\nprune removed {len(cur['pruned'])}, QD removed {len(qd['pruned'])} "
      f"-> {before} active -> {after} active")
survivors = [(s.kind, s.id) for s in lib.active()]
print("survivors:", survivors)

assert after <= before, "curation must not grow the library"
assert all(s.reuse.marginal_value >= -cfg["curation"]["prune_mv_eps"] for s in lib.active()), \
    "no harmful skill should survive"
print(f"OK §3 end-to-end on {RUN}: {before} -> {after} active; harmful/dead removed, valuable kept.")
