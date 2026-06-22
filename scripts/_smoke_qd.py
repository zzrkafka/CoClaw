"""Smoke: §3.5 QD niche dedupe -- behaviorally identical variants collapse to one; a variant that
behaves differently (its own niche) survives. Needs LKH (per-instance gap profile)."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.lesion import build_holdout                # noqa: E402
from analysis.qd import qd_dedupe                        # noqa: E402
from skills.library import SkillLibrary                  # noqa: E402
from skills.schema import Contract, Skill                # noqa: E402

NOOP = "def f(tour, C, prims):\n    return list(tour)\n"             # identity
NOOP2 = "def f(tour, C, prims):\n    return list(tour)  # dup\n"     # same behavior, different code
REV = "def f(tour, C, prims):\n    return list(tour)[::-1]\n"        # different tour -> own niche


def mk(sid, code):
    return Skill(id=sid, kind="local_search", version=1, name=sid, description_nl=sid,
                 contract=Contract(), code=code, parent_ids=[], created_round=0, status="active")


cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
cfg.setdefault("curation", {})["qd_niche"] = True
holdout = build_holdout(cfg, k=cfg["budgets"]["dev_eval_k"])

lib = SkillLibrary()
for sid, code in [("noop1", NOOP), ("noop2", NOOP2), ("rev", REV)]:
    lib.add(mk(sid, code))

res = qd_dedupe(lib, 0, holdout, cfg)
active = {s.id for s in lib.active()}
print("pruned:", res["pruned"], "| active after:", sorted(active), "| niches:", res["niches"])

assert len(res["pruned"]) == 1 and res["pruned"][0] in {"noop1", "noop2"}, \
    f"exactly one of the identical noops should be pruned, got {res['pruned']}"
assert "rev" in active, "the behaviorally-distinct variant must survive its own niche"
assert len(active) == 2, f"3 variants in 2 niches -> 2 survive, got {active}"

# disabled -> no-op
lib2 = SkillLibrary(); lib2.add(mk("a", NOOP)); lib2.add(mk("b", NOOP2))
off = qd_dedupe(lib2, 0, holdout, {"curation": {"qd_niche": False}})
assert off["enabled"] is False and len(lib2.active()) == 2, "must be a no-op when disabled"
print("OK §3.5: identical variants collapse to one niche; distinct variant kept; no-op when off.")
