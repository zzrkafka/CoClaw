"""Validate action 4 (transfer probe): a structure-based (cluster-by-cost) construct should be
GENERAL (gap holds under renumber/cost_scale/rotate); a hardcoded-magnitude one should be NARROW."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evolve import materialize_skill
from analysis.probe import _perturb, transfer_probe
from lcar.generator import Family
from solvers.reference import fill_reference

GOOD = (ROOT / "scripts" / "_smoke_judge.py").read_text().split("GOOD_CONSTRUCT = r'''")[1].split("'''")[0]

cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
fam = Family.from_config(cfg["lcar"], seed=1234, name="F_hard")
inst = fam.sample_instance(n=60, seed=1234)
fill_reference(inst)

# NARROW: memorizes THIS instance's optimal tour (hardcoded node indices) -> ~0% on base, but
# under renumber the indices point to the wrong nodes -> should be flagged 'narrow'.
NARROW = "def f(n, x, C, prims):\n    return " + repr(list(inst.ref["lkh_tour"])) + "\n"

# sanity: perturbations preserve/scale the known optimum
print("L_lkh:", inst.ref["L_lkh"],
      "| renumber:", _perturb(inst, "renumber").ref["L_lkh"],
      "| cost_scale:", _perturb(inst, "cost_scale").ref["L_lkh"],
      "| rotate:", _perturb(inst, "rotate_coords").ref["L_lkh"])

good = transfer_probe(materialize_skill({"op": "add", "kind": "construct", "name": "good", "code": GOOD}, 0), inst, cfg)
narrow = transfer_probe(materialize_skill({"op": "add", "kind": "construct", "name": "narrow", "code": NARROW}, 0), inst, cfg)
print("structure-based construct:", good)
print("hardcoded-magnitude construct:", narrow)
print("VERDICT:", "PASS" if good["label"] == "general" and narrow["label"] == "narrow" else
      f"(good={good['label']}, narrow={narrow['label']})")
