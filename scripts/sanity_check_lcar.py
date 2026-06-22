"""Numerical sanity check for the LCAR testbed -- needs NO LKH, Gurobi, or LLM key.

Verifies the design premises behind validity gates 1 & 2 (spec section 4.3) using the
structure-oracle tour as a stand-in reference (the true gate uses L_lkh <= L_oracle, so
the toolkit gap shown here is a LOWER BOUND on the real gate-1 number):

  1. shapes / integrality of the generated instance
  2. costs are genuinely ASYMMETRIC
  3. triangle inequality is VIOLATED (so memorized metric heuristics are unfounded)
  4. structure_oracle tour cost << toolkit_baseline tour cost on the TRUE C

Run from the project root:  python scripts/sanity_check_lcar.py
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcar.generator import Family                      # noqa: E402
from lcar.oracle import (                              # noqa: E402
    structure_oracle, toolkit_baseline, euclidean_matrix,
)
from sandbox.primitives import tour_length, is_valid_tour  # noqa: E402


def asymmetry_stats(C):
    diff = np.abs(C - C.T)
    off = ~np.eye(len(C), dtype=bool)
    return diff[off].mean(), diff[off].max()


def triangle_violation_rate(C, samples, rng):
    n = len(C)
    viol = 0
    for _ in range(samples):
        i, j, k = rng.integers(0, n, 3)
        if i == j or j == k or i == k:
            continue
        if C[i, k] > C[i, j] + C[j, k] + 1e-9:
            viol += 1
    return viol / samples


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    fam = Family.from_config(cfg["lcar"], seed=cfg["instances"]["seed"], name="F")
    n = cfg["instances"]["n_main"]
    base_seed = cfg["instances"]["seed"]
    n_instances = 12

    rng = np.random.default_rng(base_seed)
    print(f"=== LCAR sanity check | family F | n={n} | {n_instances} instances ===\n")

    tool_ratios, orac_self = [], []
    asym_means = []
    tri_rates = []

    for idx in range(n_instances):
        inst = fam.sample_instance(n=n, seed=base_seed + idx)
        C = inst.C

        assert inst.x.shape == (n, 2), "x shape"
        assert C.shape == (n, n), "C shape"
        assert C.dtype.kind in "iu", "C must be integer"
        assert (np.diag(C) == 0).all(), "diagonal must be 0"

        am, ax = asymmetry_stats(C)
        asym_means.append(am)
        tri = triangle_violation_rate(C, 4000, rng)
        tri_rates.append(tri)

        t_tour = toolkit_baseline(inst)
        o_tour = structure_oracle(inst, fam)
        assert is_valid_tour(t_tour, n), "toolkit tour invalid"
        assert is_valid_tour(o_tour, n), "oracle tour invalid"

        t_len = tour_length(t_tour, C)
        o_len = tour_length(o_tour, C)
        # proxy gap: how much worse the toolkit is vs the (cheating) oracle reference
        ratio = (t_len - o_len) / o_len
        tool_ratios.append(ratio)
        orac_self.append(o_len)

        if idx < 4:
            print(f" inst {idx}: toolkit_len={t_len:>9d}  oracle_len={o_len:>9d}  "
                  f"toolkit/oracle-1={ratio:6.2%}  asym(mean|max)={am:7.1f}|{ax:7.0f}  "
                  f"tri_viol={tri:5.1%}")

    print()
    print(f" mean asymmetry |C-C^T|       : {np.mean(asym_means):8.1f}   (0 would mean symmetric)")
    print(f" mean triangle-violation rate : {np.mean(tri_rates):8.1%}   (random triples i->k vs i->j->k)")
    print(f" mean toolkit-vs-oracle gap   : {np.mean(tool_ratios):8.2%}   (LOWER BOUND on real gate-1; needs > 25%)")
    print(f"   min / max over instances   : {np.min(tool_ratios):8.2%} / {np.max(tool_ratios):.2%}")
    print()

    ok_asym = np.mean(asym_means) > 1.0
    ok_tri = np.mean(tri_rates) > 0.01
    ok_trap = np.mean(tool_ratios) > 0.25
    print(" PREMISE CHECKS")
    print(f"   [{'PASS' if ok_asym else 'FAIL'}] costs are asymmetric")
    print(f"   [{'PASS' if ok_tri else 'FAIL'}] triangle inequality violated")
    print(f"   [{'PASS' if ok_trap else 'WARN'}] toolkit trap is hard (oracle-proxy > 25%); "
          f"real gate-1 uses L_lkh and will be >= this")
    print()
    print(" NOTE: true gates 1-3 require LKH-3 (and Gurobi for gate 3); this is the")
    print("       LLM-free, solver-free smoke test that the testbed design is sound.")


if __name__ == "__main__":
    main()
