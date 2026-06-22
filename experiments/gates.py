"""Phase 0 validity gates (spec section 4.3) -- run BEFORE any LLM spend.

  Gate 1: mean toolkit-baseline gap > gates.toolkit_gap_min   -> the trap is hard enough
  Gate 2: mean structure-oracle gap < gates.oracle_gap_max    -> the structure is exploitable
  Gate 3: on small n, L_lkh == L_exact (Held-Karp)            -> LKH is a trustworthy optimum

Gates 1 & 2 bracket the discovery space the skill library must traverse: from the high
toolkit line down toward the low oracle line. If a gate fails, retune beta/pi/noise
(spec 4.3) before running experiments -- do not proceed.

Run:  python -m experiments.gates --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcar.generator import Family                                  # noqa: E402
from lcar.oracle import toolkit_baseline, structure_oracle         # noqa: E402
from solvers.reference import gap, fill_reference                  # noqa: E402
from solvers.exact import held_karp_atsp, MAX_N                    # noqa: E402
from sandbox.primitives import tour_length, is_valid_tour          # noqa: E402


def _load(config_path: str) -> dict:
    return yaml.safe_load(Path(config_path).read_text())


def gates_1_2(fam: Family, n: int, n_instances: int, base_seed: int,
              toolkit_min: float, oracle_max: float) -> dict:
    tool_gaps, orac_gaps = [], []
    print(f"\n[gates 1&2] n={n}, {n_instances} instances of family {fam.name}")
    for idx in range(n_instances):
        inst = fam.sample_instance(n=n, seed=base_seed + idx)
        fill_reference(inst)                                  # sets inst.ref['L_lkh'] via LKH
        t_tour = toolkit_baseline(inst)
        o_tour = structure_oracle(inst, fam)
        assert is_valid_tour(t_tour, n) and is_valid_tour(o_tour, n)
        tg, og = gap(t_tour, inst), gap(o_tour, inst)
        tool_gaps.append(tg)
        orac_gaps.append(og)
        if idx < 5:
            print(f"  inst {idx:2d}: L_lkh={inst.ref['L_lkh']:>9d}  "
                  f"toolkit_gap={tg:7.2%}  oracle_gap={og:7.2%}")
    mt, mo = float(np.mean(tool_gaps)), float(np.mean(orac_gaps))
    g1 = mt > toolkit_min
    g2 = mo < oracle_max
    print(f"  -> mean toolkit_gap = {mt:7.2%}  (gate 1 needs > {toolkit_min:.0%})  "
          f"[{'PASS' if g1 else 'FAIL'}]")
    print(f"  -> mean oracle_gap  = {mo:7.2%}  (gate 2 needs < {oracle_max:.0%})  "
          f"[{'PASS' if g2 else 'FAIL'}]")
    return dict(gate1=g1, gate2=g2, mean_toolkit_gap=mt, mean_oracle_gap=mo,
                toolkit_gaps=tool_gaps, oracle_gaps=orac_gaps)


def gate_3(fam: Family, n_small: int, n_instances: int, base_seed: int) -> dict:
    print(f"\n[gate 3] LKH vs exact (Held-Karp) at n={n_small}, {n_instances} instances")
    if n_small > MAX_N:
        print(f"  n_small={n_small} > Held-Karp cap {MAX_N}; reduce --gate3-n or use Gurobi.")
        return dict(gate3=False, matches=0, total=n_instances)
    matches = 0
    for idx in range(n_instances):
        inst = fam.sample_instance(n=n_small, seed=10_000 + base_seed + idx)
        L_lkh = fill_reference(inst)
        L_exact, ex_tour = held_karp_atsp(inst.C)
        assert is_valid_tour(ex_tour, n_small)
        assert tour_length(ex_tour, inst.C) == L_exact
        ok = (L_lkh == L_exact)
        matches += int(ok)
        print(f"  inst {idx:2d}: L_lkh={L_lkh:>8d}  L_exact={L_exact:>8d}  "
              f"{'==' if ok else '!=  (LKH suboptimal!)'}")
    g3 = (matches == n_instances)
    print(f"  -> {matches}/{n_instances} match  [{'PASS' if g3 else 'FAIL'}]")
    return dict(gate3=g3, matches=matches, total=n_instances)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--n-instances", type=int, default=20, help="instances for gates 1&2")
    ap.add_argument("--gate3-n", type=int, default=12, help="small n for the exact check")
    ap.add_argument("--gate3-instances", type=int, default=5)
    args = ap.parse_args()

    cfg = _load(args.config)
    fam = Family.from_config(cfg["lcar"], seed=cfg["instances"]["seed"], name="F")
    n = cfg["instances"]["n_main"]
    base_seed = cfg["instances"]["seed"]
    gcfg = cfg["gates"]

    print("=" * 64)
    print("CoClaw Phase 0 -- validity gates")
    print("=" * 64)

    r12 = gates_1_2(fam, n, args.n_instances, base_seed,
                    gcfg["toolkit_gap_min"], gcfg["oracle_gap_max"])
    r3 = gate_3(fam, args.gate3_n, args.gate3_instances, base_seed)

    all_pass = r12["gate1"] and r12["gate2"] and r3["gate3"]
    print("\n" + "=" * 64)
    print(f"  gate 1 (trap hard)        : {'PASS' if r12['gate1'] else 'FAIL'}")
    print(f"  gate 2 (structure usable) : {'PASS' if r12['gate2'] else 'FAIL'}")
    print(f"  gate 3 (LKH = optimal)    : {'PASS' if r3['gate3'] else 'FAIL'}")
    print(f"  PHASE 0: {'GO -- testbed is sound' if all_pass else 'NO-GO -- retune beta/pi/noise (spec 4.3)'}")
    print("=" * 64)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
