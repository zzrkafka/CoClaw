"""Transfer probe -- v2.1 patch action 4. POST-HOC, LABEL-ONLY robustness check on ACCEPTED
skills. Perturb the instance (renumber nodes / rescale costs / rotate coords), re-run the skill in
its minimal composition, and see if its gap HOLDS:
  - breaks under `renumber`    => it memorised node indices,
  - breaks under `cost_scale`  => it memorised magnitude thresholds,
  - breaks under `rotate_coords` => it leaned on the (misleading) coords x instead of C.
-> label 'general' | 'narrow'. frac_general over rounds = the SECOND axis of compound-vs-collapse:
a compounding library accumulates GENERAL skills; a collapsing one accumulates FRAGILE memorised
ones (fragility is itself a reportable collapse mode, cf. Library-Learning-Doesn't 2410.20274).

GUARDRAIL (patch 4): the probe is POST-HOC and produces ONLY a label/metric. It MUST NOT enter the
accept gate or the reflection feedback -- otherwise it injects cross-scale design and pollutes the
measurement of whether generalization EMERGES. (renumber/cost_scale/rotate are isomorphic or
optimum-preserving, so L_lkh is known without re-running LKH -> the probe is cheap.)

  python -m analysis.probe --run hard_breadth3 --config configs/default_hard.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

PERTURBATIONS = ("renumber", "cost_scale", "rotate_coords")
_TOL = 0.05   # a perturbed gap may exceed the base gap by at most this, else the skill is 'narrow'


def _perturb(inst, kind):
    """Return a perturbed instance whose OPTIMUM length is known (isomorphic / scaled), so gap is
    comparable to the original without re-running LKH."""
    n = inst.n
    C = np.asarray(inst.C)
    x = np.asarray(inst.x)
    L = inst.ref["L_lkh"]
    if kind == "renumber":                       # relabel nodes: isomorphic -> same L_lkh
        pi = np.random.default_rng(99).permutation(n)
        return SimpleNamespace(n=n, x=x[pi], C=C[np.ix_(pi, pi)], ref={"L_lkh": L})
    if kind == "cost_scale":                     # scale costs: optimum tour unchanged, L scales
        s = 3
        return SimpleNamespace(n=n, x=x, C=C * s, ref={"L_lkh": L * s})
    if kind == "rotate_coords":                  # rotate visible coords only: C (objective) unchanged
        th = 0.7
        r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return SimpleNamespace(n=n, x=x @ r.T, C=C, ref={"L_lkh": L})
    return inst


def _probe_gap(skill, inst, cfg):
    # the skill's OWN output (no NN-min floor, which would mask a broken skill, and whose start
    # node is not renumber-invariant). _run_skill_once runs it directly by kind.
    from agent.evolve import _run_skill_once
    _, g = _run_skill_once(skill, inst, [], cfg)
    return g


def transfer_probe(skill, base_inst, cfg, tol=_TOL, perturbations=PERTURBATIONS) -> dict:
    """Run `skill`'s own output on the original + each perturbation; 'narrow' if a perturbed gap
    blows past base+tol (or it stops producing a valid tour), else 'general'. Only assessed when
    the skill produces a meaningful tour on base (tour-producing kinds)."""
    g0 = _probe_gap(skill, base_inst, cfg)
    per, label = {}, "general"
    for pk in perturbations:
        gp = _probe_gap(skill, _perturb(base_inst, pk), cfg)
        per[pk] = (round(gp, 4) if gp is not None else None)
        if g0 is not None and (gp is None or gp > g0 + tol):
            label = "narrow"
    return {"per_perturbation_gap": per, "label": label,
            "base_gap": (round(g0, 4) if g0 is not None else None)}


def probe_library(lib, base_inst, cfg) -> float | None:
    """Label each active skill not yet labelled (probe once at acceptance), writing the label +
    per-perturbation gaps into skill.meta (-> skill_snapshot). Returns frac_general over active."""
    for s in lib.active():
        if "transfer_label" not in s.meta:
            r = transfer_probe(s, base_inst, cfg)
            s.meta["transfer_label"] = r["label"]
            s.meta["perturbation"] = r["per_perturbation_gap"]
    labels = [s.meta.get("transfer_label") for s in lib.active()]
    return (sum(l == "general" for l in labels) / len(labels)) if labels else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default_hard.yaml"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--root", default="runs")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    from analysis.lesion import build_holdout, load_library_from_run
    lib = load_library_from_run(Path(args.root) / args.run)
    active = lib.active()
    if not active:
        print(f"[probe] {args.run}: no active skills.")
        return 0
    base_inst = build_holdout(cfg, k=args.k)[0]
    print("=" * 74)
    print(f"TRANSFER PROBE   run={args.run}   active_skills={len(active)}")
    print(f"{'skill_id':36} {'kind':11} {'label':8} per-perturbation gap")
    gen = 0
    for s in active:
        r = transfer_probe(s, base_inst, cfg)
        gen += r["label"] == "general"
        print(f"{s.id:36.36} {s.kind:11} {r['label']:8} base={r['base_gap']} {r['per_perturbation_gap']}")
    print("-" * 74)
    print(f"frac_general = {gen}/{len(active)} = {gen/len(active):.0%}   "
          "(high+rising = compounding general skills; low/falling = fragile memorisation = collapse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
