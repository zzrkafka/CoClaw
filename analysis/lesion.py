"""Leave-one-out marginal value -- the node-level credit foundation (spec section 13;
v2.1 patch action 3).

For each active skill, REMOVE it from the library and re-measure mean gap on a HELD-OUT set.
The rise = that skill's marginal value (SkillBrew 2605.29440 counterfactual, but judged by the
EXACT LKH gap, not an LLM). This is what lets us state the science precisely:
  compounding  => skills carry high marginal value AND are reused;
  collapse     => skills are dead weight (marginal ~0 -> prune candidate, section 13).

Two modes (patch action 3):
  pipeline : score via the deterministic curation pipeline (evolve._run_pipeline) -- cheap,
             judge-consistent. Credits the skills the pipeline actually composes
             (construct/strategy/local_search; after the construct split, detect/order/build
             as they chain). This is the routine, per-round mode.
  resolve  : end-to-end CodeAct re-solve (agent.solver.solve_instance) -- faithful but costs
             LLM calls; credits ANY skill the agent invokes. For final figures / spot checks.

The holdout MUST be distinct from the curation dev set: a skill that only helps the instances
it was tuned on has no real marginal value, and that distinction (generalize vs overfit) is
itself part of the compound-vs-collapse story. We sample a fresh seed band for it.

Marginal sign: POSITIVE = removing the skill RAISES gap = the skill is valuable. ~0 = dead
weight. NEGATIVE = the skill is actively harmful (a strong prune signal).

  python -m analysis.lesion --run hard_breadth2 --config configs/default_hard.yaml
  python -m analysis.lesion --run <id> --mode resolve --k 6     # faithful, costs LLM
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# per-subset mean gap on the holdout (the two judge backends)
# ---------------------------------------------------------------------------
def _pipeline_mean_gap(skill_dicts, holdout, eval_budget, apply_local=True):
    """Mean LKH gap of the deterministic pipeline over the holdout, given an active skill set.

    apply_local=True (the deployed pipeline) so marginal value reflects the library AS USED --
    distinct from the step-wise grounded JUDGE, which scores constructs with local-search off."""
    from agent.evolve import _run_pipeline
    from solvers.reference import gap
    gaps = []
    for inst in holdout:
        t = _run_pipeline(inst, list(skill_dicts), eval_budget, apply_local=apply_local)
        if t is not None:
            gaps.append(gap(t, inst))
    return float(mean(gaps)) if gaps else None


def _resolve_mean_gap(active_skills, holdout, llm, cfg):
    """Mean gap of the END-TO-END agent re-solving the holdout with exactly `active_skills`."""
    import copy

    from agent.solver import solve_instance
    from skills.library import SkillLibrary
    lib = SkillLibrary()
    for s in active_skills:
        lib.add(copy.deepcopy(s))
    gaps = []
    for inst in holdout:
        sr = solve_instance(inst, lib, llm, cfg, round_idx=0)
        if sr.best_gap is not None:
            gaps.append(sr.best_gap)
    return float(mean(gaps)) if gaps else None


# ---------------------------------------------------------------------------
# core leave-one-out
# ---------------------------------------------------------------------------
def _compute(lib, holdout, cfg, mode="pipeline", llm=None, apply_local=True):
    """Return (baseline_gap, {skill_id: marginal_value}). Does NOT mutate the library."""
    active = lib.active()
    if not active:
        return None, {}
    eb = cfg["budgets"].get("eval_budget")

    if mode == "resolve":
        if llm is None:
            raise ValueError("resolve mode needs an `llm` client")
        base = _resolve_mean_gap(active, holdout, llm, cfg)

        def measure(subset):
            return _resolve_mean_gap(subset, holdout, llm, cfg)
    else:
        from agent.evolve import _skill_dict
        base = _pipeline_mean_gap([_skill_dict(s) for s in active], holdout, eb, apply_local)

        def measure(subset):
            return _pipeline_mean_gap([_skill_dict(s) for s in subset], holdout, eb, apply_local)

    marginal = {}
    for s in active:
        subset = [t for t in active if t.id != s.id]
        without = measure(subset)
        marginal[s.id] = (without - base) if (without is not None and base is not None) else 0.0
    return base, marginal


def library_value(lib, holdout, cfg, apply_local=True) -> float | None:
    """§6.1 V(t): mean-gap improvement of the WHOLE library over an EMPTY library (the NN floor) on
    the fixed holdout. Rising over rounds = the library compounds; flat/≈0 = collapse (the value is
    coming from per-instance self-correction, not the carried-over library). This is the headline
    compounding curve, the whole-library analogue of per-skill marginal value."""
    from agent.evolve import _skill_dict
    eb = cfg["budgets"].get("eval_budget")
    full = _pipeline_mean_gap([_skill_dict(s) for s in lib.active()], holdout, eb, apply_local)
    empty = _pipeline_mean_gap([], holdout, eb, apply_local)
    if full is None or empty is None:
        return None
    return empty - full                                  # >0 = the library lowers gap below NN floor


def lesion_marginal_add(lib, holdout, cfg, apply_local=True) -> dict[str, float]:
    """§3.4 LEAVE-ONE-IN (SkillBrew counterfactual, exact-gap): for each skill, drop ALL same-kind
    siblings (so a redundant sibling can't mask it), then measure the gap DROP from adding this one
    back among the other kinds. Positive = the skill carries value IN the strategy context even if
    leave-one-out under-credits it (the weak-alone/strong-combo case). Writes skill.reuse.marginal_add
    and returns {skill_id: marginal_add}. Pipeline mode only (cheap, deterministic, judge-consistent)."""
    from agent.evolve import _skill_dict
    active = lib.active()
    if not active:
        return {}
    eb = cfg["budgets"].get("eval_budget")
    out = {}
    for s in active:
        base = [t for t in active if t.kind != s.kind]          # de-redundify: no same-kind siblings
        g_base = _pipeline_mean_gap([_skill_dict(t) for t in base], holdout, eb, apply_local)
        g_with = _pipeline_mean_gap([_skill_dict(t) for t in base + [s]], holdout, eb, apply_local)
        add = (g_base - g_with) if (g_base is not None and g_with is not None) else 0.0
        out[s.id] = add
        lib.skills[s.id].reuse.marginal_add = add
    return out


def lesion_marginal_values(lib, holdout, cfg, mode="pipeline", llm=None,
                           apply_local=True) -> dict[str, float]:
    """Leave-one-out marginal value for every active skill; writes results back into the library
    (`set_marginal_value`) so they flow into section-13 reuse judgment + prune decisions.

    Returns {skill_id: marginal_value}. See module docstring for sign convention and modes."""
    _, marginal = _compute(lib, holdout, cfg, mode=mode, llm=llm, apply_local=apply_local)
    for sid, mv in marginal.items():
        lib.set_marginal_value(sid, mv)
    return marginal


# ---------------------------------------------------------------------------
# holdout + post-hoc library reconstruction (for the CLI)
# ---------------------------------------------------------------------------
def build_holdout(cfg, n=None, k=8, seed_band=7000):
    """Fresh held-out instances, DISTINCT from stream (band 0) / dev (5000) / warm (9000)."""
    from lcar.generator import Family
    from solvers.reference import fill_reference
    n = n or cfg["instances"]["n_main"]
    base = cfg["instances"]["seed"]
    fam = Family.from_config(cfg["lcar"], seed=base, name="F")
    insts = [fam.sample_instance(n=n, seed=base + seed_band + i,
                                 instance_id=f"lesion_{base + seed_band + i}") for i in range(k)]
    from concurrent.futures import ThreadPoolExecutor   # parallel LKH refs (independent, GIL-released)
    with ThreadPoolExecutor(max_workers=max(1, min(len(insts), 8))) as ex:
        list(ex.map(fill_reference, insts))
    return insts


def load_library_from_run(run_dir):
    """Rebuild a (read-only) SkillLibrary from a finished run's persisted skills, so we can
    lesion it post-hoc. Uses skill_snapshot.jsonl (last state per id) + skills/<id>.py code."""
    from skills.library import SkillLibrary
    from skills.schema import Contract, ReuseStats, Skill
    run_dir = Path(run_dir)
    snap = run_dir / "skill_snapshot.jsonl"
    states = {}
    if snap.exists():
        for line in snap.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                states[d["skill_id"]] = d           # last write wins
    lib = SkillLibrary()
    for sid, d in states.items():
        if d.get("status") != "active":
            continue
        code_path = run_dir / "skills" / f"{sid}.py"
        if not code_path.exists():
            continue
        s = Skill(id=sid, kind=d["kind"], version=d.get("version", 1),
                  name=d.get("name", sid), description_nl=d.get("description_nl", ""),
                  contract=Contract(), code=code_path.read_text(),
                  parent_ids=d.get("parent_ids", []), created_round=d.get("created_round", 0),
                  status="active", plan=d.get("plan", ""),
                  origin_family=d.get("origin_family", ""))
        rs = d.get("reuse", {})
        s.reuse = ReuseStats(**{k: v for k, v in rs.items()
                                if k in ReuseStats.__dataclass_fields__})
        lib.skills[sid] = s
        lib._instances_per_skill.setdefault(sid, set())
    return lib


def _report(run_id, base, active, marginal):
    print("=" * 74)
    print(f"LESION marginal value   run={run_id}   active_skills={len(active)}")
    print(f"baseline mean gap (full library) = "
          + (f"{base:.2%}" if base is not None else "NA"))
    print("-" * 74)
    print(f"{'skill_id':38} {'kind':12} {'marginal':>9} {'reuse_rate':>10}")
    for s in sorted(active, key=lambda s: marginal.get(s.id, 0.0), reverse=True):
        print(f"{s.id:38.38} {s.kind:12} {marginal.get(s.id, 0.0):>+8.2%} "
              f"{s.reuse.reuse_rate:>10.2f}")
    print("-" * 74)
    print("marginal = mean-gap RISE when removed. >0 valuable; ~0 dead weight; <0 harmful (prune).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id under runs/ to lesion")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default_hard.yaml"))
    ap.add_argument("--mode", default="pipeline", choices=["pipeline", "resolve"])
    ap.add_argument("--k", type=int, default=8, help="holdout size")
    ap.add_argument("--root", default="runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    lib = load_library_from_run(Path(args.root) / args.run)
    active = lib.active()
    if not active:
        print(f"[lesion] {args.run}: no active skills -> nothing to lesion.")
        return 0
    holdout = build_holdout(cfg, k=args.k)
    llm = None
    if args.mode == "resolve":
        from llm.client import LLMClient
        llm = LLMClient(cfg["model"], budget_usd=5.0)
    base, marginal = _compute(lib, holdout, cfg, mode=args.mode, llm=llm, apply_local=True)
    _report(args.run, base, active, marginal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
