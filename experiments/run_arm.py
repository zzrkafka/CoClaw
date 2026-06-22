"""Run one arm over an instance stream (spec sections 8/9/12/14).

  warm-start (optional): show the agent teacher (LKH) solutions vs a naive toolkit tour ->
                         reflect_induce -> seed initial skills -> set_seed.
  stream: for each instance -> CodeAct solve -> record -> append episode; every m -> evolve.

The per-instance gap over rounds is the compounding curve (H2): grounded+memory should
trend DOWN; no-memory / self-judge should not.

Example (short plumbing run):
  python -m experiments.run_arm --arm grounded_induce_mem --stream-len 6 --max-steps 6 \
      --warm-k 2 --budget 3
"""
from __future__ import annotations

import os
# Single-threaded BLAS BEFORE numpy import: forked sandbox children each call KMeans/BLAS; under
# a run's memory pressure multi-threaded OpenBLAS fails/hangs in C (SIGALRM can't interrupt ->
# kernel hard-timeout). 60x60 problems don't need BLAS threads. Children inherit this via fork.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evolve import (episode_from_solve, episode_from_teacher, evolve,   # noqa: E402
                          format_curation_feedback, induce_skill)
from agent.featurize import featurize                                       # noqa: E402
from agent.growth import fill_holes, slot_fill_ratio                        # noqa: E402
from agent.seeds import seed_library                                        # noqa: E402
from analysis.curation import apply_curation                                # noqa: E402
from analysis.discriminative import discriminative_dev                      # noqa: E402
from analysis.lesion import (build_holdout, lesion_marginal_add,           # noqa: E402
                            lesion_marginal_values, library_value)
from analysis.probe import probe_library                                    # noqa: E402
from analysis.qd import qd_dedupe                                           # noqa: E402
from agent.solver import solve_instance                                     # noqa: E402
from experiments.arms import get_arm                                        # noqa: E402
from experiments.records import RunRecorder                                 # noqa: E402
from lcar.generator import Family                                           # noqa: E402
from lcar.oracle import toolkit_baseline                                    # noqa: E402
from llm.client import LLMClient                                            # noqa: E402
from skills.library import SkillLibrary                                     # noqa: E402
from solvers.reference import fill_reference, gap                           # noqa: E402


def _instances(fam, n, seeds, prefix):
    return [fam.sample_instance(n=n, seed=s, instance_id=f"{prefix}_{s}") for s in seeds]


def run_arm(arm_name, cfg, *, stream_len, n, budget_usd, warm_k, run_id, verbose=True):
    arm = get_arm(arm_name)
    fam = Family.from_config(cfg["lcar"], seed=cfg["instances"]["seed"], name="F")
    base = cfg["instances"]["seed"]
    dev_k = cfg["budgets"]["dev_eval_k"]

    disc_cfg = cfg.get("discriminative", {})            # §4 discriminative eval set
    pool_k = max(dev_k, disc_cfg.get("pool_k", dev_k)) if disc_cfg.get("enabled") else dev_k
    stream = _instances(fam, n, [base + i for i in range(stream_len)], "stream")
    dev = _instances(fam, n, [base + 5000 + i for i in range(pool_k)], "dev")   # dev POOL (§4 selects)
    warm = _instances(fam, n, [base + 9000 + i for i in range(warm_k)], "warm") if warm_k else []

    t0 = time.time()
    # LKH references for gap (the ruler). LKH is compute-bound (~20s/ref on these adversarial
    # matrices) but each ref is INDEPENDENT and solve_atsp shells out to the LKH binary (releasing
    # the GIL), so a thread pool gives true parallelism. Same LKH params/seed -> identical L_lkh,
    # so this does NOT change gaps (A/B with hard_split preserved); it only fills them faster.
    from concurrent.futures import ThreadPoolExecutor
    _refs = stream + dev + warm
    _workers = min(len(_refs), (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max(1, _workers)) as _ex:
        list(_ex.map(fill_reference, _refs))
    # held-out set for lesion marginal-value (distinct seed band 7000); only the memory arm
    # accumulates a library worth lesioning each round.
    holdout = build_holdout(cfg, n=n, k=dev_k) if arm.memory == "on" else []
    if verbose:
        print(f"[setup] {len(stream)} stream + {len(dev)} dev + {len(warm)} warm "
              f"+ {len(holdout)} holdout instances, LKH refs in {time.time()-t0:.1f}s")

    rec = RunRecorder(run_id)
    rec.emit("run_meta", {"run_id": run_id, "arm": arm.__dict__, "family": "F",
                          "family_params": cfg["lcar"], "model": cfg["model"],
                          "seed": base, "stream_len": stream_len, "n": n,
                          "start_ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    for inst in stream:
        rec.emit("instance", {"run_id": run_id, "instance_id": inst.id, "family": "F",
                              "n": n, "seed": inst.seed, "L_lkh": inst.ref["L_lkh"],
                              "toolkit_gap": gap(toolkit_baseline(inst), inst)})

    lib = SkillLibrary(store_dir=str(rec.dir / "skills"))
    seed_library(lib, cfg, round_idx=-1)       # §2.1 thin seed strategy (no-op unless enabled)
    llm = LLMClient(cfg["model"], budget_usd=budget_usd)
    lessons_path = rec.dir / "lessons.jsonl"   # deposited experience (rules + regressions)

    # warm-start: induce initial skills from teacher (LKH) vs toolkit contrast
    feedback = ""  # how the previous round's proposals scored (rejection feedback)
    if warm:
        eps = [episode_from_teacher(i, i.ref["lkh_tour"], toolkit_baseline(i), featurize(i))
               for i in warm]
        evo = evolve(eps, lib, llm, cfg, arm, round_idx=-1, dev_set=dev[:dev_k], baseline_cache={},
                     curation_feedback=feedback, lessons_path=lessons_path)
        feedback = format_curation_feedback(evo["ops"])
        rec.emit("evolve", {"run_id": run_id, "arm": arm.name,
                            "after_instance_id": "warmstart", **evo})
        # §2.2 directed hole-filling: author exactly the operators the seed strategy still lacks
        holes_filled = fill_holes(lib, dev, cfg, llm, lessons_path, -1, induce=induce_skill)
        if verbose and holes_filled:
            acc = sum(1 for h in holes_filled if h["accepted"])
            print(f"[fill-holes] authored {acc}/{len(holes_filled)} hole operators "
                  f"-> {len(lib.active())} skills")
        if holdout and lib.active():
            lesion_marginal_values(lib, holdout, cfg)   # populate marginal_value before snapshot
            if cfg.get("curation", {}).get("leave_one_in", False):
                lesion_marginal_add(lib, holdout, cfg)  # §3.4 leave-one-in combination credit
        apply_curation(lib, -1, cfg)                    # §3.3 prune dead/harmful (no-op unless enabled)
        qd_dedupe(lib, -1, holdout, cfg)                # §3.5 dedupe redundant variants (no-op unless enabled)
        probe_library(lib, holdout[0] if holdout else dev[0], cfg)   # transfer-probe labels (post-hoc)
        rec.skill_snapshot(lib)
        if verbose:
            acc = sum(1 for o in evo["ops"] if o["skill_id"])
            print(f"[warm-start] proposed {len(evo['ops'])} ops, accepted {acc}; "
                  f"library now {len(lib.active())} skills")
    lib.set_seed()

    # stream
    episodes, gaps = [], []
    m = cfg["budgets"]["evolve_every_m"]
    judge_dev = dev[:dev_k]          # §4 the dev subset the grounded judge credits on (default: first k)
    last_select = -10 ** 9
    for i, inst in enumerate(stream):
        sr = solve_instance(inst, lib, llm, cfg, round_idx=i)
        rec.solve_record(arm.name, sr, i, frozen=False)
        feats = featurize(inst)
        ep = episode_from_solve(inst, sr, feats)
        if ep:
            episodes.append(ep)
        gaps.append(sr.best_gap if sr.best_gap is not None else float("nan"))
        if verbose:
            g = f"{sr.best_gap:.2%}" if sr.best_gap is not None else "NA"
            print(f"[{arm.name}] inst {i}: gap={g:>7}  evals={sr.total_evals:>9}  "
                  f"skills={len(lib.active())}  ${llm.usage.usd(llm.model):.3f}")

        if (i + 1) % m == 0 and episodes:
            # §4 (re)select the discriminative dev subset, amortized over reselect_every rounds
            if (disc_cfg.get("enabled") and len(lib.active()) >= 2
                    and (i - last_select) >= disc_cfg.get("reselect_every", 5)):
                judge_dev, gm, scores = discriminative_dev(lib, dev, cfg)
                last_select = i
                if disc_cfg.get("persist_gap_matrix"):
                    rec.emit("discrimination", {"run_id": run_id, "arm": arm.name, "round_idx": i,
                                                "instance_ids": gm["instance_ids"], "scores": scores,
                                                "selected_ids": [getattr(x, "id", "?") for x in judge_dev]})
            evo = evolve(episodes, lib, llm, cfg, arm, round_idx=i, dev_set=judge_dev, baseline_cache={},
                         curation_feedback=feedback, lessons_path=lessons_path)
            feedback = format_curation_feedback(evo["ops"])
            rec.emit("evolve", {"run_id": run_id, "arm": arm.name,
                                "after_instance_id": inst.id, **evo})
            if holdout and lib.active():
                lesion_marginal_values(lib, holdout, cfg)   # populate marginal_value before snapshot
                if cfg.get("curation", {}).get("leave_one_in", False):
                    lesion_marginal_add(lib, holdout, cfg)  # §3.4 leave-one-in combination credit
            curation = apply_curation(lib, i, cfg)          # §3.3 prune dead/harmful before snapshot
            curation["qd"] = qd_dedupe(lib, i, holdout, cfg)   # §3.5 dedupe redundant variants
            frac_general = probe_library(lib, holdout[0] if holdout else dev[0], cfg)
            rec.skill_snapshot(lib)
            valid = [g for g in gaps if g == g]
            rec.emit("metric", {"run_id": run_id, "arm": arm.name, "round_idx": i,
                                "mean_gap": float(np.mean(valid)) if valid else None,
                                "instance_gap": gaps[-1], "n_active_skills": len(lib.active()),
                                "frac_general": frac_general, "curation": curation,
                                "slot_fill_ratio": slot_fill_ratio(lib, cfg),   # §2.5 bootstrap signal
                                "library_value": (library_value(lib, holdout, cfg)
                                                  if (holdout and lib.active()) else None),  # §6.1 V(t)
                                "cost": {"usd": llm.usage.usd(llm.model),
                                         "prompt_tokens": llm.usage.prompt_tokens,
                                         "completion_tokens": llm.usage.completion_tokens}})

    valid = [g for g in gaps if g == g]
    if verbose:
        print(f"\n[{arm.name}] DONE. per-instance gaps: "
              f"{['NA' if g != g else f'{g:.1%}' for g in gaps]}")
        if valid:
            half = max(1, len(valid) // 2)
            print(f"  first-half mean gap {np.mean(valid[:half]):.2%} -> "
                  f"last-half {np.mean(valid[half:]):.2%}  (down = compounding)")
        print(f"  library: {len(lib.active())} active skills | "
              f"spend ~${llm.usage.usd(llm.model):.3f} ({llm.usage.calls} calls) | "
              f"records in {rec.dir}")
    return {"gaps": gaps, "n_skills": len(lib.active()), "usd": llm.usage.usd(llm.model)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="grounded_induce_mem")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--stream-len", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-budget", type=int, default=None, help="cap cost-lookups per solve")
    ap.add_argument("--dev-eval-k", type=int, default=None)
    ap.add_argument("--warm-k", type=int, default=2)
    ap.add_argument("--budget", type=float, default=5.0, help="USD spend cap")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=None, help="override instances.seed (multi-seed runs)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["instances"]["seed"] = args.seed
    if args.max_steps:
        cfg["budgets"]["max_action_steps"] = args.max_steps
    if args.eval_budget:
        cfg["budgets"]["eval_budget"] = args.eval_budget
    if args.dev_eval_k:
        cfg["budgets"]["dev_eval_k"] = args.dev_eval_k
    stream_len = args.stream_len or cfg["instances"]["stream_len"]
    n = args.n or cfg["instances"]["n_main"]
    run_id = args.run_id or f"{args.arm}_{time.strftime('%Y%m%d_%H%M%S')}"

    run_arm(args.arm, cfg, stream_len=stream_len, n=n, budget_usd=args.budget,
            warm_k=args.warm_k, run_id=run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
