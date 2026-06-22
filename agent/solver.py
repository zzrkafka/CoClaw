"""Per-instance CodeAct solve loop (spec section 8) -- the agent (INV-2).

A multi-step loop: THINK -> one code action (executed in a stateful session) -> observe ->
decide the next action, until the agent sets DONE or the action-step budget runs out. The
solution is the PRODUCT OF THE TRAJECTORY, not one final solve() program.

The agent keeps its running answer in FINAL_TOUR (captured every action, never lost); the
harness tracks the best-by-length across all actions and measures the LKH gap for
reporting (the agent never sees the gap). This loop does NOT modify the library
(evolution happens in agent/evolve.py).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from agent.featurize import featurize
from agent.harness import recovery_note
from agent.playbook import harness_playbook
from llm.prompt_loader import render_messages
from sandbox.executor import CodeActSession, Limits
from sandbox.primitives import tour_length
from skills.schema import SIGNATURES

try:
    from solvers.reference import gap as _gap  # needs inst.ref['L_lkh']
except Exception:  # noqa: BLE001
    _gap = None


@dataclass
class SolveResult:
    instance_id: str
    best_tour: list | None
    best_length: int | None
    best_gap: float | None
    finalized: bool
    n_steps: int
    total_evals: int
    trajectory: list = field(default_factory=list)


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_codeact(text: str) -> tuple[str, str]:
    """Extract (think, code) from the model output. Both are optional/robust."""
    think_m = _THINK_RE.search(text)
    think = think_m.group(1).strip() if think_m else ""
    code_m = _CODE_RE.search(text)
    code = code_m.group(1) if code_m else ""
    return think, code


def _skills_block(skills) -> str:
    if not skills:
        return "(none retrieved yet -- write your own code to probe C and build a tour)"
    return "\n".join(
        f"- {s.id} : {s.kind} : {SIGNATURES.get(s.kind, '?')} : {s.description_nl}"
        for s in skills
    )


def _plans_block(skills) -> str:
    """§1.3: declarative guidance (plan) carried by retrieved skills -- chiefly strategies. This is
    the plan half of plan+scaffold: it STEERS the agent (the scaffold code is what gets grounded).
    Empty unless harness.inject_plans is on, so the baseline solve is byte-for-byte unchanged."""
    rows = [f"- [{s.id} ({s.kind})] {s.plan.strip()}"
            for s in skills if getattr(s, "plan", "").strip()]
    return "\n".join(rows)


def _ctx_of(inst, skills, eval_budget=None) -> dict:
    return {
        "n": inst.n, "x": np.asarray(inst.x), "C": np.asarray(inst.C),
        "skills": [{"id": s.id, "kind": s.kind, "code": s.code} for s in skills],
        "eval_budget": eval_budget,
    }


def _c_stats(inst) -> str:
    """Neutral, faithful stats (spec 3.5): C is the objective and asymmetric; no cluster hint."""
    C = np.asarray(inst.C)
    mask = ~np.eye(inst.n, dtype=bool)
    off = C[mask]
    asym = float(np.abs(C - C.T)[mask].mean())
    return (f"mean={off.mean():.0f}, std={off.std():.0f}, min={int(off.min())}, "
            f"max={int(off.max())}, mean|C-C^T|={asym:.0f} (asymmetric)")


def _x_preview(inst) -> str:
    return str(np.asarray(inst.x)[:3].round(2).tolist())


def solve_instance(inst, lib, llm, cfg, round_idx: int, frozen: bool = False,
                   limits: Limits | None = None) -> SolveResult:
    max_steps = cfg["budgets"]["max_action_steps"]
    eval_budget = cfg["budgets"].get("eval_budget")
    feats = featurize(inst)
    skills = lib.retrieve(feats, k=cfg["budgets"]["retrieve_top_k"])

    inject_plans = cfg.get("harness", {}).get("inject_plans", False)   # §1.3, off by default
    plans_block = _plans_block(skills) if inject_plans else ""
    playbook_block = harness_playbook(cfg)                             # §1.5, off by default
    debug_recovery = cfg.get("harness", {}).get("debug_recovery", False)   # §1.4, off by default
    debug_skills = [s for s in skills if s.kind == "debug"]
    msgs = list(render_messages(
        "codeact_system", n=inst.n, c_stats=_c_stats(inst), x_preview=_x_preview(inst),
        skills_block=_skills_block(skills), plans_block=plans_block,
        playbook_block=playbook_block, max_steps=max_steps, eval_budget=eval_budget))

    sess = CodeActSession(_ctx_of(inst, skills, eval_budget),
                          limits or Limits(cpu_seconds=60, wall_timeout=15.0))
    traj: list[dict] = []
    best_tour, best_len = None, None
    have_ref = (_gap is not None) and ("L_lkh" in inst.ref)
    total_evals, finalized = 0, False

    try:
        for step in range(max_steps):
            out = llm.complete(msgs, temperature=0.3)  # lower temp -> less solve variance
            think, code = parse_codeact(out)

            if not code.strip():
                summary = "[no python code action found; emit exactly one ```python ... ``` block]"
                obs_tour, skills_called, n_evals, exec_ms, obs_error = None, [], total_evals, 0.0, None
            else:
                obs = sess.act(code)
                total_evals = max(total_evals, obs.n_evals)
                obs_tour, skills_called = obs.tour, obs.skills_called
                n_evals, exec_ms, summary, obs_error = obs.n_evals, obs.exec_ms, obs.summary, obs.error
                finalized = obs.finalized

            prev_best = best_len                          # to detect a regression this action
            cur_len = tour_length(obs_tour, inst.C) if obs_tour is not None else None
            cur_gap = _gap(obs_tour, inst) if (obs_tour is not None and have_ref) else None
            if cur_len is not None and (best_len is None or cur_len < best_len):
                best_tour, best_len = obs_tour, cur_len
            for sid in skills_called:
                lib.record_invocation(sid, inst.id, round_idx)

            traj.append(dict(step=step, think=think, code=code,
                             code_sha=hashlib.sha1(code.encode()).hexdigest()[:12] if code else "",
                             observation=summary, tour=obs_tour, gap=cur_gap,
                             skills_called=skills_called, n_evals=n_evals, exec_ms=exec_ms))
            if finalized:
                break

            remaining = max_steps - step - 1
            # RECOVER stage (agent/harness.py) -- explicit, pluggable seam for the §1.4 debug skill.
            debug_hint = ""
            if debug_recovery and debug_skills:
                failed = bool(obs_error) or obs_tour is None
                regressed = (cur_len is not None and prev_best is not None and cur_len > prev_best)
                if failed or regressed:
                    did = debug_skills[0].id
                    debug_hint = (f"If your tour is stuck or got worse, call "
                                  f"skills['{did}'](FINAL_TOUR, C, prims) to recover, then keep its "
                                  "result ONLY if prims.tour_length drops.")
            note = recovery_note(remaining=remaining, obs_error=obs_error, best_len=best_len,
                                 debug_hint=debug_hint)
            msgs += [("assistant", out), ("user", (summary or "(no output)") + note)]
    finally:
        sess.close()

    best_gap = _gap(best_tour, inst) if (best_tour is not None and have_ref) else None
    return SolveResult(instance_id=inst.id, best_tour=best_tour, best_length=best_len,
                       best_gap=best_gap, finalized=finalized, n_steps=len(traj),
                       total_evals=total_evals, trajectory=traj)
