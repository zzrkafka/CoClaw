"""Evolution / curation (spec section 9) -- the library's compound-vs-collapse engine.

Every `evolve_every_m` instances: reflect on recent solves -> propose typed skill ops ->
CURATE (accept/reject) -> apply + version. The two science arms differ ONLY in the judge:
  - grounded  : measure the candidate's REAL effect on dev-instance gap (exact objective).
  - selfjudge : the LLM rates the candidate from code+description, never running it.

DESIGN NOTE (grounded judge cost) -- the spec says "end-to-end re-solve dev with
library+candidate". Re-running the full LLM agent per candidate is ~dev_eval_k * 2 * steps
model calls each (infeasible at stream_len=150). We instead score the candidate by running
it (and the library) through a FIXED deterministic pipeline on the dev instances and
measuring the exact LKH gap. This keeps the judge grounded by the true objective and cheap,
and preserves the grounded-vs-self-judge contrast (H3). Swappable if the budget allows the
full re-solve.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from agent.lessons import (append_lesson, lessons_block, lint_skill, load_lessons,
                           retrieve_lessons, vote_lessons)
from llm.prompt_loader import render_messages
from sandbox.executor import Limits, run_in_sandbox
from sandbox.primitives import tour_length
from skills.schema import KIND, SIGNATURES, Contract, Skill
from solvers.reference import gap


# ---------------------------------------------------------------------------
# arm + episode views
# ---------------------------------------------------------------------------
@dataclass
class Arm:
    name: str
    curation: str   # "grounded" | "selfjudge"
    reflect: str    # "induce" | "scoreonly"
    memory: str     # "on" | "off"


@dataclass
class Episode:
    """What reflection sees for one instance: a better and a worse solution + C structure."""
    instance_id: str
    n: int
    C: np.ndarray
    better_tour: list
    better_len: int
    worse_tour: list | None
    worse_len: int | None
    c_clusters: int = 0
    c_silhouette: float = 0.0


def episode_from_solve(inst, sr, feats: dict | None = None) -> Episode | None:
    """Better = the agent's best tour; worse = the worst valid tour it tried."""
    if sr.best_tour is None:
        return None
    worse_tour, worse_len = None, None
    for t in sr.trajectory:
        if t["tour"] is not None:
            L = tour_length(t["tour"], inst.C)
            if worse_len is None or L > worse_len:
                worse_tour, worse_len = t["tour"], L
    return Episode(inst.id, inst.n, np.asarray(inst.C), sr.best_tour, sr.best_length,
                   worse_tour, worse_len,
                   c_clusters=(feats or {}).get("c_n_clusters", 0),
                   c_silhouette=(feats or {}).get("c_silhouette", 0.0))


def episode_from_teacher(inst, teacher_tour, agent_tour, feats: dict | None = None) -> Episode:
    """Warm-start: better = the teacher's (LKH) tour, worse = the agent's naive attempt."""
    return Episode(inst.id, inst.n, np.asarray(inst.C), list(teacher_tour),
                   tour_length(teacher_tour, inst.C),
                   list(agent_tour) if agent_tour is not None else None,
                   tour_length(agent_tour, inst.C) if agent_tour is not None else None,
                   c_clusters=(feats or {}).get("c_n_clusters", 0),
                   c_silhouette=(feats or {}).get("c_silhouette", 0.0))


# ---------------------------------------------------------------------------
# prompt-block formatting
# ---------------------------------------------------------------------------
def _edges(tour, C, cap: int = 60) -> str:
    n = len(tour)
    parts = []
    for i in range(min(n, cap)):
        a, b = tour[i], tour[(i + 1) % n]
        parts.append(f"{a}->{b}({int(C[a, b])})")
    return " ".join(parts) + (" ..." if n > cap else "")


def _trajectories_block(episodes: list[Episode], max_eps: int = 3) -> str:
    out = []
    for ep in episodes[-max_eps:]:
        out.append(
            f"# instance {ep.instance_id} (n={ep.n}; C shows ~{ep.c_clusters} cost-clusters, "
            f"silhouette {ep.c_silhouette:.2f})")
        out.append(f"BETTER (len {ep.better_len}): {ep.better_tour}")
        out.append(f"  better edges: {_edges(ep.better_tour, ep.C)}")
        if ep.worse_tour is not None:
            out.append(f"WORSE (len {ep.worse_len}): {ep.worse_tour}")
            out.append(f"  worse edges: {_edges(ep.worse_tour, ep.C)}")
    return "\n".join(out)


def _scores_block(episodes: list[Episode], max_eps: int = 8) -> str:
    out = []
    for ep in episodes[-max_eps:]:
        out.append(f"instance {ep.instance_id}: best_len={ep.better_len} "
                   f"(worst_tried={ep.worse_len})")
    return "\n".join(out)


def library_summary(lib) -> str:
    rows = []
    for s in lib.active():
        rows.append(f"{s.id} : {s.kind} : {s.description_nl} : "
                    f"reuse_rate={s.reuse.reuse_rate:.2f} : marginal={s.reuse.marginal_value:.3f}")
    return "\n".join(rows) if rows else "(empty library)"


# ---------------------------------------------------------------------------
# proposal parsing + materialization
# ---------------------------------------------------------------------------
def parse_ops(raw: str) -> tuple[str, list[dict]]:
    """Parse the strict-JSON reflection output -> (analysis, ops). Robust to code fences."""
    txt = raw.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", txt, re.DOTALL)
    if m:
        txt = m.group(1).strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        b0, b1 = txt.find("{"), txt.rfind("}")
        if b0 < 0 or b1 < 0:
            return "", []
        try:
            data = json.loads(txt[b0:b1 + 1])
        except json.JSONDecodeError:
            return "", []
    ops = [o for o in data.get("ops", []) if isinstance(o, dict) and o.get("op") in
           {"add", "modify", "merge", "prune"} and o.get("kind") in KIND | {None}]
    return data.get("analysis", ""), ops


# ---------------------------------------------------------------------------
# focused per-skill authoring (harnessed induction): the planner (reflect) decides WHICH
# typed skills to add; each one's code is then written by a SEPARATE focused call with a
# clean, kind-specific context. Measured: a focused construct prompt yields near-optimal code
# (1-2% gap), while the overloaded all-in-one reflect call yields buggy/eval-hungry code.
# ---------------------------------------------------------------------------
_AUTHOR_HINTS = {
    "construct":   # BUILD-only after the atomization (struct + order are handed to you)
        "BUILD ONLY. struct is a dict with struct['labels'] (per-node cluster id); `order` is a list "
        "of cluster ids in visit order (may be None -> fall back to sorted(set(labels))). Do NOT "
        "cluster and do NOT solve the cluster order here (that is the detect/order skills' job). "
        "Visit clusters in `order`, nearest-neighbour within each (cheapest entry node from the "
        "previous cluster), reading costs as C[u, v]; verify prims.is_valid_tour and append any "
        "missing node before returning.",
    "order":
        "Return the cluster ids (from struct['labels']) in the CHEAPEST directed visit order. Build "
        "the K x K inter-cluster cost IC[a][b] = cheapest DIRECTED edge from a node of cluster a to a "
        "node of cluster b (C[u, v]); brute-force ALL orderings of the FEW clusters (K is small) for "
        "the cheapest directed cycle; return the list of cluster ids in that order. Never permute nodes.",
    "strategy":
        "Orchestrate other skills via skills[\"<id>\"](...). Plan: cluster -> solve the small "
        "cluster-TSP for the cheap visit order -> construct in that order -> a SHORT bounded "
        "improvement. Do NOT bake an unbounded local search inside.",
    "local_search":
        "Return the SAME node set, reordered. C is ASYMMETRIC: accept a move ONLY if "
        "prims.tour_length recomputed on the FULL new tour decreases (never a symmetric 2-opt "
        "delta), and BOUND the passes to a small fixed number.",
    "repair":
        "Reinsert the removed nodes into `partial` cheaply; return a valid permutation of range(n).",
    "destroy":
        "Remove k nodes (e.g. on the most expensive edges); return (partial_tour, removed_nodes).",
    "diagnose":
        "Return a dict of cheap signals about where the tour is weak (most expensive edges / "
        "cluster boundaries). No heavy computation.",
    "debug":
        "RECOVER a stuck/failed `tour` into a better one (same node set). Re-diagnose what drives "
        "cost, fix the dominant layer, BOUND passes, and accept a change ONLY if prims.tour_length "
        "on the FULL tour decreases (never a symmetric delta on asymmetric C). Return a valid "
        "permutation of range(n).",
}


def _parse_code_json(raw: str) -> str:
    """Extract the skill code from an author response. Robust to: strict JSON, fenced JSON,
    a ```python code block, or a bare `def f(...)` span (the model occasionally drops JSON)."""
    txt = raw.strip()
    inner = txt
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", txt, re.DOTALL)
    if m:
        inner = m.group(1).strip()
    for cand in (inner, txt):
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and d.get("code"):
                return d["code"]
        except json.JSONDecodeError:
            a, b = cand.find("{"), cand.rfind("}")
            if a >= 0 and b > a:
                try:
                    d = json.loads(cand[a:b + 1])
                    if isinstance(d, dict) and d.get("code"):
                        return d["code"]
                except json.JSONDecodeError:
                    pass
    pm = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)   # raw python block
    if pm and "def f(" in pm.group(1):
        return pm.group(1).strip()
    fm = re.search(r"(def f\(.*)", raw, re.DOTALL)              # bare def f(...) to end
    if fm:
        return fm.group(1).strip()
    return ""


def author_skill(kind: str, intent: str, rules_block: str, llm, tries: int = 3,
                 feedback: str = "", prev_code: str = "", persona: str = "") -> str:
    """Write ONE skill's code in a focused, clean context (kind-specific signature + rules).

    persona injects a diverse generation angle (breadth). feedback/prev_code drive a MINIMAL-patch
    revision (depth) inside the agentic loop. Retries on an unparseable / empty response (a parse
    miss must NOT fall back to the planner's overloaded code).
    """
    msgs = render_messages("author_skill", kind=kind,
                           signature=SIGNATURES.get(kind, SIGNATURES["construct"]),
                           rules_block=rules_block, intent=intent or "(lower future gap)",
                           hint=_AUTHOR_HINTS.get(kind, ""), persona=persona or "",
                           feedback=feedback or "", prev_code=prev_code or "")
    for _ in range(tries):
        code = _parse_code_json(llm.complete(msgs, json_mode=True, max_tokens=4000))
        if code and "def f(" in code:
            return code
    return ""


def materialize_skill(op: dict, round_idx: int) -> Skill:
    c = op.get("contract") or {}
    return Skill(
        id="", kind=op.get("kind", "construct"), version=1,
        name=op.get("name", op.get("kind", "skill")),
        description_nl=op.get("description_nl", ""),
        contract=Contract(inputs=c.get("inputs", []), requires=c.get("requires", []),
                          preserves=c.get("preserves", []), produces=c.get("produces", [])),
        code=op.get("code", ""), parent_ids=[], created_round=round_idx, status="active",
        plan=op.get("plan", ""),                            # §1.3 declarative guidance (strategy)
        origin_family=op.get("origin_family", ""),          # §6 provenance (transfer attribution)
        meta={"claimed_improvement": op.get("claimed_improvement", "")},
        tags=[],
    )


# ---------------------------------------------------------------------------
# grounded judge: fixed deterministic pipeline over library skills (no LLM)
# ---------------------------------------------------------------------------
_PIPELINE_CODE = r'''
def _nn_tour():
    visited = [False] * n
    t = [0]; visited[0] = True
    for _ in range(n - 1):
        last = t[-1]; best = -1; bestc = None
        for j in range(n):
            if not visited[j]:
                cc = C[last, j]
                if bestc is None or cc < bestc:
                    bestc = cc; best = j
        t.append(best); visited[best] = True
    return t

def _apply_local(t):
    # apply each local_search to the running best; KEEP ONLY if it strictly improves
    best = list(t); best_len = prims.tour_length(best, C)
    for sid, k in skill_kinds.items():
        if k != "local_search":
            continue
        try:
            nt = skills[sid](list(best), C, prims)
            if prims.is_valid_tour(nt, n):
                nl = prims.tour_length(nt, C)
                if nl < best_len:
                    best, best_len = list(nt), nl
        except Exception:
            pass
    return best

# NN is ALWAYS a candidate, so adding any skill can only help (min over a superset).
# `apply_local` (prepended by _run_pipeline) toggles the local-search stage: with it OFF,
# constructs are judged on RAW build quality (step-wise / partial-credit fix).
# ATOMIZED construct (v2.1 patch action 1): compose detect(diagnose) -> order -> build(construct).
# A default struct and order=None are ALWAYS included, so detect's marginal reflects its EDGE over
# the free prim and order's marginal reflects the cheap visit order it adds -- this is what gives
# lesion per-atom resolution instead of one monolith eating all the credit.
_detects = [s for s, k in skill_kinds.items() if k == "diagnose"]
_orders = [s for s, k in skill_kinds.items() if k == "order"]
_builds = [s for s, k in skill_kinds.items() if k == "construct"]

def _default_struct():
    try:
        _lab = prims.cluster_by_cost(C)
        return {"labels": [int(v) for v in _lab], "k": len(set(int(v) for v in _lab))}
    except Exception:
        return {"labels": [0] * n, "k": 1}

def _build_call(_b, _st, _ord):
    try:
        return prims.sanitize_tour(skills[_b](n, x, C, prims, _st, _ord), n, C)
    except TypeError:
        return prims.sanitize_tour(skills[_b](n, x, C, prims), n, C)   # old-signature construct

_al = apply_local
_nn = _nn_tour()
_cands = [_apply_local(_nn) if _al else _nn]

for _sid in [s for s, k in skill_kinds.items() if k == "strategy"]:
    try:
        _t = prims.sanitize_tour(skills[_sid](n, x, C, skills, prims), n, C)
        _cands.append(_apply_local(_t) if _al else _t)
    except Exception:
        pass

_structs = []
for _sid in _detects[:2]:
    try:
        _d = skills[_sid](list(_nn), x, C, prims)
        if isinstance(_d, dict) and "labels" in _d:
            _structs.append(_d)
    except Exception:
        pass
_structs.append(_default_struct())   # always-present fallback -> detect credited on its EDGE only

for _st in _structs:
    _ords = [None]                    # always-present arbitrary order -> order credited on its EDGE
    for _osid in _orders[:2]:
        try:
            _o = skills[_osid](_st, C, prims)
            if _o is not None:
                _ords.append(list(_o))
        except Exception:
            pass
    for _ord in _ords:
        for _b in _builds[:2]:
            try:
                _t = _build_call(_b, _st, _ord)
                _cands.append(_apply_local(_t) if _al else _t)
            except Exception:
                pass

FINAL_TOUR = min(_cands, key=lambda tt: prims.tour_length(tt, C))
'''

# apply ONE candidate local-search to a fixed base tour, UNCONDITIONALLY -- so a regression is
# detected (not hidden by keep-if-improving). `_base` is prepended as a literal by the caller.
_APPLY_LS_CODE = r'''
_after = list(_base)
try:
    _after = prims.sanitize_tour(skills["__cand__"](list(_base), C, prims), n, C)
except Exception:
    pass
FINAL_TOUR = _after
'''


def _skill_dict(s: Skill) -> dict:
    return {"id": s.id, "kind": s.kind, "code": s.code}


def _run_pipeline(inst, skill_dicts: list[dict], eval_budget=None, apply_local=True) -> list | None:
    ctx = {"n": inst.n, "x": np.asarray(inst.x), "C": np.asarray(inst.C),
           "skills": skill_dicts, "eval_budget": eval_budget}
    code = f"apply_local = {bool(apply_local)}\n" + _PIPELINE_CODE
    res = run_in_sandbox(code, ctx, Limits(cpu_seconds=30, wall_timeout=10.0))
    return res.tour if res.feasible else None


def eval_on_dev(cand: Skill, dev_set, lib, cfg, baseline_cache: dict | None = None) -> dict:
    """Grounded judge for a construct/strategy candidate, STEP-WISE.

    Constructs/strategies are scored with the local-search stage OFF (apply_local=False) so the
    candidate is credited on its RAW construct quality -- a sibling (or future) local-search can
    neither hide a bad constructor nor inflate it. delta = mean(cand_gap - lib_gap) (negative =
    improvement); regressed = the candidate made the raw construct stage WORSE.

    Runs under the same eval_budget the agent faces, so an eval-hungry skill that would blow the
    budget fails here too. (baseline_cache is ignored now that base depends on the stage mode.)
    """
    k = cfg["budgets"]["dev_eval_k"]
    eb = cfg["budgets"].get("eval_budget")
    devs = dev_set[:k]
    lib_dicts = [_skill_dict(s) for s in lib.active()]
    cand_dicts = lib_dicts + [_skill_dict(cand)]
    deltas, cand_gaps, base_gaps, feasible = [], [], [], True
    for inst in devs:
        bt = _run_pipeline(inst, lib_dicts, eb, apply_local=False)
        ct = _run_pipeline(inst, cand_dicts, eb, apply_local=False)
        if ct is None or bt is None:
            feasible = False
            continue
        bg, cg = gap(bt, inst), gap(ct, inst)
        deltas.append(cg - bg)
        cand_gaps.append(cg)
        base_gaps.append(bg)
    if not deltas:
        return {"delta": 0.0, "feasible": False, "cand_gap": None, "base_gap": None,
                "regressed": False}
    md = float(np.mean(deltas))
    return {"delta": md, "feasible": feasible, "cand_gap": float(np.mean(cand_gaps)),
            "base_gap": float(np.mean(base_gaps)), "regressed": md > 1e-9}


def eval_local_search(cand: Skill, dev_set, lib, cfg) -> dict:
    """Grounded judge for a local_search candidate, STEP-WISE.

    Apply the candidate UNCONDITIONALLY to the construct-stage best tour and measure the real
    gap change. Unlike the keep-if-improving pipeline (which hides harm), this DETECTS a
    regression -- e.g. the asymmetric-2-opt bug that drags a 1.4% tour to ~37%. Accept only if
    it actually improves; a regression is reported (-> rejection feedback + a deposited lesson).
    """
    k = cfg["budgets"]["dev_eval_k"]
    eb = cfg["budgets"].get("eval_budget")
    devs = dev_set[:k]
    lib_dicts = [_skill_dict(s) for s in lib.active()]
    cand_dict = [{"id": "__cand__", "kind": "local_search", "code": cand.code}]
    deltas, cand_gaps, base_gaps, regressed = [], [], [], False
    for inst in devs:
        base_tour = _run_pipeline(inst, lib_dicts, eb, apply_local=False)
        if base_tour is None:
            continue
        bg = gap(base_tour, inst)
        code = f"_base = {[int(v) for v in base_tour]}\n" + _APPLY_LS_CODE
        ctx = {"n": inst.n, "x": np.asarray(inst.x), "C": np.asarray(inst.C),
               "skills": cand_dict, "eval_budget": eb}
        res = run_in_sandbox(code, ctx, Limits(cpu_seconds=30, wall_timeout=10.0))
        at = res.tour if res.feasible else None
        if at is None:
            continue
        ag = gap(at, inst)
        deltas.append(ag - bg)
        cand_gaps.append(ag)
        base_gaps.append(bg)
        if ag > bg + 1e-9:
            regressed = True
    if not deltas:
        return {"delta": 0.0, "feasible": False, "cand_gap": None, "base_gap": None,
                "regressed": False}
    return {"delta": float(np.mean(deltas)), "feasible": True,
            "cand_gap": float(np.mean(cand_gaps)), "base_gap": float(np.mean(base_gaps)),
            "regressed": regressed}


def eval_debug(cand: Skill, dev_set, lib, cfg) -> dict:
    """Grounded judge for a DEBUG (recovery) skill (§1.4), STEP-WISE.

    Apply the candidate to a STUCK cost-NN tour (the kind of failed/degraded state a debug skill
    exists to rescue) and measure the REPAIR gap delta -- gap(before) vs gap(after). This is the
    cleanest credit signal: a debug skill earns its place only by actually pulling a bad tour
    toward optimal. Same call shape as local_search, but the base is deliberately degraded (not the
    library's best), so it credits recovery rather than incremental polish. Regression is detected
    (it must not make the stuck tour even worse)."""
    k = cfg["budgets"]["dev_eval_k"]
    eb = cfg["budgets"].get("eval_budget")
    devs = dev_set[:k]
    cand_dict = [{"id": "__cand__", "kind": "debug", "code": cand.code}]
    deltas, cand_gaps, base_gaps, regressed = [], [], [], False
    for inst in devs:
        base_tour = _nn_tour_py(inst.C, inst.n)        # the stuck/degraded base to recover from
        bg = gap(base_tour, inst)
        code = f"_base = {[int(v) for v in base_tour]}\n" + _APPLY_LS_CODE
        ctx = {"n": inst.n, "x": np.asarray(inst.x), "C": np.asarray(inst.C),
               "skills": cand_dict, "eval_budget": eb}
        res = run_in_sandbox(code, ctx, Limits(cpu_seconds=30, wall_timeout=10.0))
        at = res.tour if res.feasible else None
        if at is None:
            continue
        ag = gap(at, inst)
        deltas.append(ag - bg)
        cand_gaps.append(ag)
        base_gaps.append(bg)
        if ag > bg + 1e-9:
            regressed = True
    if not deltas:
        return {"delta": 0.0, "feasible": False, "cand_gap": None, "base_gap": None,
                "regressed": False}
    return {"delta": float(np.mean(deltas)), "feasible": True,
            "cand_gap": float(np.mean(cand_gaps)), "base_gap": float(np.mean(base_gaps)),
            "regressed": regressed}


# ---------------------------------------------------------------------------
# self-judge (the collapse-prone arm)
# ---------------------------------------------------------------------------
def self_judge(cand: Skill, op: dict, lib, llm) -> dict:
    msgs = render_messages("curate_selfjudge", op_kind=op.get("op"), kind=cand.kind,
                           description_nl=cand.description_nl, code=cand.code,
                           library_summary=library_summary(lib))
    raw = llm.complete(msgs, json_mode=True)
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        r = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        r = {}
    return {"value": float(r.get("value", 0.0)),
            "predicted_gap_delta": float(r.get("predicted_gap_delta", 0.0)),
            "accept": bool(r.get("accept", False)), "reason": r.get("reason", "")}


# ---------------------------------------------------------------------------
# apply + orchestrate
# ---------------------------------------------------------------------------
def apply_op(lib, op: dict, cand: Skill | None, round_idx: int) -> str | None:
    kind = op.get("op")
    if kind == "add" and cand is not None:
        return lib.add(cand, round_idx)
    if kind == "modify" and cand is not None and op.get("target_id"):
        return lib.modify(op["target_id"], cand, round_idx)
    if kind == "merge" and cand is not None:
        parents = op.get("parent_ids") or ([op["target_id"]] if op.get("target_id") else [])
        return lib.merge(parents, cand, round_idx)
    if kind == "prune" and op.get("target_id"):
        lib.prune(op["target_id"], round_idx)
        return op["target_id"]
    return None


def format_curation_feedback(op_records: list[dict]) -> str:
    """Turn the previous round's judge outcomes into reflection feedback (rejection learning)."""
    if not op_records:
        return "(none yet)"
    lines = []
    for o in op_records:
        j = o.get("judge", {})
        status = "ACCEPTED" if j.get("accepted") else "REJECTED"
        head = f"- {o.get('op')}/{o.get('kind')}"
        if j.get("lint_hard"):
            lines.append(f"{head}: BLOCKED by supervisor ({'; '.join(j['lint_hard'])}) -> {status}.")
            continue
        extra = ""
        if j.get("regressed"):
            extra += " -- it made the tour WORSE (regression)"
        if j.get("lint_warn"):
            extra += " [supervisor: " + "; ".join(j["lint_warn"])[:200] + "]"
        if "dev_gap_delta" in j:  # grounded: real measured numbers
            cg, bg = j.get("cand_gap"), j.get("base_gap")
            if j.get("probation"):
                lines.append(f"{head}: admitted on validity (probation){extra} -> {status}.")
            elif cg is not None:
                lines.append(f"{head}: reached ~{cg:.1%} dev gap (baseline ~{bg:.1%}, optimal 0%); "
                             f"delta {j.get('dev_gap_delta'):+.1%}{extra} -> {status}.")
            else:
                lines.append(f"{head}: produced no valid/measurable improvement{extra} -> {status}.")
        else:  # self-judge: only the model's own opinion
            lines.append(f"{head}: self-rated {j.get('llm_rating')}, predicted "
                         f"{j.get('predicted_gap_delta')}{extra} -> {status}.")
    return "\n".join(lines)


_PROBATION_KINDS = {"repair", "destroy", "diagnose", "order"}   # not standalone-scorable by gap;
# order/diagnose are admitted on validity, then credited/pruned by lesion (analysis/lesion.py).


# ---------------------------------------------------------------------------
# agentic closed-loop induction: author -> grounded run -> self-correct; self-distilled lessons
# ---------------------------------------------------------------------------
def _revise_feedback(kind: str, res: dict) -> str:
    """Grounded facts handed back to the author for self-correction (no hand-fed bug recipe)."""
    cg, bg = res.get("cand_gap"), res.get("base_gap")
    if not res.get("feasible") or cg is None:
        return ("It failed to run or returned nothing measurable. Make it robust: read costs as "
                "C[i, j], BOUND every loop, return a full permutation of range(n).")
    if res.get("regressed"):
        return (f"It made the tour WORSE (reached ~{cg:.0%}; baseline ~{bg:.0%}). On this ASYMMETRIC "
                "matrix accept a move ONLY if prims.tour_length on the FULL tour decreases; bound passes.")
    return (f"It reached ~{cg:.0%} dev gap, but baseline is ~{bg:.0%} and optimal ~0% -- it is NOT "
            "beating the baseline, so it is not exploiting the structure that dominates cost. "
            "Reconsider what actually drives cost here and exploit it.")


def distill_lesson(kind: str, code: str, res: dict, llm, round_idx: int) -> dict | None:
    """The agent reads its own failed attempt + the grounded outcome and writes a GENERAL rule to
    avoid this class of mistake next time. This REPLACES hand-written rules (Pillar 4)."""
    cg, bg = res.get("cand_gap"), res.get("base_gap")
    msgs = render_messages("distill_lesson", kind=kind, code=code,
                           cand_gap=(f"{cg:.0%}" if isinstance(cg, float) else "n/a"),
                           base_gap=(f"{bg:.0%}" if isinstance(bg, float) else "n/a"),
                           regressed=bool(res.get("regressed")), feasible=bool(res.get("feasible")))
    raw = llm.complete(msgs, json_mode=True, max_tokens=600)
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return None
    rule = (d.get("rule") or "").strip()
    if not rule:
        return None
    sha = hashlib.sha1(rule.encode()).hexdigest()[:8]
    return {"id": f"auto-{kind}-{sha}", "rule": rule, "cause": d.get("cause", ""),
            "source": "agent-distilled", "round": round_idx,
            "condition": {"kind": kind}, "votes": 0}


# ---- execution-grounded evidence (ExecReport) + separate debugger critic (Direction 1) ----
def _nn_tour_py(C, n):
    C = np.asarray(C)
    visited = [False] * n; t = [0]; visited[0] = True
    for _ in range(n - 1):
        last = t[-1]; row = C[last]; best, bc = -1, None
        for j in range(n):
            if not visited[j] and (bc is None or row[j] < bc):
                bc, best = row[j], j
        t.append(best); visited[best] = True
    return t


def _run_skill_once(cand, inst, lib_dicts, cfg):
    """Run ONE skill directly by kind (not the min-pipeline), to capture its OWN behavior +
    hang traceback. Returns (Result, gap-or-None)."""
    kind = cand.kind
    eb = cfg["budgets"].get("eval_budget")
    base = _nn_tour_py(inst.C, inst.n)
    if kind == "construct":
        tail, sk = "FINAL_TOUR = f(n, x, C, prims)", []
    elif kind == "strategy":
        tail, sk = "FINAL_TOUR = f(n, x, C, skills, prims)", lib_dicts
    elif kind == "local_search":
        tail, sk = f"FINAL_TOUR = f({base}, C, prims)", []
    elif kind == "debug":                                  # §1.4 recover the stuck NN base
        tail, sk = f"FINAL_TOUR = f({base}, C, prims)", []
    elif kind == "repair":
        tail, sk = f"FINAL_TOUR = f({base[: max(2, inst.n // 2)]}, n, C, prims)", []
    elif kind == "diagnose":
        tail, sk = f"_d = f({base}, x, C, prims); print('OUT', type(_d).__name__); FINAL_TOUR = {base}", []
    elif kind == "order":
        tail, sk = ("_lab = prims.cluster_by_cost(C); "
                    "_st = {'labels': [int(v) for v in _lab], 'k': len(set(int(v) for v in _lab))}; "
                    f"_o = f(_st, C, prims); print('OUT', type(_o).__name__); FINAL_TOUR = {base}"), []
    elif kind == "destroy":
        tail, sk = f"_r = f({base}, C, prims, 3); print('OUT', type(_r).__name__); FINAL_TOUR = {base}", []
    else:
        tail, sk = "FINAL_TOUR = list(range(n))", []
    ctx = {"n": inst.n, "x": np.asarray(inst.x), "C": np.asarray(inst.C), "skills": sk, "eval_budget": eb}
    res = run_in_sandbox(cand.code + "\n" + tail + "\n", ctx, Limits(cpu_seconds=20, wall_timeout=6.0))
    return res, (gap(res.tour, inst) if res.tour is not None else None)


def static_flags(kind, code):
    """Cheap static smells for the debugger (esp. the factorial blow-up behind our hangs)."""
    flags = []
    m = re.search(r"permutations\(\s*range\(\s*([A-Za-z_0-9]+)", code)
    if m:
        flags.append(f"uses permutations(range({m.group(1)})) -- factorial blow-up if "
                     f"{m.group(1)} is the node count, not the small cluster count")
    flags += lint_skill(kind, code)[1]
    return flags


def exec_report(cand, dev_set, lib, cfg) -> dict:
    """Run the candidate on 1-2 dev probes and assemble grounded evidence for the debugger:
    status (hang/error/invalid/noop/regress/ok) + the hang traceback + per-dev gaps + static smells."""
    devs = dev_set[:2]
    lib_dicts = [_skill_dict(s) for s in lib.active()]
    per, hang, error, bases = [], "", "", []
    for inst in devs:
        res, g = _run_skill_once(cand, inst, lib_dicts, cfg)
        per.append({"inst": inst.id, "gap": g})
        if res.hang_trace and not hang:
            hang = res.hang_trace
        if res.error and not error:
            error = res.error
        bases.append(gap(_nn_tour_py(inst.C, inst.n), inst))
    valid = [p["gap"] for p in per if p["gap"] is not None]
    base = float(np.mean(bases)) if bases else None
    thr = cfg["budgets"]["accept_threshold"]
    if not valid:
        status, cg = ("hang" if hang else ("error" if error else "invalid")), None
    else:
        cg = float(np.mean(valid))
        status = ("ok" if base is not None and cg < base - thr
                  else "regress" if base is not None and cg > base + 0.02 else "noop")
    return {"status": status, "hang_trace": hang, "error": error, "per_dev": per,
            "base_gap": base, "cand_gap": cg, "static_flags": static_flags(cand.kind, cand.code)}


def debugger(code, report, llm) -> dict:
    """Separate critic role (fresh context): read code + grounded ExecReport, output ONE root
    cause + ONE fix directive. Does NOT rewrite -- the author (Learner) applies the directive."""
    parts = []
    for p in report["per_dev"]:
        g = p["gap"]
        parts.append(f"{p['inst']}: " + (f"{g:.0%}" if g is not None else "FAILED"))
    msgs = render_messages("debugger", code=code, status=report["status"],
                           hang_trace=(report["hang_trace"] or "(none)")[:1200],
                           error=report["error"] or "(none)", per_dev="; ".join(parts),
                           base_gap=(f"{report['base_gap']:.0%}" if report["base_gap"] is not None else "n/a"),
                           static_flags="; ".join(report["static_flags"]) or "(none)")
    raw = llm.complete(msgs, json_mode=True, max_tokens=500)
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {"root_cause": (d.get("root_cause") or "").strip(),
            "fix_directive": (d.get("fix_directive") or "").strip()}


def deterministic_fix(report) -> str:
    return ("Your skill returned an invalid node set. Return a permutation of ALL range(n) "
            "(each node once); verify prims.is_valid_tour(tour, n) and append missing nodes first.")


_MECHANICAL = {"invalid"}


# NOTE: combination credit ("weak alone, strong in combination") is handled by the patch's
# analysis/lesion.py (leave-one-out marginal value, SkillBrew with LKH gap), computed post-hoc
# each round in run_arm -- NOT by an accept-time counterfactual here (that would muddy the clean
# accept gate: build -> gap, detect/order -> probation, then lesion credits/prunes).

# diverse generation angles (breadth). Cold-start seeds; archive-contrastive selection comes
# with the QD library (Stage 2b). Each is one persona injected into the author prompt.
_GEN_PERSONAS = {
    "construct": [   # BUILD-only: assemble from the given struct + order
        "visit clusters in the given `order`, nearest-neighbour within each cluster",
        "lay each cluster's nodes contiguously following `order`, cheapest entry node from the "
        "previous cluster's exit",
    ],
    "order": [
        "build the K x K inter-cluster cost and brute-force the cheapest DIRECTED order over the "
        "FEW clusters",
        "greedy nearest-cluster walk on the inter-cluster cost matrix, starting from cluster 0",
    ],
    "strategy": [
        "compose: call a construct skill for a cluster-ordered tour, then ONE bounded improvement",
        "ALNS-lite: construct, then a few BOUNDED destroy-repair rounds on the most expensive edges",
    ],
    "local_search": [
        "or-opt: move a single node to its cheapest position; bounded passes; accept only if the "
        "FULL prims.tour_length drops",
        "swap the order of two whole clusters; accept only if the FULL prims.tour_length drops",
        "asymmetric 2-opt: reverse a segment, accept ONLY if prims.tour_length on the full tour "
        "decreases; few bounded passes",
    ],
    "repair": ["cheapest-insertion of the removed nodes"],
    "destroy": ["remove the k nodes sitting on the most expensive edges"],
    "diagnose": [   # =DETECT: return the cluster structure for order/build to consume
        "return {'labels': prims.cluster_by_cost(C) as a list of per-node cluster ids, 'k': K}"],
    "debug": [   # =RECOVER: rescue a stuck/failed tour; accept only if the FULL tour cost drops
        "re-diagnose the structure of the given tour, rebuild the dominant layer (e.g. the visit "
        "order), and keep the result only if prims.tour_length on the full tour decreases",
        "targeted repair: find the few most expensive edges and reinsert their endpoints at their "
        "cheapest positions; bounded passes; accept only if the FULL tour cost drops"],
}


def _personas(kind, n):
    ps = _GEN_PERSONAS.get(kind, [])
    return ps[:n] if ps else [""]


def _grounded(cand, kind, dev_set, lib, cfg):
    if kind == "local_search":
        return eval_local_search(cand, dev_set, lib, cfg)
    if kind == "debug":                                   # §1.4 recovery skill: repair-delta credit
        return eval_debug(cand, dev_set, lib, cfg)
    if kind in _PROBATION_KINDS:
        return {"delta": 0.0, "feasible": True, "cand_gap": None, "base_gap": None,
                "regressed": False, "probation": True}
    return eval_on_dev(cand, dev_set, lib, cfg)


def induce_skill(op, lib, dev_set, cfg, llm, rules_block, round_idx, lessons_path,
                 n_breadth=3, max_revise=1):
    """Item 3 wrapper: AutoGuide-retrieve ONLY the lessons relevant to THIS skill kind, run the
    breadth/depth core with them, then ExpeL-vote those lessons by whether the op succeeded."""
    kind = op.get("kind")
    retrieved = retrieve_lessons(load_lessons(lessons_path), kind)
    rb = lessons_block(retrieved) if retrieved else rules_block
    code, res, accepted = _induce_core(op, lib, dev_set, cfg, llm, rb, round_idx, lessons_path,
                                       n_breadth, max_revise)
    vote_lessons(lessons_path, [l.get("id") for l in retrieved], 1 if accepted else -1)
    return code, res, accepted


def _induce_core(op, lib, dev_set, cfg, llm, rules_block, round_idx, lessons_path,
                 n_breadth=3, max_revise=1):
    """BREADTH-then-DEPTH induction for ONE add/modify op (grounded arms).

    BREADTH: author n_breadth DIVERSE personas, grounded-eval each, pick the best by exact gap --
    independent shots beat sequential repair (which whack-a-moles: each fix adds a new bug).
    DEPTH: if the best isn't accepted, a SEPARATE debugger reads its execution evidence and the
    author applies ONE MINIMAL patch (<= max_revise). On non-convergence the agent self-distills a
    clean lesson from the debugger's grounded root cause. Returns (code, res, accepted)."""
    kind, intent = op.get("kind"), op.get("description_nl", "")
    thr = cfg["budgets"]["accept_threshold"]

    def ok(r):
        return r["feasible"] and (r.get("probation") or r["delta"] <= -thr)

    # --- breadth: diverse candidates, grounded-selected ---
    graded = []
    for persona in _personas(kind, n_breadth):
        code = author_skill(kind, intent, rules_block, llm, persona=persona)
        if not code or "def f(" not in code:
            continue
        cand = materialize_skill(dict(op, code=code), round_idx)
        res = _grounded(cand, kind, dev_set, lib, cfg)
        graded.append((code, cand, res))
        if ok(res):
            return code, res, True
    if not graded:
        return "", {"feasible": False, "delta": 0.0, "cand_gap": None, "base_gap": None,
                    "regressed": False}, False
    graded.sort(key=lambda g: g[2]["cand_gap"] if g[2].get("cand_gap") is not None else 9.9)
    code, cand, res = graded[0]

    # --- depth: debugger-guided MINIMAL patch on the best candidate ---
    last_root = ""
    for _ in range(max_revise):
        report = exec_report(cand, dev_set, lib, cfg)
        if report["status"] in _MECHANICAL:
            directive = deterministic_fix(report)
        else:
            d = debugger(code, report, llm)
            directive = d["fix_directive"] or _revise_feedback(kind, res)
            last_root = d["root_cause"] or last_root
        new = author_skill(kind, intent, rules_block, llm, feedback=directive, prev_code=code)
        if not new or "def f(" not in new:
            continue
        ncand = materialize_skill(dict(op, code=new), round_idx)
        nres = _grounded(ncand, kind, dev_set, lib, cfg)
        if nres.get("cand_gap") is not None and (res.get("cand_gap") is None
                                                 or nres["cand_gap"] < res["cand_gap"]):
            code, cand, res = new, ncand, nres   # keep the better of old/new
        if ok(res):
            return code, res, True

    lesson = distill_lesson(kind, code, res, llm, round_idx)
    if lesson is not None:
        if last_root:
            lesson["cause"] = last_root
        if lessons_path is not None:
            append_lesson(lessons_path, lesson)
    return code, res, False


def evolve(episodes: list[Episode], lib, llm, cfg, arm: Arm, round_idx: int,
           dev_set, baseline_cache: dict | None = None, curation_feedback: str = "",
           lessons_path=None) -> dict:
    """One evolution step (step-wise grounded + partial credit + supervisor + lessons).

    Pipeline per proposed op:
      1) SUPERVISOR lint -- hard errors reject before any dev eval; warnings annotate feedback.
      2) GROUNDED judge by kind: construct/strategy scored on raw construct quality
         (local-search OFF); local_search applied unconditionally to detect regression;
         repair/destroy/diagnose admitted on validity (probation -- pruned later if unused).
      3) PARTIAL CREDIT: each op judged independently, so a good constructor is accepted even if
         a sibling local-search is rejected.
      4) LESSON DEPOSITION: a regression (or lint-flagged then confirmed) appends a lesson so
         the next round avoids the same low-level mistake.

    curation_feedback: how the PREVIOUS round's proposals scored (rejection learning).
    """
    fam = (cfg.get("problem") or {}).get("family", "F")   # §6 stamp skills with their origin family
    deposited = load_lessons(lessons_path)
    lblock = lessons_block(deposited)
    tmpl = "reflect_induce" if arm.reflect == "induce" else "reflect_scoreonly"
    common = dict(library_summary=library_summary(lib),
                  curation_feedback=curation_feedback or "(none yet)",
                  lessons_block=lblock)
    if arm.reflect == "induce":
        msgs = render_messages(tmpl, trajectories_block=_trajectories_block(episodes), **common)
    else:
        msgs = render_messages(tmpl, scores_block=_scores_block(episodes), **common)
    raw = llm.complete(msgs, json_mode=True, max_tokens=8000)  # skill code can be long
    analysis, ops = parse_ops(raw)
    # Dependency-respecting order: detect -> order -> build/strategy -> local_search, so a build is
    # judged AFTER the order it composes with is already in the library (else build alone cannot beat
    # cost-NN and would be wrongly rejected via the standalone path). prune first.
    _OP_PRI = {"diagnose": 0, "order": 1, "destroy": 2, "repair": 2,
               "construct": 3, "strategy": 4, "local_search": 5, "debug": 6}
    ops = sorted(ops, key=lambda o: -1 if o.get("op") == "prune" else _OP_PRI.get(o.get("kind"), 9))
    thr = cfg["budgets"]["accept_threshold"]

    op_records = []
    for op in ops:
        opname, kind = op.get("op"), op.get("kind")
        judge = {"type": arm.curation}

        if opname == "prune":
            skill_id = apply_op(lib, op, None, round_idx)
            judge["accepted"] = True
            op_records.append({"op": opname, "kind": kind, "skill_id": skill_id, "code_sha": "",
                               "judge": judge,
                               "claimed_improvement": op.get("claimed_improvement", "")})
            continue
        if opname not in ("add", "modify") or kind not in KIND:
            continue

        if arm.curation == "grounded":
            # AGENTIC closed loop: author -> grounded run -> agent revises from the measured
            # outcome (self-correction); on non-convergence the agent distills its own lesson.
            code, res, accept = induce_skill(op, lib, dev_set, cfg, llm, lblock, round_idx,
                                             lessons_path)
            if not code:
                judge.update(accepted=False, author_failed=True)
                op_records.append({"op": opname, "kind": kind, "skill_id": None, "code_sha": "",
                                   "judge": judge,
                                   "claimed_improvement": op.get("claimed_improvement", "")})
                continue
            op = dict(op, code=code, origin_family=fam)
            cand = materialize_skill(op, round_idx)
            _, warn = lint_skill(kind, code)
            if warn:
                judge["lint_warn"] = warn
            judge.update(dev_gap_delta=res["delta"], cand_gap=res["cand_gap"],
                         base_gap=res["base_gap"], feasibility_ok=res["feasible"],
                         regressed=res.get("regressed", False),
                         probation=res.get("probation", False), accepted=accept)
        else:  # self-judge: focused author ONCE, NO grounded loop (it cannot verify a fix)
            code = author_skill(kind, op.get("description_nl", ""), lblock, llm)
            if not code or "def f(" not in code:
                judge.update(accepted=False, author_failed=True)
                op_records.append({"op": opname, "kind": kind, "skill_id": None, "code_sha": "",
                                   "judge": judge,
                                   "claimed_improvement": op.get("claimed_improvement", "")})
                continue
            op = dict(op, code=code, origin_family=fam)
            cand = materialize_skill(op, round_idx)
            _, warn = lint_skill(kind, code)
            if warn:
                judge["lint_warn"] = warn
            r = self_judge(cand, op, lib, llm)
            accept = r["accept"]
            judge.update(llm_rating=r["value"], predicted_gap_delta=r["predicted_gap_delta"],
                         accepted=accept, reason=r["reason"])

        skill_id = apply_op(lib, op, cand, round_idx) if accept else None
        if accept and skill_id is not None:
            cand.meta.update({k: v for k, v in judge.items() if k != "type"})
        code_sha = hashlib.sha1(cand.code.encode()).hexdigest()[:12]
        op_records.append({"op": opname, "kind": kind, "skill_id": skill_id, "code_sha": code_sha,
                           "judge": judge, "claimed_improvement": op.get("claimed_improvement", "")})

    if arm.memory == "off":
        lib.reset_to_seed()

    return {"reflection": analysis, "ops": op_records, "library_version": lib.version,
            "round_idx": round_idx}
