"""Discriminative evaluation set (improvement-plan §4) -- IRT-discrimination / instance-space view.

The skill x instance gap matrix is ALREADY computed inside lesion every round and then thrown away.
We persist it and use it: an instance is DISCRIMINATIVE if removing different skills changes its gap
very differently (it stably separates valuable skills from worthless ones). Crediting skills on the
most discriminative instances makes the signal sharper AND cheaper (fewer dev re-solves).

GUARDRAILS (§4.2), to avoid Goodharting a curated set:
  - a fraction of the dev pool is reserved as a stable ANCHOR (representative, not chosen for
    discrimination) so credit cannot drift onto a self-selected subset;
  - the discriminative subset is used ONLY to rank/credit during the judge -- the FINAL value is
    still settled on the full representative holdout (lesion, unchanged);
  - selection is "pick from already-run instances first" (§4.3 select-then-generate); a parametric
    generator is a later hook.

Cross-seed stability (§4.2 ③) and a generator (§4.3) are documented extension points; the pilot
selects from the existing dev pool, which is the zero-cost first step the plan calls for.
"""
from __future__ import annotations

from statistics import pstdev


def _per_instance_gaps(skill_dicts, instances, eb, apply_local=True):
    from agent.evolve import _run_pipeline
    from solvers.reference import gap
    out = []
    for inst in instances:
        t = _run_pipeline(inst, list(skill_dicts), eb, apply_local=apply_local)
        out.append(gap(t, inst) if t is not None else None)
    return out


def gap_matrix(lib, instances, cfg, apply_local=True) -> dict:
    """The skill x instance gap matrix (leave-one-out): base[i] = full-library gap on instance i;
    deltas[sid][i] = gap rise on instance i when sid is removed (its per-instance marginal value)."""
    from agent.evolve import _skill_dict
    active = lib.active()
    eb = cfg["budgets"].get("eval_budget")
    base = _per_instance_gaps([_skill_dict(s) for s in active], instances, eb, apply_local)
    deltas = {}
    for s in active:
        subset = [_skill_dict(t) for t in active if t.id != s.id]
        woi = _per_instance_gaps(subset, instances, eb, apply_local)
        deltas[s.id] = [(w - b) if (w is not None and b is not None) else None
                        for w, b in zip(woi, base)]
    ids = [getattr(inst, "id", str(i)) for i, inst in enumerate(instances)]
    return {"base": base, "deltas": deltas, "instance_ids": ids}


def discrimination_scores(gm: dict) -> list[float]:
    """Per-instance discrimination = spread (pop. stdev) of the leave-one-out deltas ACROSS skills.
    High = this instance separates skills' marginal contributions = good for credit; low = every
    skill matters the same here (or none do) = uninformative for ranking."""
    deltas, n = gm["deltas"], len(gm["base"])
    scores = []
    for i in range(n):
        col = [deltas[sid][i] for sid in deltas if deltas[sid][i] is not None]
        scores.append(float(pstdev(col)) if len(col) >= 2 else 0.0)
    return scores


def select_discriminative(instances, scores, k, anchor_frac=0.34):
    """Pick up to k instances: the most discriminative ones PLUS a reserved anchor fraction (stable,
    lowest-index = representative) as the anti-Goodhart guardrail. Returns instances in original
    order. When k >= len(instances) it returns all (nothing to select)."""
    n = len(instances)
    k = min(k, n)
    if k >= n:
        return list(instances)
    n_anchor = max(1, round(k * anchor_frac))
    by_disc = sorted(range(n), key=lambda i: scores[i], reverse=True)
    disc = by_disc[: k - n_anchor]
    anchors = [i for i in range(n) if i not in set(disc)][:n_anchor]   # stable representative anchor
    chosen = sorted(set(disc) | set(anchors))
    return [instances[i] for i in chosen]


def discriminative_dev(lib, dev_pool, cfg, apply_local=True):
    """Convenience: compute the gap matrix + scores over dev_pool and return
    (selected_dev, gm, scores). selected_dev is the dev_eval_k-sized credit subset (+ anchor)."""
    k = cfg["budgets"]["dev_eval_k"]
    anchor = cfg.get("discriminative", {}).get("anchor_frac", 0.34)
    gm = gap_matrix(lib, dev_pool, cfg, apply_local=apply_local)
    scores = discrimination_scores(gm)
    return select_discriminative(dev_pool, scores, k, anchor_frac=anchor), gm, scores
