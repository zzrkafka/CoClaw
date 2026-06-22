"""Quality-Diversity niche management for variants (improvement-plan §3.5).

The bloat in hard_split was largely near-duplicate variants of the same kind (many `cluster_by_cost_*`
/ `bounded_2opt_*`). A variant should survive ONLY if it occupies a DISTINCT behavioral niche -- it
wins on instances the others lose. Otherwise it is redundant and should be dropped, keeping the best
representative of its niche.

Behavioral descriptor = the per-holdout-instance gap profile of the skill's OWN output, discretized
into a niche key. Variants of the same kind with the same niche key are behaviorally equivalent ->
keep the lowest-mean-gap one, prune the rest. Variants with different niche keys (they win/lose on
different instances) are complementary -> both kept. This is the lightweight, deterministic core of
the two-layer QD archive (Stage 2b); merging into a single combined variant (which needs authoring)
is left to the LLM merge op -- here we DEDUPE by niche.

Only kinds whose own output IS a tour have a meaningful solo descriptor here (construct / strategy /
local_search / debug). order / diagnose / repair / destroy are credited in-context by lesion +
leave-one-in (§3.1/§3.4), not deduped solo (a future pipeline-context descriptor can extend this).
"""
from __future__ import annotations

from statistics import mean

# kinds whose own output is a full tour (so a per-instance gap profile is meaningful solo)
_TOUR_KINDS = {"construct", "strategy", "local_search", "debug"}
_BIN = 0.01   # gap bucket width for the niche key (1%): finer = more niches survive


def behavior_vector(skill, holdout, cfg) -> list[float | None]:
    """Per-instance gap of the skill's OWN output on the holdout (None where it produced no tour)."""
    from agent.evolve import _run_skill_once
    out = []
    for inst in holdout:
        _, g = _run_skill_once(skill, inst, [], cfg)
        out.append(g)
    return out


def niche_key(vec: list[float | None], bins: float = _BIN) -> tuple:
    """Discretize a gap profile into a niche signature: skills that perform the same (bucketed) on
    every instance share a niche; a skill that wins where others lose lands in a different niche."""
    return tuple(None if g is None else round(g / bins) for g in vec)


def _mean_gap(vec):
    vals = [g for g in vec if g is not None]
    return mean(vals) if vals else float("inf")


def qd_dedupe(lib, round_idx: int, holdout, cfg) -> dict:
    """§3.5 hook: within each tour-producing kind, group active skills by behavioral niche; keep the
    best (lowest mean gap) per niche, prune the redundant rest. No-op unless curation.qd_niche is on.
    Call AFTER the marginal-value prune so it only dedupes survivors."""
    if not (cfg or {}).get("curation", {}).get("qd_niche", False):
        return {"enabled": False, "pruned": [], "niches": {}}
    groups: dict[tuple, list] = {}
    for s in lib.active():
        if s.kind not in _TOUR_KINDS:
            continue
        vec = behavior_vector(s, holdout, cfg)
        if all(g is None for g in vec):
            continue                                    # never produced a tour -> not deduped here
        groups.setdefault((s.kind, niche_key(vec)), []).append((s.id, _mean_gap(vec)))
    pruned = []
    for members in groups.values():
        if len(members) <= 1:
            continue
        members.sort(key=lambda m: m[1])                # best (lowest gap) first
        for sid, _ in members[1:]:                      # the rest of this niche are redundant
            lib.prune(sid, round_idx)
            pruned.append(sid)
    return {"enabled": True, "pruned": pruned,
            "niches": {f"{k[0]}:{hash(k[1]) & 0xffff:04x}": len(v) for k, v in groups.items()},
            "n_active_after": len(lib.active())}
