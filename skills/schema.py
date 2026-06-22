"""Typed skill data classes (spec section 7.1).

A skill is a versioned, typed Python function with a contract, an NL description (for
retrieval/reflection), and -- the scientific core -- ReuseStats that let us measure
whether the library compounds (reuse value rises) or collapses (it is dead weight).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Allowed skill kinds. Each kind has a fixed function signature (below) that skill code
# must obey, so the agent can import & call any retrieved skill uniformly in CodeAct.
# The construct monolith is ATOMIZED (v2.1 patch action 1) into detect (diagnose) -> order ->
# build (construct), each independently creditable via lesion (analysis/lesion.py).
KIND = {"construct", "order", "local_search", "repair", "destroy", "diagnose", "strategy", "debug"}

SIGNATURES = {
    # construct is now BUILD-only: assemble a tour from a given structure + visit order. It must
    # NOT inline-detect (diagnose's job) or solve the cluster order (order's job). struct/order are
    # optional so a bare construct still runs (degraded) and old-signature constructs stay callable.
    "construct":    "def f(n, x, C, prims, struct=None, order=None) -> list[int]",
    "order":        "def f(struct, C, prims) -> list[int]",        # cluster ids in cheap visit order
    "local_search": "def f(tour, C, prims) -> list[int]",          # improves a tour
    "repair":       "def f(partial, n, C, prims) -> list[int]",
    "destroy":      "def f(tour, C, prims, k) -> tuple[list[int], list[int]]",  # (partial, removed)
    "diagnose":     "def f(tour, x, C, prims) -> dict",            # =DETECT: {"labels":[...], "k":K}
    "strategy":     "def f(n, x, C, skills, prims) -> list[int]",  # orchestrates other skills
    # §1.4 DEBUG = solve-time RECOVERY: rescue a failed/stuck/regressed tour into a better one.
    # Graded by the repair gap delta (gap before vs after) -- the cleanest grounded credit signal.
    "debug":        "def f(tour, C, prims) -> list[int]",          # recover a failed/stuck tour
}


@dataclass
class Contract:
    inputs: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    preserves: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)


@dataclass
class ReuseStats:
    """Scientific-core fields: how much the library is actually reused & worth."""
    invocations: int = 0            # total calls across instances
    instances_used_on: int = 0      # number of distinct instances that called it
    created_round: int = 0
    last_used_round: int = -1
    marginal_value: float = 0.0     # lesion LEAVE-ONE-OUT: mean-gap rise when this skill is removed
    marginal_add: float = 0.0       # §3.4 LEAVE-ONE-IN: gap drop when added back among other kinds
                                    # (credits weak-alone/strong-combo skills siblings would mask)
    reuse_rate: float = 0.0         # instances_used_on / (now_round - created_round)


@dataclass
class Skill:
    id: str                         # e.g. "ls_cluster_order_v3"
    kind: str                       # in KIND
    version: int
    name: str
    description_nl: str             # what it does / when to use (for retrieval & reflection)
    contract: Contract
    code: str                       # Python source obeying this kind's signature
    parent_ids: list[str]          # lineage from modify/merge
    created_round: int
    status: str                     # active | deprecated | pruned
    # §1.3 (plan+scaffold): declarative guidance -- do / avoid / why -- injected into the agent's
    # solve context to STEER it (the `code` above is the scaffold, ground-able by exact gap). Most
    # meaningful on kind=="strategy" (the library's plan-bearing face); blank on bare operators.
    plan: str = ""
    # §6 provenance: which problem family this skill was INDUCED on (+ created_round above). On a
    # cross-family transfer run, crediting Y's gap by origin_family separates reused X-strategy from
    # newly-induced Y-operators -- the signature of the transfer claim (§6.2).
    origin_family: str = ""
    meta: dict = field(default_factory=dict)        # {dev_gap_delta, feasibility_ok, ...}
    reuse: ReuseStats = field(default_factory=ReuseStats)
    tags: list[str] = field(default_factory=list)   # instance-feature tags (for retrieval)

    def signature(self) -> str:
        return SIGNATURES[self.kind]
