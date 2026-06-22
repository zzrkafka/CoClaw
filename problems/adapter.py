"""Problem-adapter interface (improvement-plan §5.4a) -- INTERFACE ONLY (Stage-B discipline).

The harness + science loop (solve / evolve / lesion / curation) are problem-agnostic; everything
problem-specific is reached through a small adapter:

    feasible(sol, inst)   -> bool        # is this a valid solution?
    objective(sol, inst)  -> float       # the TRUE cost we minimize (lower = better)
    gap(sol, inst)        -> float        # (objective - reference) / reference  (the exact ruler)
    fill_reference(inst)                  # attach the exact/near-exact reference (LKH etc.)
    typed_vocab           -> kinds        # the operator kinds the strategy dispatches over
    solution_repr         -> str          # how a solution is represented (doc/handle)

This file DEFINES the interface and a faithful routing adapter that wraps the EXISTING F_hard
functions, so adopting it is zero-behavior-change. It deliberately does NOT reroute the hot path
(solver/evolve still call the functions directly) -- that swap is Stage C, when a second family
(CVRP / MaxCut) is actually added. Per the plan's discipline: design the interface now, do NOT add
new problems yet (that would pollute "does generalization EMERGE"). cross-family transfer is then a
matter of registering a second adapter with its own typed_vocab + reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# the typed operator vocabulary the routing family dispatches over (schema.KIND); a different family
# supplies its own (e.g. a scheduling family: detect-bottleneck / order-ops / build-schedule / ...).
_ROUTING_KINDS = ("diagnose", "order", "construct", "local_search", "repair", "destroy",
                  "strategy", "debug")


@dataclass
class ProblemAdapter:
    """A problem family behind a uniform interface. The fields are callables so a new family just
    supplies its own feasibility / objective / reference without touching the science loop."""
    name: str
    feasible: Callable          # (sol, inst) -> bool
    objective: Callable         # (sol, inst) -> float (minimize)
    gap: Callable               # (sol, inst) -> float (exact ruler)
    fill_reference: Callable    # (inst) -> None (attach reference in-place)
    typed_vocab: tuple = field(default_factory=tuple)
    solution_repr: str = ""


def routing_adapter() -> ProblemAdapter:
    """F_hard / LCAR routing, wrapping the existing functions verbatim (faithful = zero-change)."""
    from sandbox.primitives import is_valid_tour, tour_length
    from solvers.reference import fill_reference, gap

    def feasible(sol, inst):
        return sol is not None and is_valid_tour(sol, inst.n)

    def objective(sol, inst):
        return tour_length(sol, inst.C)

    return ProblemAdapter(
        name="routing", feasible=feasible, objective=objective,
        gap=lambda sol, inst: gap(sol, inst), fill_reference=fill_reference,
        typed_vocab=_ROUTING_KINDS,
        solution_repr="a tour = permutation of range(n); cost = sum C[t[i], t[i+1]] around the cycle",
    )


# registry -- get_adapter(name) is the single lookup the loop would use once routed through adapters.
_REGISTRY: dict[str, Callable[[], ProblemAdapter]] = {"routing": routing_adapter}


def get_adapter(name: str = "routing") -> ProblemAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown problem adapter {name!r}; registered: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def register_adapter(name: str, factory: Callable[[], ProblemAdapter]) -> None:
    """Stage-C hook: register a second family (CVRP / MaxCut) without touching the science loop."""
    _REGISTRY[name] = factory
