"""Unified gap + reference filling (spec section 4.3).

gap(tour, inst) is THE objective metric across the whole project: relative excess over the
LKH reference length. Lower is better; 0 means matching LKH.
"""
from __future__ import annotations

from sandbox.primitives import tour_length
from solvers import lkh


def gap(tour, inst) -> float:
    """Relative gap of `tour` vs the LKH reference for `inst` (requires ref['L_lkh'])."""
    L_ref = inst.ref["L_lkh"]
    return (tour_length(tour, inst.C) - L_ref) / L_ref


def fill_reference(inst, runs: int = 10, max_trials: int = 10000, seed: int = 1) -> int:
    """Run LKH on inst.C and store L_lkh (+ the tour) in inst.ref. Returns L_lkh."""
    tour = lkh.solve_atsp(inst.C, runs=runs, max_trials=max_trials, seed=seed)
    L = tour_length(tour, inst.C)
    inst.ref["L_lkh"] = L
    inst.ref["lkh_tour"] = tour
    return L
