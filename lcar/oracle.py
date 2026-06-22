"""Structure oracle + toolkit baseline (spec section 3.4).

These bracket the "discovery space" the skill library must traverse:
  - toolkit_baseline : what a memorized Euclidean agent does -> should be BAD on C (gate 1)
  - structure_oracle : cheats with the hidden cluster order -> should be NEAR-OPTIMAL (gate 2)

How far the library moves from the toolkit line toward the oracle line == how much it
compounds.
"""
from __future__ import annotations

import numpy as np

from .generator import pairwise_euclidean


def euclidean_matrix(x: np.ndarray) -> np.ndarray:
    """Symmetric Euclidean distance matrix over the visible coords x (the decoy metric)."""
    return pairwise_euclidean(np.asarray(x, dtype=float))


def nearest_neighbor(coords) -> list[int]:
    """Greedy nearest-neighbor tour over `coords` (geometry only), starting at node 0."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    D = euclidean_matrix(coords)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        nxt, best = -1, np.inf
        for j in range(n):
            if not visited[j] and D[last, j] < best:
                best, nxt = D[last, j], j
        tour.append(nxt)
        visited[nxt] = True
    return tour


def two_opt(nodes, C, closed: bool = True, max_pass: int = 60) -> list[int]:
    """2-opt local search over an ordering of `nodes`, scored by matrix `C`.

    closed=True  -> closed cycle (full tour; toolkit baseline).
    closed=False -> open path (cluster segment; structure oracle), endpoints free for
                    cheap inter-cluster stitching.

    Uses an O(1)-delta fast path when C is symmetric and closed (the toolkit's n=50 case).
    For an asymmetric C, reversing a segment also flips its internal edge directions, so
    we recompute the affected-edge delta exactly -- only ever on small cluster segments.
    """
    t = list(nodes)
    m = len(t)
    if m < 4:
        return t
    C = np.asarray(C)
    if closed and C.shape[0] == C.shape[1] and np.array_equal(C, C.T):
        return _two_opt_symmetric_closed(t, C, max_pass)
    return _two_opt_general(t, C, closed, max_pass)


def _two_opt_symmetric_closed(t: list[int], D, max_pass: int) -> list[int]:
    """O(1)-delta 2-opt for a CLOSED tour under a SYMMETRIC matrix D (first-improvement)."""
    m = len(t)
    improved, passes = True, 0
    while improved and passes < max_pass:
        improved, passes = False, passes + 1
        for i in range(1, m - 1):
            a = t[i - 1]
            b = t[i]
            d_ab = D[a, b]
            for j in range(i + 1, m):
                c = t[j]
                e = t[(j + 1) % m]
                if a == e:                       # edges share a node through the wrap
                    continue
                # drop (a,b)+(c,e), add (a,c)+(b,e) by reversing t[i..j]
                if D[a, c] + D[b, e] < d_ab + D[c, e] - 1e-9:
                    t[i:j + 1] = t[i:j + 1][::-1]
                    b = t[i]
                    d_ab = D[a, b]
                    improved = True
    return t


def _two_opt_general(t: list[int], C, closed: bool, max_pass: int) -> list[int]:
    """2-opt for possibly-ASYMMETRIC C with incremental move deltas.

    Reversing segment t[i+1..j] flips the direction of its internal edges, so the move
    delta carries a running sum of reversed-vs-forward internal edges, maintained as the
    segment grows (O(1) amortized per candidate, O(m^2) per pass). First-improvement.
    """
    m = len(t)
    eps = 1e-9
    improved, passes = True, 0
    while improved and passes < max_pass:
        improved, passes = False, passes + 1
        for i in range(m - 1):
            ti, ti1 = t[i], t[i + 1]
            base_left = C[ti, ti1]
            fi = ri = 0  # forward / reversed internal-edge sums over t[i+1..j]
            for j in range(i + 2, m):
                fi += C[t[j - 1], t[j]]
                ri += C[t[j], t[j - 1]]
                if closed:
                    nxt = t[(j + 1) % m]
                    if ti == nxt:                      # degenerate (shares a node)
                        continue
                    old = base_left + fi + C[t[j], nxt]
                    new = C[ti, t[j]] + ri + C[ti1, nxt]
                elif j + 1 < m:
                    nxt = t[j + 1]
                    old = base_left + fi + C[t[j], nxt]
                    new = C[ti, t[j]] + ri + C[ti1, nxt]
                else:                                  # open path, reversing the suffix
                    old = base_left + fi
                    new = C[ti, t[j]] + ri
                if new < old - eps:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
                    break  # tour changed; running sums stale -> next i
    return t


def toolkit_baseline(inst) -> list[int]:
    """What a recalled Euclidean agent would do: NN + 2-opt on the VISIBLE geometry,
    blind to the C structure. Scored on C externally -> should be poor (gate 1)."""
    t = nearest_neighbor(inst.x)
    return two_opt(t, euclidean_matrix(inst.x), closed=True)


def structure_oracle(inst, fam) -> list[int]:
    """Near-optimal structure-aware upper bound (gate 2).

    Cheats with the HIDDEN types k + the family's G: mask out every EXPENSIVE cross-type
    transition (keep only intra-cluster edges and cheap forward-cycle boundaries), then
    LKH-solve the structure-restricted ATSP. Because the true optimum is itself
    structure-respecting (clusters visited contiguously in cheap_cycle order -- verified:
    LKH tours have exactly K type-runs), the restricted optimum ~= the global optimum.
    So gate 2 measures: does KNOWING the structure let you reach ~optimal? (Yes.)

    A simple cluster-order + 2-opt heuristic is NOT enough here: the cost is dominated by
    the K inter-cluster bridge edges, and choosing good bridge nodes is a global decision
    that local search handles poorly -- hence we solve the restricted problem properly.
    """
    from solvers import lkh  # lazy import keeps lcar/ light and avoids import cycles

    k = inst.hidden["k"]
    Gpair = fam.G[k[:, None], k[None, :]]                 # G value per node pair, by type
    allowed = (k[:, None] == k[None, :]) | (Gpair == fam.G_cheap)
    # forbid expensive transitions: 1e7 >> any real tour cost (~1e6) so they are never
    # chosen, yet small enough to stay well within LKH's int range under PRECISION=1.
    INF = 10 ** 7
    C_masked = np.where(allowed, inst.C, INF).astype(int)
    np.fill_diagonal(C_masked, 0)
    return lkh.solve_atsp(C_masked, runs=10, max_trials=10000, seed=1)
