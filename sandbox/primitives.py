"""Primitives available to agent code inside the CodeAct sandbox (spec section 6).

These are the ONLY non-numpy helpers agent-written code may call (besides retrieved
skills). Cost access goes through CountingCosts so that `n_evals` -- the count of
ground-truth cost lookups -- becomes an observable proxy for foresight (faculty F4).
"""
from __future__ import annotations

import numpy as np


def dist_matrix(coords) -> np.ndarray:
    """Full pairwise Euclidean distance matrix for n x d coordinates.

    NOTE: this is geometry over the *visible* coords x. It is NOT the objective C and
    may be actively misleading on LCAR -- exposed only because a naive agent expects it.
    """
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def tour_length(tour, C) -> int:
    """Closed-tour cost: sum of C[t[i], t[i+1]] with wraparound.

    Works with a plain ndarray or a CountingCosts wrapper (each lookup is counted).
    """
    n = len(tour)
    total = 0
    for i in range(n):
        total += C[tour[i], tour[(i + 1) % n]]
    return int(total)


def is_valid_tour(tour, n) -> bool:
    """True iff `tour` is a permutation of range(n) (a feasible Hamiltonian cycle)."""
    if tour is None or len(tour) != n:
        return False
    seen = sorted(int(t) for t in tour)
    return seen == list(range(n))


def sanitize_tour(tour, n, C=None) -> list:
    """Repair any output into a valid permutation of range(n): drop out-of-range / duplicate
    nodes, then re-insert missing nodes at their CHEAPEST position (greedy). The harness applies
    this to every skill's tour output, so a mechanical slip (dropped start node, duplicate) is
    fixed and the skill is judged on QUALITY -- never rejected for a permutation error.
    """
    seen, out = set(), []
    for v in (tour or []):
        try:
            v = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= v < n and v not in seen:
            seen.add(v)
            out.append(v)
    if not out:
        return list(range(n))
    for m in (i for i in range(n) if i not in seen):
        if C is None or len(out) < 2:
            out.append(m)
            continue
        best_pos, best_inc = len(out) - 1, None
        for p in range(len(out)):
            a, b = out[p], out[(p + 1) % len(out)]
            inc = C[a, m] + C[m, b] - C[a, b]
            if best_inc is None or inc < best_inc:
                best_inc, best_pos = inc, p
        out.insert(best_pos + 1, m)
    return out


def _read_full(C) -> np.ndarray:
    """Read the whole cost matrix once through the (counted) accessor -> local float array."""
    n = C.shape[0] if hasattr(C, "shape") else len(C)
    return np.array([[C[i, j] for j in range(n)] for i in range(n)], dtype=float)


def cluster_by_cost(C) -> "np.ndarray":
    """Structural clustering of nodes by their COST PROFILES (rows of symmetrized C).

    Nodes of the same latent type have similar cost-to-everyone profiles, so KMeans on the
    rows (k auto-picked by silhouette) recovers the latent clusters -- unlike a single cost
    threshold, which merges clusters via the cheap forward-boundary edges. Reads C once
    (~n^2 counted lookups). Returns int labels.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    M = _read_full(C)
    n = len(M)
    S = 0.5 * (M + M.T)
    best_labels, best_s = None, -1.0
    for k in range(2, min(9, n)):
        labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(S)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(S, labels)
        if s > best_s:
            best_s, best_labels = s, labels
    return best_labels if best_labels is not None else np.zeros(n, dtype=int)


def _cheapest_cluster_cycle(IC: np.ndarray) -> list[int]:
    """Cheapest DIRECTED Hamiltonian cycle over K clusters given inter-cluster cost IC."""
    K = len(IC)
    if K <= 8:
        from itertools import permutations
        best, best_order = None, list(range(K))
        for p in permutations(range(1, K)):
            order = [0] + list(p)
            cost = sum(IC[order[i], order[(i + 1) % K]] for i in range(K))
            if best is None or cost < best:
                best, best_order = cost, order
        return best_order
    visited = [False] * K
    order = [0]
    visited[0] = True
    for _ in range(K - 1):
        last = order[-1]
        nxt = min((j for j in range(K) if not visited[j]), key=lambda j: IC[last, j])
        order.append(nxt)
        visited[nxt] = True
    return order


def cheap_cluster_order(C, labels) -> list[int]:
    """The CHEAP directed order to VISIT clusters (the latent structure of LCAR).

    Given node->cluster labels, build the inter-cluster cost (cheapest edge between each
    ordered cluster pair) and return the cheapest directed cycle over clusters. Visiting
    clusters in this order pays the cheap forward transitions instead of the expensive ones.
    Returns a list of cluster ids in visit order.
    """
    M = _read_full(C)
    labels = np.asarray([int(v) for v in labels])
    clusters = sorted(set(labels.tolist()))
    K = len(clusters)
    idx = {c: np.where(labels == c)[0] for c in clusters}
    IC = np.full((K, K), np.inf)
    for ai, a in enumerate(clusters):
        for bi, b in enumerate(clusters):
            if ai != bi and len(idx[a]) and len(idx[b]):
                IC[ai, bi] = M[np.ix_(idx[a], idx[b])].min()
    order = _cheapest_cluster_cycle(IC)
    return [clusters[i] for i in order]


class EvalBudgetExceeded(Exception):
    """Raised when a solve's cost-evaluation budget is exhausted -- forces the agent to be
    efficient (exploit structure) rather than brute-force search."""


class CountingCosts:
    """Wrap the integer cost matrix C; count every element access.

    Agent code reads costs as C[i, j] (or C[(i, j)]). Each read adds the number of elements
    touched to .count, so the session-cumulative .count == n_evals, the F4 foresight signal:
    learning to look before you leap shows up as the same gap reached with fewer cost
    evaluations. With a `budget`, exceeding it raises EvalBudgetExceeded (a regime where
    quality is NOT free via brute force, so a structure-exploiting skill pays off).
    """

    __slots__ = ("C", "count", "budget")

    def __init__(self, C, budget=None):
        self.C = np.asarray(C)
        self.count = 0
        self.budget = budget

    def __getitem__(self, ij):
        val = self.C[ij]
        self.count += int(getattr(val, "size", 1))  # vectorized reads count their elements
        if self.budget is not None and self.count > self.budget:
            raise EvalBudgetExceeded(
                f"cost-eval budget {self.budget} exhausted (used {self.count}); "
                f"exploit the structure of C instead of brute-force search")
        return val

    @property
    def shape(self):
        return self.C.shape

    def __len__(self):
        return len(self.C)

    def raw(self) -> np.ndarray:
        """The underlying ndarray, WITHOUT counting -- analysis/reference use only."""
        return self.C
