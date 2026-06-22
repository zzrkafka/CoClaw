"""Exact small-n ATSP via Held-Karp DP -- for validity gate 3 without a Gurobi license.

Gate 3 (spec 4.3) checks that LKH actually reaches the optimum on small instances. For
n up to ~16 this exact DP is instant and needs no external solver. For larger exact
checks, use solvers/gurobi_exact.py (academic license).
"""
from __future__ import annotations

import numpy as np

MAX_N = 16


def held_karp_atsp(C) -> tuple[int, list[int]]:
    """Optimal ATSP tour for the integer cost matrix C (n <= MAX_N). Returns (length, tour)."""
    C = np.asarray(C)
    n = len(C)
    if n > MAX_N:
        raise ValueError(f"held_karp_atsp supports n <= {MAX_N}; got n={n}. Use Gurobi for larger n.")
    INF = float("inf")
    size = 1 << n
    full = size - 1
    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    dp[1][0] = 0  # path covering only node 0, ending at 0

    for mask in range(size):
        if not (mask & 1):
            continue  # node 0 is the fixed start; every state includes it
        row = dp[mask]
        for j in range(n):
            base = row[j]
            if base == INF or not (mask >> j) & 1:
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                nc = base + int(C[j, k])
                if nc < dp[nm][k]:
                    dp[nm][k] = nc
                    parent[nm][k] = j

    best, best_j = INF, -1
    for j in range(1, n):
        if dp[full][j] == INF:
            continue
        c = dp[full][j] + int(C[j, 0])
        if c < best:
            best, best_j = c, j

    tour: list[int] = []
    mask, j = full, best_j
    while j != -1:
        tour.append(j)
        pj = parent[mask][j]
        mask ^= (1 << j)
        j = pj
    tour.reverse()
    return int(best), tour
