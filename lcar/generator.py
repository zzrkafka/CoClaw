"""LCAR -- Latent-Cluster Asymmetric Routing (spec section 3).

The testbed. Each instance is an asymmetric, triangle-inequality-violating routing
problem whose cheap structure lives in a HIDDEN cluster cycle, while the VISIBLE
coordinates are a decoy. The hidden structure (pi, cheap_cycle, G) is fixed across a
family, so it is discoverable and reusable -- which is what makes a skill library
meaningful here.

Design goals (see section 3.1):
  - asymmetric + triangle-violating costs  -> defeat memorized Euclidean NN / 2-opt
  - hidden, cross-instance-fixed structure -> there is something to accumulate
  - misleading visible coords              -> geometric-nearest != cost-cheap
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def unit_circle_points(K: int) -> np.ndarray:
    """K points evenly spaced on the unit circle -- the hidden type centers mu."""
    ang = 2.0 * np.pi * np.arange(K) / K
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def square_corner_points(K: int, scale: float = 2.0) -> np.ndarray:
    """K visible-cluster centers nu with neat geometry, well separated from each other.

    For K == 4 these are literally the corners of a square (the canonical LCAR case);
    for other K they are evenly placed on a circle of radius `scale` (still tidy and
    separated). The point is only that the VISIBLE clusters look clean and regular --
    their tidiness is the decoy.
    """
    if K == 4:
        s = scale
        return np.array([[s, s], [-s, s], [-s, -s], [s, -s]], dtype=float)
    ang = 2.0 * np.pi * np.arange(K) / K
    return scale * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def pairwise_euclidean(z: np.ndarray) -> np.ndarray:
    """Full symmetric pairwise Euclidean distance matrix (the metric term, on z)."""
    diff = z[:, None, :] - z[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


# ---------------------------------------------------------------------------
# instance container
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    n: int
    x: np.ndarray                       # n x 2 visible coords (may mislead)
    C: np.ndarray                       # n x n asymmetric INTEGER cost matrix = the objective
    hidden: dict = field(default_factory=dict)   # {k, v} -- analysis/oracle ONLY, never given to agent
    ref: dict = field(default_factory=dict)      # filled later: L_lkh / L_exact / gaps
    id: str = ""
    family: str = "F"
    seed: int | None = None


# ---------------------------------------------------------------------------
# family
# ---------------------------------------------------------------------------
class Family:
    """A fixed LCAR family: shared hidden structure (mu, nu, pi, G, cheap_cycle).

    Sampling an instance draws fresh nodes but reuses this structure, so a skill that
    captures the cheap cluster order generalizes across the family's instances.
    """

    def __init__(self, K, alpha, beta, sigma_z, sigma_x, eps_sigma,
                 cost_scale, pi, cheap_cycle, G_cheap, G_exp, seed=0, name="F",
                 g_mode="cheap_cycle", g_seed=0):
        self.K = int(K)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma_z = float(sigma_z)
        self.sigma_x = float(sigma_x)
        self.eps_sigma = float(eps_sigma)
        self.cost_scale = int(cost_scale)
        self.G_cheap = float(G_cheap)
        self.G_exp = float(G_exp)
        self.seed = int(seed)
        self.name = name

        self.mu = unit_circle_points(self.K)        # hidden type centers
        self.nu = square_corner_points(self.K)      # visible cluster centers (tidy geometry)
        self.pi = np.asarray(pi, dtype=int)         # visible cluster v -> cost-type pi[v]
        self.cheap_cycle = list(cheap_cycle)        # type order of the cheap forward cycle

        # G: asymmetric transition cost between hidden types.
        self.g_mode = g_mode
        if g_mode == "random":
            # DECEPTIVE: a fixed random asymmetric inter-cluster cost = a hard small TSP over
            # clusters. The optimal cluster-VISIT-ORDER requires solving this TSP. Cheap local
            # search over NODES cannot reorder whole clusters, so it gets stuck near a greedy
            # order; cheap_cluster_order (which brute-forces the K-cluster TSP) finds the
            # optimum -> the structural skill reaches quality self-discovery cannot.
            grng = np.random.default_rng(g_seed)
            self.G = grng.uniform(self.G_cheap, self.G_exp, (self.K, self.K))
            np.fill_diagonal(self.G, 0.0)
        else:
            # forward edges along cheap_cycle are cheap (G_cheap); everything else expensive.
            self.G = np.full((self.K, self.K), self.G_exp, dtype=float)
            np.fill_diagonal(self.G, 0.0)           # within-type transitions are free
            for idx, a in enumerate(self.cheap_cycle):
                nxt = self.cheap_cycle[(idx + 1) % self.K]
                self.G[a, nxt] = self.G_cheap

    def sample_instance(self, n: int, seed: int, instance_id: str = "") -> Instance:
        rng = np.random.default_rng(seed)
        v = rng.integers(0, self.K, n)                      # visible cluster of each node
        x = self.nu[v] + rng.normal(0, self.sigma_x, (n, 2))
        k = self.pi[v]                                      # hidden type (shuffled from v)
        z = self.mu[k] + rng.normal(0, self.sigma_z, (n, 2))

        d = pairwise_euclidean(z)                           # metric term (symmetric)
        trans = self.G[k[:, None], k[None, :]]              # structure term (asymmetric)
        eps = rng.normal(0, self.eps_sigma, (n, n))         # per-ordered-pair noise

        c = self.alpha * d + self.beta * trans + eps
        C = np.round(np.maximum(c, 0.0) * self.cost_scale).astype(int)
        np.fill_diagonal(C, 0)

        return Instance(
            n=n, x=x, C=C,
            hidden=dict(k=k, v=v),     # never exposed to the agent
            ref={},
            id=instance_id or f"{self.name}_n{n}_s{seed}",
            family=self.name,
            seed=seed,
        )

    # -- construction from config ------------------------------------------------
    @classmethod
    def from_config(cls, lcar_cfg: dict, seed: int = 0, name: str = "F",
                    overrides: dict | None = None) -> "Family":
        """Build a Family from the `lcar:` config block, optionally overriding fields
        (used for F_prime, which overrides pi and cheap_cycle)."""
        params = dict(
            K=lcar_cfg["K"], alpha=lcar_cfg["alpha"], beta=lcar_cfg["beta"],
            sigma_z=lcar_cfg["sigma_z"], sigma_x=lcar_cfg["sigma_x"],
            eps_sigma=lcar_cfg["eps_sigma"], cost_scale=lcar_cfg["cost_scale"],
            pi=lcar_cfg["pi"], cheap_cycle=lcar_cfg["cheap_cycle"],
            G_cheap=lcar_cfg["G_cheap"], G_exp=lcar_cfg["G_exp"],
            g_mode=lcar_cfg.get("g_mode", "cheap_cycle"), g_seed=lcar_cfg.get("g_seed", 0),
        )
        if overrides:
            params.update(overrides)
        return cls(seed=seed, name=name, **params)
