"""Instance features for skill retrieval + analysis (spec section 8 / 7.3).

Computed ONLY from what the agent sees -- n, x, C -- never from inst.hidden. These tag
instances for retrieval and let analysis ask "did the agent's view (e.g. C-cluster
structure) line up with the latent truth?" without leaking the truth into the solve.
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    _HAVE_SK = True
except Exception:  # noqa: BLE001
    _HAVE_SK = False


def _best_kmeans(points: np.ndarray, ks=range(2, 7)) -> tuple[int, float]:
    """Best (k, silhouette) for KMeans over `points` (rows = samples)."""
    n = len(points)
    if not _HAVE_SK or n < 4:
        return 0, 0.0
    best_k, best_s = 0, -1.0
    for k in ks:
        if k >= n:
            break
        try:
            labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(points)
            if len(set(labels)) < 2:
                continue
            s = silhouette_score(points, labels)
        except Exception:  # noqa: BLE001
            continue
        if s > best_s:
            best_k, best_s = k, float(s)
    return best_k, max(0.0, best_s)


def _triangle_violation_rate(C: np.ndarray, samples: int = 3000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(C)
    if n < 3:
        return 0.0
    viol = 0
    for _ in range(samples):
        i, j, k = rng.integers(0, n, 3)
        if i == j or j == k or i == k:
            continue
        if C[i, k] > C[i, j] + C[j, k]:
            viol += 1
    return viol / samples


def featurize(inst) -> dict:
    """Return a flat dict of scalar features (no hidden info)."""
    x = np.asarray(inst.x, dtype=float)
    C = np.asarray(inst.C, dtype=float)
    n = inst.n
    off = ~np.eye(n, dtype=bool)
    c = C[off]
    cmean = float(c.mean()) if c.size else 0.0

    asym = float(np.abs(C - C.T)[off].mean())
    # latent structure visible from C: cluster nodes by their symmetrized cost profiles
    Csym = 0.5 * (C + C.T)
    c_k, c_sil = _best_kmeans(Csym)
    x_k, x_sil = _best_kmeans(x)

    return {
        "n": n,
        "cost_mean": cmean,
        "cost_std": float(c.std()) if c.size else 0.0,
        "cost_cv": float(c.std() / cmean) if cmean else 0.0,
        "cost_min": float(c.min()) if c.size else 0.0,
        "cost_max": float(c.max()) if c.size else 0.0,
        "asymmetry": asym,                          # mean |C - C^T| over off-diagonal
        "asymmetry_ratio": float(asym / cmean) if cmean else 0.0,
        "triangle_violation_rate": _triangle_violation_rate(C),
        "x_n_clusters": x_k,                        # visible geometric cluster count
        "x_silhouette": x_sil,
        "c_n_clusters": c_k,                        # cluster count inferable from C structure
        "c_silhouette": c_sil,
        "cluster_geometry_agrees": int(x_k == c_k) if (x_k and c_k) else 0,
    }


def feature_tags(features: dict) -> list[str]:
    """Coarse string tags for retrieval indexing."""
    tags = [f"n~{int(round(features['n'] / 10) * 10)}"]
    if features.get("asymmetry_ratio", 0) > 0.1:
        tags.append("asymmetric")
    if features.get("triangle_violation_rate", 0) > 0.02:
        tags.append("triangle_violating")
    if features.get("c_n_clusters", 0):
        tags.append(f"c_clusters~{features['c_n_clusters']}")
    if features.get("c_silhouette", 0) > 0.4:
        tags.append("strong_cost_clusters")
    return tags
