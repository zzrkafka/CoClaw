"""LKH-3 wrapper -- the main near-optimal ATSP reference (spec section 4.1).

Writes an ATSP instance in TSPLIB EXPLICIT/FULL_MATRIX form, runs the LKH binary, and
parses the resulting tour. LKH at n=50 is millisecond-fast and near-optimal, so its tour
length is our reference L_ref for gaps.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

# Diagonal sentinel for ATSP problem files: a valid tour never uses a self-loop, so the
# diagonal value is irrelevant to the optimum, but TSPLIB ATSP convention sets it large
# (not 0) to be safe across solver versions. Our off-diagonal costs are ~O(1e5), so 1e8
# is effectively infinite and well within int range.
_DIAG_INF = 10 ** 8


def find_lkh(explicit: str | None = None) -> str:
    """Locate the LKH binary. Order: explicit arg, $LKH_BINARY, PATH, ~/.local/bin/LKH."""
    for cand in (explicit, os.environ.get("LKH_BINARY"),
                 shutil.which("LKH"),
                 os.path.expanduser("~/.local/bin/LKH")):
        if cand and os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        "LKH binary not found. Build it via scripts/setup_wsl.sh, or set $LKH_BINARY."
    )


def _write_atsp(path: str, C: np.ndarray) -> None:
    n = len(C)
    M = np.array(C, dtype=np.int64, copy=True)
    np.fill_diagonal(M, _DIAG_INF)
    lines = [
        "NAME: coclaw",
        "TYPE: ATSP",
        f"DIMENSION: {n}",
        "EDGE_WEIGHT_TYPE: EXPLICIT",
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    # one matrix row per line (LKH parses whitespace-separated ints regardless of layout)
    for i in range(n):
        lines.append(" ".join(str(int(v)) for v in M[i]))
    lines.append("EOF")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_par(path: str, problem_file: str, tour_file: str,
               runs: int, max_trials: int, seed: int) -> None:
    with open(path, "w") as f:
        f.write(
            f"PROBLEM_FILE = {problem_file}\n"
            f"OUTPUT_TOUR_FILE = {tour_file}\n"
            f"RUNS = {runs}\n"
            f"MAX_TRIALS = {max_trials}\n"
            f"SEED = {seed}\n"
            # PRECISION=1: use integer costs as-is. The default (100) scales distances by
            # 100 internally, which overflows LKH's int representation for large masked
            # sentinels and trips `Gain % Precision == 0`. Our costs are already integers.
            f"PRECISION = 1\n"
            f"TRACE_LEVEL = 0\n"
        )


def _parse_tour(path: str, n: int) -> list[int]:
    """Parse a TSPLIB tour file -> 0-based permutation."""
    nodes: list[int] = []
    in_section = False
    with open(path) as f:
        for raw in f:
            tok = raw.strip()
            if not in_section:
                if tok.upper().startswith("TOUR_SECTION"):
                    in_section = True
                continue
            if tok in ("-1", "EOF", ""):
                if tok == "-1" or tok == "EOF":
                    break
                continue
            nodes.append(int(tok) - 1)  # TSPLIB is 1-based
    if sorted(nodes) != list(range(n)):
        raise ValueError(f"LKH tour is not a permutation of range({n}): got {len(nodes)} nodes")
    return nodes


def solve_atsp(C, runs: int = 10, max_trials: int = 10000, seed: int = 1,
               lkh_bin: str | None = None, timeout: float = 120.0) -> list[int]:
    """Solve the ATSP defined by integer cost matrix C; return a 0-based tour."""
    C = np.asarray(C)
    n = len(C)
    binary = find_lkh(lkh_bin)
    with tempfile.TemporaryDirectory(prefix="coclaw_lkh_") as d:
        prob = os.path.join(d, "p.atsp")
        par = os.path.join(d, "p.par")
        tour = os.path.join(d, "p.tour")
        _write_atsp(prob, C)
        _write_par(par, prob, tour, runs, max_trials, seed)
        proc = subprocess.run([binary, par], cwd=d, capture_output=True,
                              text=True, timeout=timeout)
        if not os.path.exists(tour):
            raise RuntimeError(
                f"LKH produced no tour (rc={proc.returncode}).\n"
                f"stdout tail:\n{proc.stdout[-800:]}\nstderr tail:\n{proc.stderr[-800:]}"
            )
        return _parse_tour(tour, n)
