"""Sandbox execution (spec section 6) -- WSL2/Ubuntu (POSIX).

Two execution faces:
  - CodeActSession : STATEFUL kernel. The agent emits one code action at a time; variables
                     (and the cumulative cost-eval count) persist across actions. This is
                     what makes the solve a multi-step CodeAct loop (INV-2), not one-shot
                     program synthesis.
  - run_in_sandbox : ONE-SHOT. Execute a code blob once and read back a tour -- used to
                     evaluate a candidate skill (the grounded curation judge, section 9).

Isolation: each session runs in a forked child process with setrlimit (CPU + address
space) and a per-action timeout (SIGALRM soft inside the child; a parent-side hard kill as
backstop). Costs are read through CountingCosts so n_evals is observable (faculty F4).
"""
from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
import resource
import signal
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from .primitives import (CountingCosts, cluster_by_cost,
                         dist_matrix, is_valid_tour, sanitize_tour, tour_length)


@dataclass
class Limits:
    # wall_timeout is the PRIMARY per-action cap (SIGALRM -> catchable TimeoutError, the
    # session survives). cpu_seconds is a GENEROUS backstop (must be > wall_timeout); if it
    # ever fires it kills the kernel, which the session then restarts. mem_bytes caps the
    # address space (MemoryError, usually catchable).
    cpu_seconds: int = 60
    mem_bytes: int = 2 * 1024 ** 3
    wall_timeout: float = 15.0


@dataclass
class Obs:
    summary: str                         # compact stdout/err the agent sees next turn
    finalized: bool = False              # agent set a valid FINAL_TOUR
    tour: list | None = None             # current/last full tour produced this action
    skills_called: list = field(default_factory=list)
    n_evals: int = 0                     # cumulative cost-matrix lookups (whole session)
    exec_ms: float = 0.0
    error: str | None = None
    hang_trace: str = ""                 # faulthandler dump of WHERE it stalled (for the debugger)


@dataclass
class Result:                            # one-shot run_in_sandbox return
    tour: list | None
    feasible: bool
    n_evals: int
    exec_ms: float
    error: str | None = None
    hang_trace: str = ""


class _Prims:
    """The `prims` object exposed to agent/skill code."""
    dist_matrix = staticmethod(dist_matrix)
    tour_length = staticmethod(tour_length)
    is_valid_tour = staticmethod(is_valid_tour)
    sanitize_tour = staticmethod(sanitize_tour)               # repair -> valid permutation
    cluster_by_cost = staticmethod(cluster_by_cost)            # node -> cluster labels (from C)
    # NOTE: cheap_cluster_order is deliberately NOT a primitive -- the agent must INDUCE the
    # small cluster-TSP solver, so the library's accumulation of it is what compounds.


_HANG_TRACE = {"s": ""}   # SIGALRM captures the stack here (no watchdog thread -> no BLAS clash)


def _timeout_handler(signum, frame):
    try:
        _HANG_TRACE["s"] = "".join(traceback.format_stack(frame))
    except Exception:  # noqa: BLE001
        pass
    raise TimeoutError("action exceeded wall-clock budget")


class SkillDict(dict):
    """The `skills` mapping handed to agent + strategy code. Backward-compatible -- it IS a dict, so
    skills["<id>"](...) is unchanged -- but it ALSO supports kind-based dispatch so a strategy
    scaffold can compose operators by KIND without hard-coding ids (§2.1-§2.3). Unknown kind -> []."""
    def set_kinds(self, kinds: dict) -> None:
        self._kinds = dict(kinds)

    def ids_of_kind(self, kind: str) -> list:
        return [sid for sid in self if getattr(self, "_kinds", {}).get(sid) == kind]

    def of_kind(self, kind: str) -> list:
        return [self[sid] for sid in self.ids_of_kind(kind)]

    def first_of_kind(self, kind: str):
        ids = self.ids_of_kind(kind)
        return self[ids[0]] if ids else None


def _make_namespace(ctx):
    """Build the persistent execution namespace from ctx={n, x, C, skills, eval_budget?}."""
    costs = CountingCosts(ctx["C"], budget=ctx.get("eval_budget"))
    prims = _Prims()
    skills_called: list[str] = []

    skills = SkillDict()
    skill_kinds: dict[str, str] = {}
    for s in ctx.get("skills", []):
        sid, code = s["id"], s["code"]
        skill_kinds[sid] = s.get("kind", "")
        loc: dict = {}
        try:
            exec(code, {"np": np, "prims": prims}, loc)
        except Exception as e:  # noqa: BLE001  -- a malformed skill must not kill the kernel
            skills[sid] = _broken_skill(sid, repr(e))
            continue
        fn = loc.get("f")
        skills[sid] = _wrap_skill(sid, fn, skills_called) if callable(fn) else _broken_skill(sid, "no f()")
    skills.set_kinds(skill_kinds)

    ns = {
        "np": np, "math": math,
        "n": ctx["n"], "x": ctx["x"], "C": costs,
        "prims": prims, "skills": skills, "skill_kinds": skill_kinds,
        "__skills_called__": skills_called,
    }
    return ns, costs, skills_called


def _wrap_skill(sid, fn, called):
    def w(*a, **k):
        called.append(sid)
        return fn(*a, **k)
    w.__name__ = f"skill_{sid}"
    return w


def _broken_skill(sid, why):
    def w(*a, **k):
        raise RuntimeError(f"skill {sid} is unavailable: {why}")
    return w


def _skill_frames(raw: str) -> str:
    """Pull the skill's own frames out of a faulthandler dump (the agent code runs as
    '<string>'); fall back to the tail if the stall is inside a C/library call."""
    if not raw:
        return ""
    lines = [l for l in raw.splitlines() if "<string>" in l or "Timeout" in l]
    return ("\n".join(lines) or raw.strip()[-700:])[:900]


def _exec_action(ns, costs, skills_called, code, limits) -> Obs:
    """Execute one code action in the persistent namespace `ns`; return an Obs."""
    skills_called.clear()
    buf = io.StringIO()
    err = None
    _HANG_TRACE["s"] = ""
    t0 = time.perf_counter()
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max(0.05, limits.wall_timeout))
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, ns)
    except Exception as e:  # noqa: BLE001  -- agent code errors are observations, not crashes
        err = f"{type(e).__name__}: {e}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    exec_ms = (time.perf_counter() - t0) * 1000.0
    # on a (catchable) wall-timeout the handler captured the stack -> WHERE it hung, for the debugger
    hang_trace = _skill_frames(_HANG_TRACE["s"]) if _HANG_TRACE["s"] else ""

    # FINAL_TOUR is the running answer: captured EVERY action (so good work is never lost),
    # updatable, and does NOT end the solve. The agent ends early by setting DONE = True.
    tour = None
    ft = ns.get("FINAL_TOUR")
    if ft is not None:
        try:
            cand = [int(v) for v in ft]
        except Exception:  # noqa: BLE001
            cand = None
        if cand is not None and is_valid_tour(cand, ns["n"]):
            tour = cand
        elif cand is not None:
            # AUTO-REPAIR rather than reject: a dropped/duplicate node must not kill the attempt.
            tour = sanitize_tour(cand, ns["n"], ns["C"])
            err = (err + " | " if err else "") + "FINAL_TOUR auto-repaired to a permutation"
        else:
            err = (err + " | " if err else "") + "FINAL_TOUR is not a list of ints"
    finalized = bool(ns.get("DONE"))
    out = buf.getvalue()
    summary = out[-1500:] if out else ""
    if err:
        summary = (summary + "\n" if summary else "") + f"[error] {err}"
    return Obs(summary=summary.strip(), finalized=finalized, tour=tour,
               skills_called=list(skills_called), n_evals=costs.count,
               exec_ms=exec_ms, error=err, hang_trace=hang_trace)


def _worker(conn, ctx, limits: Limits):
    """Child process: set limits, build the namespace, serve actions until told to stop."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 2))
        resource.setrlimit(resource.RLIMIT_AS, (limits.mem_bytes, limits.mem_bytes))
    except (ValueError, OSError):
        pass  # limits are best-effort; a missing cap must not break the kernel
    ns, costs, skills_called = _make_namespace(ctx)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        obs = _exec_action(ns, costs, skills_called, msg, limits)
        conn.send(obs)
    conn.close()


class CodeActSession:
    """Stateful sandbox session: agent sends code actions; namespace persists (INV-2).

    A catchable wall-timeout / MemoryError keeps the SAME kernel alive (namespace intact).
    A hard kill (CPU backstop, OOM-abort, crash) is recovered by RESTARTING the kernel with
    a fresh namespace (up to max_restarts), so one runaway action does not end the solve --
    the harness keeps the best tour found so far regardless.
    """

    def __init__(self, ctx: dict, limits: Limits | None = None, max_restarts: int = 4):
        self.ctx = ctx
        self.n = ctx["n"]
        self.limits = limits or Limits()
        self._mpf = mp.get_context("fork")
        self._max_restarts = max_restarts
        self._restarts = 0
        self._proc = None
        self._parent = None
        self._dead = True
        self._start()

    def _start(self) -> None:
        self._parent, child = self._mpf.Pipe()
        self._proc = self._mpf.Process(target=_worker, args=(child, self.ctx, self.limits),
                                       daemon=True)
        self._proc.start()
        child.close()
        self._dead = False

    def _restart(self) -> bool:
        self._kill()
        if self._restarts >= self._max_restarts:
            return False
        self._restarts += 1
        self._start()
        return True

    def act(self, code: str) -> Obs:
        if self._dead and not self._restart():
            return Obs(summary="[session permanently dead: too many kernel crashes]", error="dead")
        try:
            self._parent.send(code)
        except (BrokenPipeError, OSError):
            return self._reset_obs()
        if not self._parent.poll(self.limits.wall_timeout + 5.0):  # kernel stuck
            return self._reset_obs(hard_timeout=True)
        try:
            return self._parent.recv()
        except (EOFError, OSError):                                 # kernel crashed
            return self._reset_obs()

    def _reset_obs(self, hard_timeout: bool = False) -> Obs:
        ok = self._restart()
        if not ok:
            return Obs(summary="[session dead after repeated kernel crashes]", error="dead")
        why = "hard-timed-out" if hard_timeout else "crashed (resource limit / OOM)"
        return Obs(summary=f"[kernel {why}; namespace was RESET -- prior variables are gone, "
                           f"rebuild from n/x/C/prims/skills]", error="kernel reset")

    def _kill(self) -> None:
        self._dead = True
        with contextlib.suppress(Exception):
            if self._proc and self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(0.5)
                if self._proc.is_alive():          # stuck in C (terminate ignored) -> SIGKILL + reap
                    self._proc.kill()
                    self._proc.join(0.5)
        with contextlib.suppress(Exception):
            if self._parent:
                self._parent.close()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._proc and self._proc.is_alive():
                self._parent.send(None)
                self._proc.join(timeout=2.0)
        self._kill()


def run_in_sandbox(code: str, ctx: dict, limits: Limits | None = None, retries: int = 2) -> Result:
    """One-shot: run `code` once, read back FINAL_TOUR (sec 9 judge).

    Retries on a TRANSIENT kernel reset/crash (fork or resource hiccup), which can otherwise
    mark correct, fast code infeasible. A DETERMINISTIC failure (code exception / invalid tour)
    is returned immediately -- not retried.
    """
    last: Result | None = None
    for _ in range(retries + 1):
        sess = CodeActSession(ctx, limits)
        try:
            obs = sess.act(code)
            feasible = obs.tour is not None and is_valid_tour(obs.tour, ctx["n"])
            last = Result(tour=obs.tour, feasible=feasible, n_evals=obs.n_evals,
                          exec_ms=obs.exec_ms, error=obs.error, hang_trace=obs.hang_trace)
        finally:
            sess.close()
        if last.feasible or last.error not in ("kernel reset", "dead"):
            return last
    return last
