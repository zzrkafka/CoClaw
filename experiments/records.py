"""JSONL run records (spec section 12). One file per record type under runs/<run_id>/."""
from __future__ import annotations

import json
from pathlib import Path

_FILES = {
    "run_meta": "run_meta.jsonl",
    "instance": "instance_record.jsonl",
    "solve": "solve_record.jsonl",
    "evolve": "evolve_record.jsonl",
    "skill": "skill_snapshot.jsonl",
    "metric": "metric_record.jsonl",
    "discrimination": "discrimination_record.jsonl",   # §4 persisted gap-matrix discrimination
}


class RunRecorder:
    def __init__(self, run_id: str, root: str | Path = "runs"):
        self.run_id = run_id
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, record: dict) -> None:
        rec = {"type": kind, **record}
        with (self.dir / _FILES[kind]).open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def solve_record(self, arm: str, sr, round_idx: int, frozen: bool) -> None:
        """Trajectory record -- store code_sha (not full code) to keep files small."""
        actions = [{
            "step": t["step"], "think": t["think"][:400], "code_sha": t["code_sha"],
            "observation": (t["observation"] or "")[:600], "tour": t["tour"], "gap": t["gap"],
            "skills_called": t["skills_called"], "n_evals": t["n_evals"], "exec_ms": t["exec_ms"],
        } for t in sr.trajectory]
        self.emit("solve", {
            "run_id": self.run_id, "arm": arm, "instance_id": sr.instance_id,
            "round_idx": round_idx, "frozen": frozen, "actions": actions,
            "best_gap": sr.best_gap, "best_length": sr.best_length,
            "best_tour": sr.best_tour, "total_evals": sr.total_evals,
            "finalized": sr.finalized, "n_steps": sr.n_steps,
        })

    def skill_snapshot(self, lib) -> None:
        for st in lib.skill_states():
            self.emit("skill", {"run_id": self.run_id, "library_version": lib.version, **st})
