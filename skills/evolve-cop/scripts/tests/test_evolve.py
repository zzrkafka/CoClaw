from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = SKILL_ROOT.parents[1]
SOLVE_ROOT = REPO_ROOT / "skills" / "solve-cop"
sys.path.insert(0, str(SCRIPT_ROOT))

import evolve  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def run_result(
    path: Path,
    objectives: dict[str, float],
    *,
    validator_sha256: str = "v" * 64,
    wall_time_sec: float = 1,
) -> None:
    write_json(
        path,
        {
            "version": 1,
            "trial_id": path.stem,
            "family": "tsp",
            "objective_sense": "min",
            "evaluation_kind": "full",
            "status": "valid",
            "batch": {"sha256": "b" * 64},
            "evaluator": {"validator_sha256": validator_sha256},
            "solver": {
                "path": "solver.py",
                "sha256": "0" * 64,
                "returncode": 0,
                "wall_time_sec": wall_time_sec,
            },
            "instances": [
                {"id": key, "status": "valid", "objective": value}
                for key, value in sorted(objectives.items())
            ],
            "summary": {"all_valid": True},
        },
    )


class EvolveToolTests(unittest.TestCase):
    def test_snapshot_stage_request_compare_and_accept(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-solve"
            shutil.copytree(SOLVE_ROOT / "references" / "families", source / "references" / "families")
            shutil.copytree(
                SOLVE_ROOT / "scripts" / "families" / "tsp",
                source / "scripts" / "families" / "tsp",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            snapshot_root = root / "snapshot"
            candidate_root = root / "candidate"
            evolve.snapshot(source, "tsp", snapshot_root)
            evolve.stage(snapshot_root, candidate_root)
            candidate_skill = candidate_root / "solve-cop"
            guide = candidate_skill / "references" / "families" / "tsp.md"
            guide.write_text(guide.read_text(encoding="utf-8") + "\nCandidate marker.\n", encoding="utf-8")

            batch = root / "batch.json"
            write_json(root / "a.json", {"version": 1, "family": "tsp", "data": {}})
            write_json(
                batch,
                {
                    "version": 1,
                    "run_id": "batch",
                    "family": "tsp",
                    "time_limit_sec": 10,
                    "instances": [{"id": "a", "path": "a.json", "seed": 0, "time_limit_sec": 5}],
                },
            )
            request_path = root / "request.json"
            request = evolve.request(
                "candidate",
                "tsp",
                candidate_skill,
                batch,
                10,
                1000,
                ["short probe"],
                request_path,
            )
            self.assertTrue(request["blind"])

            champion = root / "champion.json"
            candidate = root / "candidate.json"
            run_result(champion, {"a": 100, "b": 200})
            run_result(candidate, {"a": 90, "b": 190}, wall_time_sec=0.5)
            comparison_path = root / "comparison.json"
            comparison = evolve.compare(champion, candidate, comparison_path)
            self.assertTrue(comparison["eligible"])
            self.assertEqual(2, comparison["summary"]["wins"])
            self.assertEqual(0.5, comparison["summary"]["wall_time_ratio"])

            decision_path = root / "decision.json"
            evolve.decide(
                source,
                "tsp",
                candidate_skill,
                comparison_path,
                "accept",
                "matched improvement",
                decision_path,
            )
            self.assertIn(
                "Candidate marker.",
                (source / "references" / "families" / "tsp.md").read_text(encoding="utf-8"),
            )

    def test_request_rejects_reference_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            family_root = root / "family"
            family_root.mkdir()
            manifest = root / "batch.json"
            write_json(manifest, {"family": "tsp", "oracle": {"value": 1}})
            with self.assertRaises(evolve.EvolutionError):
                evolve.request(
                    "candidate",
                    "tsp",
                    family_root,
                    manifest,
                    10,
                    100,
                    [],
                    root / "request.json",
                )

    def test_compare_rejects_different_evaluators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            champion = root / "champion.json"
            candidate = root / "candidate.json"
            run_result(champion, {"a": 100})
            run_result(candidate, {"a": 90}, validator_sha256="x" * 64)
            with self.assertRaises(evolve.EvolutionError):
                evolve.compare(champion, candidate, root / "comparison.json")


if __name__ == "__main__":
    unittest.main()
