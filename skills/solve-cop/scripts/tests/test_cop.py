from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT))

import cop  # noqa: E402


class CopToolTests(unittest.TestCase):
    def test_catalog_and_recognition_respect_initializing_status(self) -> None:
        catalog = cop.scan(SKILL_ROOT)
        tsp = next(item for item in catalog["families"] if item["family"] == "tsp")
        self.assertFalse(tsp["usable_by_solve"])
        fixture = (
            SKILL_ROOT
            / "scripts"
            / "families"
            / "tsp"
            / "tests"
            / "fixtures"
            / "valid"
            / "triangle.tsp"
        )
        ordinary = cop.recognize(SKILL_ROOT, fixture, allow_initializing=False)
        evolution = cop.recognize(SKILL_ROOT, fixture, allow_initializing=True)
        self.assertEqual("unknown", ordinary["status"])
        self.assertEqual("matched", evolution["status"])
        self.assertEqual("tsp", evolution["family"])

    def test_parse_and_run_one_batch_with_independent_validation(self) -> None:
        source = (
            SKILL_ROOT
            / "scripts"
            / "families"
            / "tsp"
            / "tests"
            / "fixtures"
            / "valid"
            / "tiny5.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instances = root / "instances"
            instances.mkdir()
            normalized = instances / "tiny5.json"
            cop.parse(SKILL_ROOT, "tsp", source, "tiny5", normalized, "tiny5.json")
            batch = {
                "version": 1,
                "run_id": "test",
                "family": "tsp",
                "time_limit_sec": 5,
                "instances": [
                    {"id": "tiny5", "path": "instances/tiny5.json", "seed": 0, "time_limit_sec": 2}
                ],
            }
            manifest = root / "batch.json"
            manifest.write_text(json.dumps(batch), encoding="utf-8")
            solver = root / "solver.py"
            solver.write_text(
                textwrap.dedent(
                    """
                    import argparse, json
                    from pathlib import Path

                    p = argparse.ArgumentParser()
                    p.add_argument("--manifest", type=Path, required=True)
                    p.add_argument("--output-dir", type=Path, required=True)
                    a = p.parse_args()
                    batch = json.loads(a.manifest.read_text())
                    a.output_dir.mkdir(parents=True, exist_ok=True)
                    for entry in batch["instances"]:
                        instance = json.loads((a.manifest.parent / entry["path"]).read_text())
                        tour = instance["data"]["node_ids"]
                        result = {
                            "version": 1,
                            "instance_id": entry["id"],
                            "status": "candidate",
                            "solution": {"tour": tour},
                            "telemetry": {}
                        }
                        (a.output_dir / f'{entry["id"]}.solution.json').write_text(json.dumps(result))
                    """
                ),
                encoding="utf-8",
            )
            references = root / "references.json"
            references.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "family": "tsp",
                        "values": {"tiny5": {"kind": "optimum", "value": 44}},
                    }
                ),
                encoding="utf-8",
            )
            result = cop.run_trial(
                SKILL_ROOT,
                manifest,
                solver,
                root / "trial",
                "cheap",
                references,
                allow_initializing=True,
            )
            self.assertEqual("valid", result["status"])
            self.assertEqual(44.0, result["instances"][0]["objective"])
            self.assertEqual(0.0, result["instances"][0]["gap_percent"])

    def test_all_contract_files_are_valid_json(self) -> None:
        contract_root = SKILL_ROOT / "references" / "contracts" / "v1"
        schemas = list(contract_root.glob("*.schema.json"))
        self.assertEqual(9, len(schemas))
        for path in schemas + list((contract_root / "examples").glob("*.json")):
            with self.subTest(path=path.name):
                self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)


if __name__ == "__main__":
    unittest.main()
