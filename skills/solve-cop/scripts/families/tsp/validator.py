"""Independently validate a simple symmetric TSP solution."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


FAMILY = "tsp"


class ValidationError(ValueError):
    pass


def validate(instance_path: Path, solution_path: Path) -> dict[str, Any]:
    try:
        instance = _load(instance_path)
        model = _model(instance)
    except (OSError, ValueError) as exc:
        return _result("unknown", "error", None, None, [f"invalid normalized instance: {exc}"])

    instance_id = model["instance_id"]
    try:
        solution = _load(solution_path)
    except (OSError, ValueError) as exc:
        return _result(instance_id, "invalid", False, None, [f"invalid solution JSON: {exc}"])
    if not isinstance(solution, dict):
        return _result(instance_id, "invalid", False, None, ["solution root must be an object"])

    violations: list[str] = []
    if solution.get("version") != 1:
        violations.append("solution version must be 1")
    if solution.get("instance_id") != instance_id:
        violations.append("solution instance_id does not match")
    status = solution.get("status")
    if status == "no_candidate" and not violations:
        return _result(instance_id, "no_candidate", None, None, [])
    if status in {"error", "interrupted"} and not violations:
        return _result(instance_id, "error", None, None, [str(solution.get("message", status))])
    if status != "candidate":
        violations.append("status must be candidate, no_candidate, error, or interrupted")

    payload = solution.get("solution")
    tour = payload.get("tour") if isinstance(payload, dict) else None
    if not isinstance(tour, list):
        violations.append("solution.tour must be an array")
        return _result(instance_id, "invalid", False, None, violations)
    tour = [str(node) for node in tour]
    if len(tour) == len(model["node_ids"]) + 1 and tour[0] == tour[-1]:
        tour = tour[:-1]
    if len(tour) != len(model["node_ids"]):
        violations.append("tour length does not match the number of nodes")
    if len(set(tour)) != len(tour):
        violations.append("tour contains duplicate nodes")
    missing = sorted(set(model["node_ids"]) - set(tour))
    extra = sorted(set(tour) - set(model["node_ids"]))
    if missing:
        violations.append(f"tour misses nodes: {missing}")
    if extra:
        violations.append(f"tour contains unknown nodes: {extra}")
    if violations:
        return _result(instance_id, "invalid", False, None, violations)

    objective = _tour_cost(tour, model)
    reported = solution.get("objective")
    reported_number: float | None = None
    if reported is not None:
        if isinstance(reported, bool) or not isinstance(reported, (int, float)) or not math.isfinite(float(reported)):
            violations.append("reported objective must be finite")
        else:
            reported_number = float(reported)
            tolerance = 1e-8 * max(1.0, abs(objective))
            if abs(reported_number - objective) > tolerance:
                violations.append(f"reported objective {reported_number} does not match {objective}")
    if violations:
        return _result(instance_id, "invalid", False, objective, violations, reported_number)
    return _result(instance_id, "valid", True, objective, [], reported_number)


def _result(
    instance_id: str,
    status: str,
    feasible: bool | None,
    objective: float | None,
    violations: list[str],
    reported: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": 1,
        "instance_id": instance_id,
        "status": status,
        "feasible": feasible,
        "violations": violations,
    }
    if objective is not None:
        result["objective"] = objective
    if reported is not None:
        result["reported_objective"] = reported
    elif status in {"valid", "invalid"}:
        result["reported_objective"] = None
    return result


def _model(instance: Any) -> dict[str, Any]:
    if not isinstance(instance, dict) or instance.get("version") != 1 or instance.get("family") != FAMILY:
        raise ValidationError("expected version-1 TSP normalized instance")
    instance_id = instance.get("instance_id")
    data = instance.get("data")
    if not isinstance(instance_id, str) or not isinstance(data, dict):
        raise ValidationError("instance_id or data is invalid")
    node_ids = data.get("node_ids")
    if not isinstance(node_ids, list) or len(node_ids) < 3:
        raise ValidationError("node_ids are invalid")
    node_ids = [str(value) for value in node_ids]
    if len(set(node_ids)) != len(node_ids):
        raise ValidationError("node_ids are not unique")
    representation = data.get("representation")
    model = {"instance_id": instance_id, "node_ids": node_ids, "representation": representation}
    if representation == "coordinates_2d":
        coordinates = data.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != len(node_ids):
            raise ValidationError("coordinates are invalid")
        normalized: list[tuple[float, float]] = []
        for row in coordinates:
            if not isinstance(row, list) or len(row) != 2:
                raise ValidationError("coordinate row is invalid")
            x, y = float(row[0]), float(row[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValidationError("coordinate is not finite")
            normalized.append((x, y))
        model["coordinates"] = normalized
    elif representation == "distance_matrix":
        matrix = data.get("distance_matrix")
        if not isinstance(matrix, list) or len(matrix) != len(node_ids):
            raise ValidationError("distance matrix is invalid")
        model["matrix"] = [[float(value) for value in row] for row in matrix]
    else:
        raise ValidationError("unsupported representation")
    return model


def _tour_cost(tour: list[str], model: dict[str, Any]) -> float:
    positions = {node: index for index, node in enumerate(model["node_ids"])}
    total = 0.0
    for index, source in enumerate(tour):
        target = tour[(index + 1) % len(tour)]
        i, j = positions[source], positions[target]
        if model["representation"] == "coordinates_2d":
            x1, y1 = model["coordinates"][i]
            x2, y2 = model["coordinates"][j]
            total += int(math.hypot(x1 - x2, y1 - y2) + 0.5)
        else:
            total += model["matrix"][i][j]
    return total


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def self_test() -> dict[str, Any]:
    root = Path(__file__).resolve().parent / "tests" / "fixtures"
    cases = {
        "tiny5-valid.solution.json": "valid",
        "explicit4-valid.solution.json": "valid",
        "tiny5-duplicate.solution.json": "invalid",
        "tiny5-format-error.solution.json": "invalid",
        "tiny5-objective-mismatch.solution.json": "invalid",
        "triangle-no-candidate.solution.json": "no_candidate",
    }
    instances = {
        "tiny5": root / "normalized" / "tiny5.instance.json",
        "explicit4": root / "normalized" / "explicit4.instance.json",
        "triangle": root / "normalized" / "triangle.instance.json",
    }
    failures: list[str] = []
    for filename, expected in cases.items():
        key = "explicit4" if filename.startswith("explicit4") else "triangle" if filename.startswith("triangle") else "tiny5"
        observed = validate(instances[key], root / "solutions" / filename)["status"]
        if observed != expected:
            failures.append(f"{filename}: expected {expected}, got {observed}")
    return {"version": 1, "family": FAMILY, "status": "pass" if not failures else "fail", "failures": failures}


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--self-test", action="store_true")
    cli.add_argument("--instance", type=Path)
    cli.add_argument("--solution", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(argv)
    if args.self_test:
        result = self_test()
    elif args.instance is None or args.solution is None:
        result = _result("unknown", "error", None, None, ["--instance and --solution are required"])
    else:
        result = validate(args.instance, args.solution)
    _write(args.output, result)
    return 0 if result.get("status") in {"pass", "valid", "no_candidate"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
