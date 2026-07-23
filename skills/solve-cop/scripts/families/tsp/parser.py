"""Parse supported symmetric TSP inputs with a small stable interface."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


FAMILY = "tsp"
FORMAT_JSON = "coclaw-tsp-json-v1"
FORMAT_TSPLIB = "tsplib-tsp-euc2d-v1"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TspInputError(ValueError):
    def __init__(self, message: str, status: str = "ambiguous"):
        super().__init__(message)
        self.status = status


def parse_input(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        try:
            raw = json.loads(text, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise TspInputError(f"invalid JSON: {exc}") from exc
        return _parse_json(raw)
    return _parse_tsplib(text)


def normalize(path: Path, instance_id: str, source_path: str | None = None) -> dict[str, Any]:
    if not ID_RE.fullmatch(instance_id):
        raise TspInputError("instance_id is invalid")
    data, format_id, metadata = parse_input(path)
    metadata = dict(metadata)
    metadata["format"] = format_id
    if source_path:
        metadata["source_path"] = source_path
    return {
        "version": 1,
        "family": FAMILY,
        "instance_id": instance_id,
        "objective": {"sense": "min", "name": "tour_length"},
        "data": data,
        "metadata": metadata,
    }


def probe(path: Path) -> dict[str, Any]:
    try:
        data, format_id, metadata = parse_input(path)
    except OSError as exc:
        return _probe("error", None, {}, str(exc))
    except TspInputError as exc:
        return _probe(exc.status, None, {}, str(exc))
    observations = {
        "dimension": data["dimension"],
        "representation": data["representation"],
        "symmetric": True,
        "name": metadata.get("name"),
    }
    return _probe("compatible", format_id, observations, "supported symmetric TSP")


def _probe(
    status: str, format_id: str | None, observations: dict[str, Any], message: str
) -> dict[str, Any]:
    return {
        "version": 1,
        "family": FAMILY,
        "status": status,
        "format": format_id,
        "observations": observations,
        "message": message,
    }


def _parse_json(raw: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise TspInputError("JSON root must be an object")
    problem_type = str(raw.get("type", "")).upper()
    if problem_type and problem_type != "TSP":
        raise TspInputError(f"unsupported problem type: {problem_type}", "incompatible")
    edge_type = str(raw.get("edge_weight_type", "")).upper()
    name = str(raw.get("name", "unnamed"))

    if edge_type == "EUC_2D":
        nodes = raw.get("nodes")
        if not isinstance(nodes, list) or len(nodes) < 3:
            raise TspInputError("EUC_2D JSON requires at least three nodes")
        node_ids: list[str] = []
        coordinates: list[list[float]] = []
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise TspInputError(f"node {index} must be an object")
            node_id = str(node.get("id", ""))
            if not node_id or node_id in node_ids:
                raise TspInputError("node IDs must be nonempty and unique")
            x = _number(node.get("x"), f"node {node_id} x")
            y = _number(node.get("y"), f"node {node_id} y")
            node_ids.append(node_id)
            coordinates.append([x, y])
        data = {
            "representation": "coordinates_2d",
            "edge_weight_type": "EUC_2D",
            "dimension": len(node_ids),
            "node_ids": node_ids,
            "coordinates": coordinates,
        }
    elif edge_type == "EXPLICIT":
        raw_ids = raw.get("node_ids")
        matrix = raw.get("distance_matrix")
        if not isinstance(raw_ids, list) or len(raw_ids) < 3:
            raise TspInputError("EXPLICIT JSON requires at least three node_ids")
        node_ids = [str(value) for value in raw_ids]
        if any(not value for value in node_ids) or len(set(node_ids)) != len(node_ids):
            raise TspInputError("node_ids must be nonempty and unique")
        data = {
            "representation": "distance_matrix",
            "edge_weight_type": "EXPLICIT",
            "dimension": len(node_ids),
            "node_ids": node_ids,
            "distance_matrix": _matrix(matrix, len(node_ids)),
        }
    else:
        raise TspInputError(f"unsupported edge_weight_type: {edge_type or 'missing'}", "incompatible")

    metadata: dict[str, Any] = {"name": name}
    if "comment" in raw:
        metadata["comment"] = str(raw["comment"])[:1000]
    return data, FORMAT_JSON, metadata


def _parse_tsplib(text: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    headers: dict[str, str] = {}
    coordinates: list[tuple[str, float, float]] = []
    in_coordinates = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper == "NODE_COORD_SECTION":
            in_coordinates = True
            continue
        if not in_coordinates:
            if ":" in line:
                key, value = line.split(":", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                key, value = parts
            headers[key.strip().upper()] = value.strip()
            continue
        parts = line.split()
        if len(parts) < 3:
            raise TspInputError(f"invalid coordinate row: {line}")
        node_id = parts[0]
        coordinates.append(
            (node_id, _number(parts[1], f"node {node_id} x"), _number(parts[2], f"node {node_id} y"))
        )

    problem_type = headers.get("TYPE", "").upper()
    if problem_type != "TSP":
        raise TspInputError(f"unsupported TSPLIB TYPE: {problem_type or 'missing'}", "incompatible")
    edge_type = headers.get("EDGE_WEIGHT_TYPE", "").upper()
    if edge_type != "EUC_2D":
        raise TspInputError(f"unsupported TSPLIB EDGE_WEIGHT_TYPE: {edge_type or 'missing'}", "incompatible")
    try:
        dimension = int(headers.get("DIMENSION", ""))
    except ValueError as exc:
        raise TspInputError("invalid or missing TSPLIB DIMENSION") from exc
    if dimension < 3 or len(coordinates) != dimension:
        raise TspInputError("TSPLIB coordinate count does not match DIMENSION")
    node_ids = [row[0] for row in coordinates]
    if len(set(node_ids)) != dimension:
        raise TspInputError("TSPLIB node IDs must be unique")
    data = {
        "representation": "coordinates_2d",
        "edge_weight_type": "EUC_2D",
        "dimension": dimension,
        "node_ids": node_ids,
        "coordinates": [[row[1], row[2]] for row in coordinates],
    }
    metadata = {"name": headers.get("NAME", "unnamed")}
    if "COMMENT" in headers:
        metadata["comment"] = headers["COMMENT"][:1000]
    return data, FORMAT_TSPLIB, metadata


def _matrix(raw: Any, dimension: int) -> list[list[float]]:
    if not isinstance(raw, list) or len(raw) != dimension:
        raise TspInputError("distance_matrix size does not match node_ids")
    matrix: list[list[float]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != dimension:
            raise TspInputError("distance_matrix must be square")
        values = [_number(value, f"distance[{i}]") for value in row]
        if any(value < 0 for value in values):
            raise TspInputError("distances must be nonnegative")
        matrix.append(values)
    for i in range(dimension):
        if abs(matrix[i][i]) > 1e-9:
            raise TspInputError("distance matrix diagonal must be zero")
        for j in range(i + 1, dimension):
            if not math.isclose(matrix[i][j], matrix[j][i], rel_tol=1e-12, abs_tol=1e-9):
                raise TspInputError("distance matrix is asymmetric", "incompatible")
    return matrix


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TspInputError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TspInputError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise TspInputError(f"{label} must be finite")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _write_json(path: Path, value: Any) -> None:
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
    valid = [root / "valid" / "tiny5.json", root / "valid" / "explicit4.json", root / "valid" / "triangle.tsp"]
    invalid = [root / "invalid" / "asymmetric.json", root / "invalid" / "missing-node.tsp"]
    failures: list[str] = []
    for path in valid:
        first = normalize(path, path.stem)
        second = normalize(path, path.stem)
        if first != second or probe(path)["status"] != "compatible":
            failures.append(f"valid fixture failed: {path.name}")
    for path in invalid:
        if probe(path)["status"] == "compatible":
            failures.append(f"invalid fixture accepted: {path.name}")
    return {"version": 1, "family": FAMILY, "status": "pass" if not failures else "fail", "failures": failures}


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser()
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--parse", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    cli.add_argument("--input", type=Path)
    cli.add_argument("--instance-id")
    cli.add_argument("--source-path")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(argv)
    try:
        if args.self_test:
            result = self_test()
        elif args.probe:
            if args.input is None:
                raise TspInputError("--input is required")
            result = probe(args.input)
        else:
            if args.input is None or not args.instance_id:
                raise TspInputError("--input and --instance-id are required")
            result = normalize(args.input, args.instance_id, args.source_path)
        _write_json(args.output, result)
        return 0 if result.get("status") != "fail" else 1
    except (OSError, TspInputError) as exc:
        _write_json(args.output, _probe("error", None, {}, str(exc)) if args.probe else {"version": 1, "family": FAMILY, "status": "error", "message": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
