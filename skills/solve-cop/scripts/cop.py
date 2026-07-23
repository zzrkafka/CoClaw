"""Small public tool surface for solve-cop.

The Agent chooses actions. This tool only scans families, invokes parsers, runs one
solver trial, and independently validates its solutions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


class CopError(RuntimeError):
    pass


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(skill_root: Path, relative: str, label: str, must_exist: bool = True) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise CopError(f"{label} must be a safe path relative to the Skill root")
    root = skill_root.resolve()
    path = (root / raw).resolve()
    if path != root and root not in path.parents:
        raise CopError(f"{label} escapes the Skill root")
    if must_exist and not path.is_file():
        raise CopError(f"{label} does not exist: {relative}")
    return path


def _manifest(skill_root: Path, family: str) -> dict[str, Any]:
    path = skill_root / "references" / "families" / f"{family}.json"
    try:
        value = _load(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CopError(f"cannot read family manifest {path}: {exc}") from exc
    _validate_manifest(value, family, skill_root)
    return value


def _validate_manifest(value: Any, expected_family: str, skill_root: Path) -> None:
    if not isinstance(value, dict) or value.get("version") != 1:
        raise CopError("family manifest must be a version-1 object")
    if value.get("family") != expected_family:
        raise CopError("family manifest ID must match its filename")
    if value.get("status") not in {"initializing", "active", "disabled"}:
        raise CopError("family status is invalid")
    objective = value.get("objective")
    if not isinstance(objective, dict) or objective.get("sense") not in {"min", "max"}:
        raise CopError("family objective is invalid")
    if not isinstance(value.get("formats"), list) or not value["formats"]:
        raise CopError("family formats must be a nonempty array")
    semantics = value.get("semantics")
    if not isinstance(semantics, dict) or not isinstance(semantics.get("requires"), list) or not isinstance(
        semantics.get("rejects"), list
    ):
        raise CopError("family semantics must declare requires and rejects")
    for field in ("guide", "parser", "validator"):
        relative = value.get(field)
        if not isinstance(relative, str):
            raise CopError(f"family {field} path is missing")
        _asset(skill_root, relative, f"family {field}")


def scan(skill_root: Path) -> dict[str, Any]:
    family_dir = skill_root / "references" / "families"
    families: list[dict[str, Any]] = []
    for path in sorted(family_dir.glob("*.json")):
        problems: list[str] = []
        value: dict[str, Any] | None = None
        try:
            value = _load(path)
            _validate_manifest(value, path.stem, skill_root)
        except (OSError, ValueError, json.JSONDecodeError, CopError) as exc:
            problems.append(str(exc))
        declared = value.get("status") if isinstance(value, dict) else "invalid"
        usable = declared == "active" and not problems
        families.append(
            {
                "family": path.stem,
                "declared_status": declared,
                "usable_by_solve": usable,
                "problems": problems,
            }
        )
    return {"version": 1, "families": families}


def _run_parser(
    manifest: dict[str, Any],
    skill_root: Path,
    input_path: Path,
    output_path: Path,
    *,
    probe: bool,
    instance_id: str | None = None,
    source_path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    parser = _asset(skill_root, manifest["parser"], "family parser")
    command = [sys.executable, str(parser), "--probe" if probe else "--parse", "--input", str(input_path), "--output", str(output_path)]
    if not probe:
        if not instance_id:
            raise CopError("instance_id is required for parsing")
        command.extend(["--instance-id", instance_id])
        if source_path:
            command.extend(["--source-path", source_path])
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


def recognize(skill_root: Path, input_path: Path, allow_initializing: bool) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="coclaw-recognize-") as temporary:
        for row in scan(skill_root)["families"]:
            if row["problems"]:
                continue
            if row["declared_status"] != "active" and not (
                allow_initializing and row["declared_status"] == "initializing"
            ):
                continue
            family = row["family"]
            manifest = _manifest(skill_root, family)
            output = Path(temporary) / f"{family}.json"
            try:
                process = _run_parser(manifest, skill_root, input_path, output, probe=True)
                probe = _load(output) if output.exists() else {
                    "version": 1,
                    "family": family,
                    "status": "error",
                    "format": None,
                    "observations": {},
                    "message": process.stderr.strip() or "parser produced no probe",
                }
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                probe = {
                    "version": 1,
                    "family": family,
                    "status": "error",
                    "format": None,
                    "observations": {},
                    "message": str(exc),
                }
            candidates.append({"family": family, "declared_status": row["declared_status"], "probe": probe})
    matches = [item for item in candidates if item["probe"].get("status") == "compatible"]
    status = "matched" if len(matches) == 1 else "ambiguous" if len(matches) > 1 else "unknown"
    return {
        "version": 1,
        "status": status,
        "family": matches[0]["family"] if len(matches) == 1 else None,
        "matches": matches,
        "probes": candidates,
    }


def parse(
    skill_root: Path,
    family: str,
    input_path: Path,
    instance_id: str,
    output_path: Path,
    source_path: str | None,
) -> dict[str, Any]:
    manifest = _manifest(skill_root, family)
    process = _run_parser(
        manifest,
        skill_root,
        input_path,
        output_path,
        probe=False,
        instance_id=instance_id,
        source_path=source_path,
    )
    if process.returncode != 0:
        detail = process.stderr.strip()
        if output_path.exists():
            try:
                detail = _load(output_path).get("message", detail)
            except Exception:
                pass
        raise CopError(f"parser failed: {detail}")
    value = _load(output_path)
    if not isinstance(value, dict) or value.get("family") != family or value.get("instance_id") != instance_id:
        raise CopError("parser returned an invalid normalized instance")
    return value


def _batch(path: Path) -> dict[str, Any]:
    try:
        value = _load(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CopError(f"cannot read batch manifest: {exc}") from exc
    if not isinstance(value, dict) or value.get("version") != 1:
        raise CopError("batch manifest must be a version-1 object")
    if not isinstance(value.get("run_id"), str) or not isinstance(value.get("family"), str):
        raise CopError("batch manifest run_id or family is invalid")
    limit = value.get("time_limit_sec")
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit <= 0:
        raise CopError("batch time_limit_sec must be positive")
    entries = value.get("instances")
    if not isinstance(entries, list) or not entries:
        raise CopError("batch instances must be a nonempty array")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise CopError("batch instance entry must be an object")
        instance_id = entry.get("id")
        if not isinstance(instance_id, str) or not instance_id or instance_id in seen:
            raise CopError("batch instance IDs must be nonempty and unique")
        seen.add(instance_id)
        if not isinstance(entry.get("path"), str):
            raise CopError(f"instance {instance_id} path is invalid")
        if isinstance(entry.get("seed"), bool) or not isinstance(entry.get("seed"), int) or entry["seed"] < 0:
            raise CopError(f"instance {instance_id} seed is invalid")
        per_limit = entry.get("time_limit_sec")
        if isinstance(per_limit, bool) or not isinstance(per_limit, (int, float)) or per_limit <= 0:
            raise CopError(f"instance {instance_id} time limit is invalid")
    return value


def _relative_instance(manifest_path: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise CopError("instance paths must be safe and relative to the batch manifest")
    base = manifest_path.parent.resolve()
    path = (base / raw).resolve()
    if path != base and base not in path.parents:
        raise CopError("instance path escapes the batch directory")
    if not path.is_file():
        raise CopError(f"normalized instance does not exist: {relative}")
    return path


def _references(path: Path | None, family: str) -> dict[str, Any]:
    if path is None:
        return {}
    value = _load(path)
    if not isinstance(value, dict) or value.get("version") != 1 or value.get("family") != family:
        raise CopError("objective references do not match the run family")
    values = value.get("values")
    if not isinstance(values, dict):
        raise CopError("objective references values must be an object")
    return values


def _gap(value: float, reference: float, sense: str) -> float:
    denominator = max(abs(reference), 1e-12)
    return (
        (value - reference) / denominator * 100.0
        if sense == "min"
        else (reference - value) / denominator * 100.0
    )


def run_trial(
    skill_root: Path,
    manifest_path: Path,
    solver_path: Path,
    trial_dir: Path,
    evaluation_kind: str,
    references_path: Path | None,
    allow_initializing: bool,
    evaluator_root: Path | None = None,
) -> dict[str, Any]:
    batch = _batch(manifest_path)
    family = batch["family"]
    manifest = _manifest(skill_root, family)
    if manifest["status"] != "active" and not (
        allow_initializing and manifest["status"] == "initializing"
    ):
        raise CopError(f"family {family} is not usable for this run")
    if evaluation_kind not in {"cheap", "full"}:
        raise CopError("evaluation kind must be cheap or full")
    if not solver_path.is_file():
        raise CopError("solver does not exist")
    if trial_dir.exists() and any(trial_dir.iterdir()):
        raise CopError("trial directory must be new or empty")
    trial_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir = trial_dir / "solutions"
    validations_dir = trial_dir / "validations"
    solutions_dir.mkdir()
    validations_dir.mkdir()
    frozen_solver = trial_dir / "solver.py"
    shutil.copy2(solver_path, frozen_solver)
    solver_hash = _sha256(frozen_solver)
    stdout_path = trial_dir / "solver.stdout.txt"
    stderr_path = trial_dir / "solver.stderr.txt"
    command = [
        sys.executable,
        str(frozen_solver),
        "--manifest",
        str(manifest_path.resolve()),
        "--output-dir",
        str(solutions_dir.resolve()),
    ]
    started = time.perf_counter()
    returncode: int | None
    timed_out = False
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(batch["time_limit_sec"]) + 5.0,
            cwd=manifest_path.parent,
        )
        returncode = process.returncode
        stdout_path.write_text(process.stdout, encoding="utf-8")
        stderr_path.write_text(process.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        returncode = None
        timed_out = True
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "solver timed out", encoding="utf-8")
    wall_time = time.perf_counter() - started

    evaluator_skill_root = evaluator_root.resolve() if evaluator_root else skill_root
    evaluator_manifest = _manifest(evaluator_skill_root, family)
    validator = _asset(evaluator_skill_root, evaluator_manifest["validator"], "family validator")
    reference_values = _references(references_path, family)
    sense = evaluator_manifest["objective"]["sense"]
    records: list[dict[str, Any]] = []
    for entry in batch["instances"]:
        instance_id = entry["id"]
        instance_path = _relative_instance(manifest_path, entry["path"])
        solution_path = solutions_dir / f"{instance_id}.solution.json"
        validation_path = validations_dir / f"{instance_id}.validation.json"
        if solution_path.is_file():
            validation_process = subprocess.run(
                [
                    sys.executable,
                    str(validator),
                    "--instance",
                    str(instance_path),
                    "--solution",
                    str(solution_path),
                    "--output",
                    str(validation_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if not validation_path.exists():
                validation = {
                    "version": 1,
                    "instance_id": instance_id,
                    "status": "error",
                    "feasible": None,
                    "violations": [validation_process.stderr.strip() or "validator produced no output"],
                }
                _write(validation_path, validation)
            else:
                validation = _load(validation_path)
        else:
            validation = {
                "version": 1,
                "instance_id": instance_id,
                "status": "invalid",
                "feasible": False,
                "violations": ["solver did not publish a solution file"],
            }
            _write(validation_path, validation)
        objective = validation.get("objective") if isinstance(validation, dict) else None
        record: dict[str, Any] = {
            "id": instance_id,
            "status": validation.get("status", "error") if isinstance(validation, dict) else "error",
            "solution": str(solution_path.relative_to(trial_dir)).replace("\\", "/") if solution_path.exists() else None,
            "validation": str(validation_path.relative_to(trial_dir)).replace("\\", "/"),
            "objective": objective,
        }
        reference = reference_values.get(instance_id)
        if isinstance(objective, (int, float)) and isinstance(reference, dict) and isinstance(
            reference.get("value"), (int, float)
        ):
            record["reference"] = {"kind": reference.get("kind"), "value": float(reference["value"])}
            record["gap_percent"] = _gap(float(objective), float(reference["value"]), sense)
        records.append(record)

    valid_records = [item for item in records if item["status"] == "valid"]
    gaps = [float(item["gap_percent"]) for item in valid_records if "gap_percent" in item]
    status = (
        "valid"
        if returncode == 0 and len(valid_records) == len(records)
        else "partial"
        if valid_records
        else "error"
    )
    result = {
        "version": 1,
        "trial_id": trial_dir.name,
        "family": family,
        "objective_sense": sense,
        "evaluation_kind": evaluation_kind,
        "status": status,
        "batch": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
            "time_limit_sec": float(batch["time_limit_sec"]),
            "instances": [
                {
                    "id": item["id"],
                    "seed": item["seed"],
                    "time_limit_sec": float(item["time_limit_sec"]),
                }
                for item in batch["instances"]
            ],
        },
        "evaluator": {
            "validator_sha256": _sha256(validator),
        },
        "solver": {
            "path": "solver.py",
            "sha256": solver_hash,
            "returncode": returncode,
            "wall_time_sec": wall_time,
            "timed_out": timed_out,
        },
        "instances": records,
        "summary": {
            "total": len(records),
            "valid": len(valid_records),
            "all_valid": len(valid_records) == len(records),
            "mean_gap_percent": sum(gaps) / len(gaps) if gaps else None,
            "max_gap_percent": max(gaps) if gaps else None,
        },
    }
    _write(trial_dir / "run-result.json", result)
    return result


def _emit(value: Any, output: Path | None) -> None:
    if output:
        _write(output, value)
    else:
        json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Minimal deterministic tools for solve-cop")
    sub = cli.add_subparsers(dest="command", required=True)

    catalog_cli = sub.add_parser("catalog")
    catalog_cli.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[1])
    catalog_cli.add_argument("--output", type=Path)

    recognize_cli = sub.add_parser("recognize")
    recognize_cli.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[1])
    recognize_cli.add_argument("--input", type=Path, required=True)
    recognize_cli.add_argument("--allow-initializing", action="store_true")
    recognize_cli.add_argument("--output", type=Path)

    parse_cli = sub.add_parser("parse")
    parse_cli.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[1])
    parse_cli.add_argument("--family", required=True)
    parse_cli.add_argument("--input", type=Path, required=True)
    parse_cli.add_argument("--instance-id", required=True)
    parse_cli.add_argument("--source-path")
    parse_cli.add_argument("--output", type=Path, required=True)

    run_cli = sub.add_parser("run")
    run_cli.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[1])
    run_cli.add_argument("--manifest", type=Path, required=True)
    run_cli.add_argument("--solver", type=Path, required=True)
    run_cli.add_argument("--trial-dir", type=Path, required=True)
    run_cli.add_argument("--kind", choices=["cheap", "full"], default="cheap")
    run_cli.add_argument("--references", type=Path)
    run_cli.add_argument("--allow-initializing", action="store_true")
    run_cli.add_argument("--evaluator-root", type=Path)

    args = cli.parse_args(argv)
    try:
        if args.command == "catalog":
            result = scan(args.skill_root.resolve())
            _emit(result, args.output)
        elif args.command == "recognize":
            result = recognize(args.skill_root.resolve(), args.input.resolve(), args.allow_initializing)
            _emit(result, args.output)
            return 0 if result["status"] == "matched" else 2
        elif args.command == "parse":
            parse(
                args.skill_root.resolve(),
                args.family,
                args.input.resolve(),
                args.instance_id,
                args.output.resolve(),
                args.source_path,
            )
        else:
            result = run_trial(
                args.skill_root.resolve(),
                args.manifest.resolve(),
                args.solver.resolve(),
                args.trial_dir.resolve(),
                args.kind,
                args.references.resolve() if args.references else None,
                args.allow_initializing,
                args.evaluator_root.resolve() if args.evaluator_root else None,
            )
            _emit(result, None)
            return 0 if result["status"] == "valid" else 2
        return 0
    except (CopError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
