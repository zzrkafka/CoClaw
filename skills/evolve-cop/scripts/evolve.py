"""Minimal deterministic tools for evolve-cop.

The Agent chooses hypotheses and actions. This tool only preserves family bytes,
creates blinded requests, compares matched run results, and applies a decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import uuid


class EvolutionError(RuntimeError):
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


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(value)
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _family_relative_paths(solve_root: Path, family: str) -> list[Path]:
    paths = [
        Path("references") / "families" / f"{family}.json",
        Path("references") / "families" / f"{family}.md",
    ]
    family_scripts = solve_root / "scripts" / "families" / family
    if not family_scripts.is_dir():
        raise EvolutionError(f"family script directory does not exist: {family_scripts}")
    for path in sorted(family_scripts.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        paths.append(path.relative_to(solve_root))
    for relative in paths[:2]:
        if not (solve_root / relative).is_file():
            raise EvolutionError(f"family asset does not exist: {relative}")
    return paths


def _copy_family(source_root: Path, destination_root: Path, family: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for relative in _family_relative_paths(source_root, family):
        source = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append({"path": relative.as_posix(), "sha256": _sha256(destination)})
    return records


def snapshot(solve_root: Path, family: str, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise EvolutionError("snapshot output must be new or empty")
    skill_copy = output / "solve-cop"
    skill_copy.mkdir(parents=True, exist_ok=True)
    files = _copy_family(solve_root.resolve(), skill_copy, family)
    record = {
        "version": 1,
        "family": family,
        "created_at": _utc_now(),
        "family_root": "solve-cop",
        "files": files,
    }
    _write(output / "snapshot.json", record)
    return record


def _verify_snapshot(snapshot_root: Path) -> dict[str, Any]:
    record = _load(snapshot_root / "snapshot.json")
    if not isinstance(record, dict) or record.get("version") != 1:
        raise EvolutionError("snapshot record is invalid")
    family_root = snapshot_root / str(record.get("family_root", ""))
    if not family_root.is_dir():
        raise EvolutionError("snapshot family root is missing")
    for item in record.get("files", []):
        if not isinstance(item, dict):
            raise EvolutionError("snapshot file record is invalid")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise EvolutionError("snapshot file path is unsafe")
        path = family_root / relative
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise EvolutionError(f"snapshot file changed or is missing: {relative}")
    return record


def stage(snapshot_root: Path, output: Path) -> dict[str, Any]:
    record = _verify_snapshot(snapshot_root)
    if output.exists() and any(output.iterdir()):
        raise EvolutionError("candidate output must be new or empty")
    destination = output / "solve-cop"
    source = snapshot_root / record["family_root"]
    for item in record["files"]:
        relative = Path(item["path"])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    candidate = {
        "version": 1,
        "family": record["family"],
        "created_at": _utc_now(),
        "base_snapshot": str((snapshot_root / "snapshot.json").resolve()),
        "family_root": "solve-cop",
    }
    _write(output / "candidate.json", candidate)
    return candidate


def _contains_forbidden_reference(value: Any) -> str | None:
    forbidden = {"oracle", "optimum", "bks", "gap", "reference", "threshold"}
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(word in lowered for word in forbidden):
                return str(key)
            found = _contains_forbidden_reference(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _contains_forbidden_reference(child)
            if found:
                return found
    return None


def request(
    side: str,
    family: str,
    family_root: Path,
    manifest: Path,
    time_sec: float,
    tokens: int,
    requested_evidence: list[str],
    output: Path,
    evaluator_root: Path | None = None,
) -> dict[str, Any]:
    if side not in {"champion", "candidate"}:
        raise EvolutionError("side must be champion or candidate")
    if not family_root.is_dir():
        raise EvolutionError("family root does not exist")
    batch = _load(manifest)
    if not isinstance(batch, dict) or batch.get("family") != family:
        raise EvolutionError("batch manifest does not match the requested family")
    forbidden = _contains_forbidden_reference(batch)
    if forbidden:
        raise EvolutionError(f"blinded batch contains forbidden reference field: {forbidden}")
    instances = batch.get("instances")
    if not isinstance(instances, list) or not instances:
        raise EvolutionError("blinded batch must contain instances")
    batch_root = manifest.parent.resolve()
    for entry in instances:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise EvolutionError("blinded batch instance path is invalid")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise EvolutionError("blinded batch instance paths must be safe and relative")
        instance_path = (batch_root / relative).resolve()
        if batch_root not in instance_path.parents or not instance_path.is_file():
            raise EvolutionError(f"blinded instance is missing or escapes the batch: {relative}")
        forbidden = _contains_forbidden_reference(_load(instance_path))
        if forbidden:
            raise EvolutionError(
                f"blinded instance {entry.get('id', relative)} contains forbidden reference field: {forbidden}"
            )
    if time_sec <= 0 or tokens <= 0:
        raise EvolutionError("time and token reservations must be positive")
    value = {
        "version": 1,
        "id": output.stem,
        "side": side,
        "family": family,
        "family_root": str(family_root.resolve()),
        "evaluator_root": str((evaluator_root or family_root).resolve()),
        "manifest": str(manifest.resolve()),
        "budget": {"time_sec": float(time_sec), "tokens": int(tokens)},
        "requested_evidence": requested_evidence,
        "blind": True,
    }
    _write(output, value)
    return value


def _run_result(path: Path) -> dict[str, Any]:
    value = _load(path)
    if not isinstance(value, dict) or value.get("version") != 1:
        raise EvolutionError(f"invalid run result: {path}")
    if value.get("objective_sense") not in {"min", "max"} or not isinstance(value.get("instances"), list):
        raise EvolutionError(f"run result lacks objective sense or instances: {path}")
    return value


def _reference_values(path: Path | None, family: str) -> dict[str, Any]:
    if path is None:
        return {}
    value = _load(path)
    if not isinstance(value, dict) or value.get("version") != 1 or value.get("family") != family:
        raise EvolutionError("objective references do not match the compared family")
    values = value.get("values")
    if not isinstance(values, dict):
        raise EvolutionError("objective references values must be an object")
    return values


def _gap(value: float, reference: float, sense: str) -> float:
    denominator = max(abs(reference), 1e-12)
    return (
        (value - reference) / denominator * 100.0
        if sense == "min"
        else (reference - value) / denominator * 100.0
    )


def compare(
    champion_path: Path,
    candidate_path: Path,
    output: Path,
    references_path: Path | None = None,
) -> dict[str, Any]:
    champion = _run_result(champion_path)
    candidate = _run_result(candidate_path)
    if champion.get("family") != candidate.get("family"):
        raise EvolutionError("run results use different families")
    if champion.get("objective_sense") != candidate.get("objective_sense"):
        raise EvolutionError("run results use different objective senses")
    if champion.get("evaluation_kind") != candidate.get("evaluation_kind"):
        raise EvolutionError("run results use different evaluation kinds")
    if champion.get("batch", {}).get("sha256") != candidate.get("batch", {}).get("sha256"):
        raise EvolutionError("run results use different batch manifests")
    if champion.get("evaluator", {}).get("validator_sha256") != candidate.get("evaluator", {}).get(
        "validator_sha256"
    ):
        raise EvolutionError("run results use different validators")
    family = champion["family"]
    sense = champion["objective_sense"]
    champion_map = {item.get("id"): item for item in champion["instances"] if isinstance(item, dict)}
    candidate_map = {item.get("id"): item for item in candidate["instances"] if isinstance(item, dict)}
    if set(champion_map) != set(candidate_map) or None in champion_map:
        raise EvolutionError("run results do not contain the same instance IDs")
    references = _reference_values(references_path, family)
    rows: list[dict[str, Any]] = []
    wins = ties = losses = 0
    improvements: list[float] = []
    champion_gaps: list[float] = []
    candidate_gaps: list[float] = []
    all_valid = champion.get("status") == "valid" and candidate.get("status") == "valid"
    for instance_id in sorted(champion_map):
        baseline = champion_map[instance_id]
        proposed = candidate_map[instance_id]
        baseline_valid = baseline.get("status") == "valid" and isinstance(baseline.get("objective"), (int, float))
        proposed_valid = proposed.get("status") == "valid" and isinstance(proposed.get("objective"), (int, float))
        all_valid = all_valid and baseline_valid and proposed_valid
        row: dict[str, Any] = {
            "id": instance_id,
            "champion_status": baseline.get("status"),
            "candidate_status": proposed.get("status"),
            "champion_objective": baseline.get("objective"),
            "candidate_objective": proposed.get("objective"),
        }
        if baseline_valid and proposed_valid:
            base = float(baseline["objective"])
            cand = float(proposed["objective"])
            improvement = (
                (base - cand) / max(abs(base), 1e-12) * 100.0
                if sense == "min"
                else (cand - base) / max(abs(base), 1e-12) * 100.0
            )
            row["relative_improvement_percent"] = improvement
            improvements.append(improvement)
            tolerance = 1e-9 * max(1.0, abs(base), abs(cand))
            if (cand < base - tolerance and sense == "min") or (cand > base + tolerance and sense == "max"):
                outcome = "win"
                wins += 1
            elif abs(cand - base) <= tolerance:
                outcome = "tie"
                ties += 1
            else:
                outcome = "loss"
                losses += 1
            row["outcome"] = outcome
            reference = references.get(instance_id)
            if isinstance(reference, dict) and isinstance(reference.get("value"), (int, float)):
                ref_value = float(reference["value"])
                champion_gap = _gap(base, ref_value, sense)
                candidate_gap = _gap(cand, ref_value, sense)
                row["reference"] = {"kind": reference.get("kind"), "value": ref_value}
                row["champion_gap_percent"] = champion_gap
                row["candidate_gap_percent"] = candidate_gap
                champion_gaps.append(champion_gap)
                candidate_gaps.append(candidate_gap)
        rows.append(row)
    result = {
        "version": 1,
        "family": family,
        "objective_sense": sense,
        "evaluation_kind": champion.get("evaluation_kind"),
        "eligible": all_valid,
        "champion": {
            "result": str(champion_path.resolve()),
            "solver_sha256": champion.get("solver", {}).get("sha256"),
            "wall_time_sec": champion.get("solver", {}).get("wall_time_sec"),
        },
        "candidate": {
            "result": str(candidate_path.resolve()),
            "solver_sha256": candidate.get("solver", {}).get("sha256"),
            "wall_time_sec": candidate.get("solver", {}).get("wall_time_sec"),
        },
        "instances": rows,
        "summary": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mean_relative_improvement_percent": sum(improvements) / len(improvements) if improvements else None,
            "worst_regression_percent": max([0.0] + [-value for value in improvements]),
            "champion_mean_gap_percent": sum(champion_gaps) / len(champion_gaps) if champion_gaps else None,
            "candidate_mean_gap_percent": sum(candidate_gaps) / len(candidate_gaps) if candidate_gaps else None,
            "candidate_max_gap_percent": max(candidate_gaps) if candidate_gaps else None,
            "wall_time_ratio": (
                float(candidate.get("solver", {}).get("wall_time_sec"))
                / float(champion.get("solver", {}).get("wall_time_sec"))
                if isinstance(candidate.get("solver", {}).get("wall_time_sec"), (int, float))
                and isinstance(champion.get("solver", {}).get("wall_time_sec"), (int, float))
                and float(champion["solver"]["wall_time_sec"]) > 0
                else None
            ),
        },
    }
    _write(output, result)
    return result


def _hashes(root: Path, family: str) -> dict[str, str]:
    return {
        relative.as_posix(): _sha256(root / relative)
        for relative in _family_relative_paths(root, family)
    }


def decide(
    solve_root: Path,
    family: str,
    candidate_root: Path,
    comparison_path: Path,
    decision_value: str,
    reason: str,
    output: Path,
) -> dict[str, Any]:
    if decision_value not in {"accept", "reject"}:
        raise EvolutionError("decision must be accept or reject")
    comparison = _load(comparison_path)
    if not isinstance(comparison, dict) or comparison.get("family") != family:
        raise EvolutionError("comparison does not match the family")
    if decision_value == "accept" and comparison.get("eligible") is not True:
        raise EvolutionError("cannot accept an ineligible comparison")
    before = _hashes(solve_root, family)
    after = before
    if decision_value == "accept":
        _family_relative_paths(candidate_root, family)
        reference_paths = [
            Path("references") / "families" / f"{family}.json",
            Path("references") / "families" / f"{family}.md",
        ]
        saved_references = {relative: (solve_root / relative).read_bytes() for relative in reference_paths}
        target_scripts = solve_root / "scripts" / "families" / family
        backup_scripts = target_scripts.parent / f".{family}.backup-{uuid.uuid4().hex}"
        incoming_scripts = target_scripts.parent / f".{family}.incoming-{uuid.uuid4().hex}"
        shutil.copytree(candidate_root / "scripts" / "families" / family, incoming_scripts)
        swapped = False
        try:
            for relative in reference_paths:
                _write_bytes(solve_root / relative, (candidate_root / relative).read_bytes())
            os.replace(target_scripts, backup_scripts)
            swapped = True
            os.replace(incoming_scripts, target_scripts)
            after = _hashes(solve_root, family)
            shutil.rmtree(backup_scripts)
        except Exception:
            for relative, content in saved_references.items():
                _write_bytes(solve_root / relative, content)
            if swapped:
                if target_scripts.exists():
                    shutil.rmtree(target_scripts)
                if backup_scripts.exists():
                    os.replace(backup_scripts, target_scripts)
            if incoming_scripts.exists():
                shutil.rmtree(incoming_scripts)
            raise
    result = {
        "version": 1,
        "family": family,
        "decision": decision_value,
        "reason": reason,
        "decided_at": _utc_now(),
        "comparison": str(comparison_path.resolve()),
        "before": before,
        "after": after,
    }
    _write(output, result)
    return result


def _print(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Minimal deterministic tools for evolve-cop")
    sub = cli.add_subparsers(dest="command", required=True)

    snapshot_cli = sub.add_parser("snapshot")
    snapshot_cli.add_argument("--solve-root", type=Path, required=True)
    snapshot_cli.add_argument("--family", required=True)
    snapshot_cli.add_argument("--output", type=Path, required=True)

    stage_cli = sub.add_parser("stage")
    stage_cli.add_argument("--snapshot", type=Path, required=True)
    stage_cli.add_argument("--output", type=Path, required=True)

    request_cli = sub.add_parser("request")
    request_cli.add_argument("--side", choices=["champion", "candidate"], required=True)
    request_cli.add_argument("--family", required=True)
    request_cli.add_argument("--family-root", type=Path, required=True)
    request_cli.add_argument("--manifest", type=Path, required=True)
    request_cli.add_argument("--evaluator-root", type=Path)
    request_cli.add_argument("--time-sec", type=float, required=True)
    request_cli.add_argument("--tokens", type=int, required=True)
    request_cli.add_argument("--evidence", action="append", default=[])
    request_cli.add_argument("--output", type=Path, required=True)

    compare_cli = sub.add_parser("compare")
    compare_cli.add_argument("--champion", type=Path, required=True)
    compare_cli.add_argument("--candidate", type=Path, required=True)
    compare_cli.add_argument("--references", type=Path)
    compare_cli.add_argument("--output", type=Path, required=True)

    decide_cli = sub.add_parser("decide")
    decide_cli.add_argument("--solve-root", type=Path, required=True)
    decide_cli.add_argument("--family", required=True)
    decide_cli.add_argument("--candidate-root", type=Path, required=True)
    decide_cli.add_argument("--comparison", type=Path, required=True)
    decide_cli.add_argument("--decision", choices=["accept", "reject"], required=True)
    decide_cli.add_argument("--reason", required=True)
    decide_cli.add_argument("--output", type=Path, required=True)

    args = cli.parse_args(argv)
    try:
        if args.command == "snapshot":
            result = snapshot(args.solve_root.resolve(), args.family, args.output.resolve())
        elif args.command == "stage":
            result = stage(args.snapshot.resolve(), args.output.resolve())
        elif args.command == "request":
            result = request(
                args.side,
                args.family,
                args.family_root.resolve(),
                args.manifest.resolve(),
                args.time_sec,
                args.tokens,
                args.evidence,
                args.output.resolve(),
                args.evaluator_root.resolve() if args.evaluator_root else None,
            )
        elif args.command == "compare":
            result = compare(
                args.champion.resolve(),
                args.candidate.resolve(),
                args.output.resolve(),
                args.references.resolve() if args.references else None,
            )
        else:
            result = decide(
                args.solve_root.resolve(),
                args.family,
                args.candidate_root.resolve(),
                args.comparison.resolve(),
                args.decision,
                args.reason,
                args.output.resolve(),
            )
        _print(result)
        return 0
    except (EvolutionError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
