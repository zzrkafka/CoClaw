#!/usr/bin/env python3
"""Summarize comparable COP solver runs without making diagnostic decisions.

RUN INPUT CONTRACT (schema_version 1)

Each input JSONL file describes exactly one instance run. Extra fields are
allowed, but the following events and fields are authoritative:

1. First event:
   {"type":"run_start","schema_version":1,"run_id":"v1-a","instance":"a",
    "solver_version":"v1","evaluator_version":"e1","seed":1,
    "budget_seconds":60,"sense":"min"}

   Required fields are run_id, instance, solver_version, evaluator_version,
   seed (use null for deterministic runs), positive budget_seconds, and sense
   in {min,max,custom}. Optional resources must be an object. Optional command
   must be a string or an array of strings.

2. Zero or more feasible, strictly improving incumbents:
   {"type":"incumbent","elapsed_seconds":0.2,"quality":120,
    "solution_ref":"incumbents/a/v1-a-0.2.sol"}

   elapsed_seconds is monotonic from process start and therefore includes
   parsing, preprocessing, compilation, and search. solution_ref is optional
   but, when present, must be a non-empty string unique across the supplied
   run set so each event can identify an immutable solution. For min/max,
   quality must be numeric and strictly improve in the declared direction.
   For custom, the evaluator owns comparison and the producer must emit only
   strict improvements; this script preserves the JSON quality without
   ordering it.

3. Final event:
   {"type":"run_end","end_to_end_seconds":1.5,"status":"completed",
    "stop_reason":"target_reached"}

   status and stop_reason are non-empty strings. Optional violations preserves
   evaluator output. Optional telemetry must be an object and preserves
   task-specific facts needed to judge stage reachability, effective search
   throughput, or scheduling. No event may follow run_end.

REFERENCE INPUT

--references accepts one JSON object keyed by instance. Each value is either a
number or {"quality": <number>, ...}. References are evaluation-side metadata;
the solver must not read them.

SNAPSHOT OUTPUT

The generated JSON has schema_version, generated_by, run_count,
solver_summaries, and runs. Each run includes feasibility, final quality,
first-feasible time, time-to-best, time spent after the best and its fraction
of end-to-end time, end-to-end time, budget-use fraction, stop reason, the full
incumbent trajectory, optional telemetry, and optional reference difference/gap facts. Each solver
summary foregrounds every positive-gap run and every run's timing profile so
that an aggregate or worst case cannot hide another instance. The script
groups summaries by solver_version and evaluator_version, but does not
diagnose causes or select a solver.

FORMAL COVERAGE CHECK

--formal-budget-seconds marks records eligible for formal coverage only when
their declared budget matches the requested ceiling and status is completed.
Optional --formal-solver-version and --formal-evaluator-version add exact
version checks. --strict-formal fails instead of silently summarizing any
ineligible supplied record. Resource comparability and the authoritative
instance list remain task-side judgments.

reference_difference is signed so that positive means worse than the reference,
zero means equal, and negative means better. Unlike percentage gap, it remains
defined when a numeric reference is zero.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


class ContractError(ValueError):
    """Raised when a run record violates the common measurement contract."""


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def discover_inputs(raw_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*.jsonl") if item.is_file())
        else:
            raise ContractError(f"input does not exist: {path}")
    unique = sorted({item.resolve() for item in files}, key=lambda item: str(item).lower())
    if not unique:
        raise ContractError("no JSONL run records found")
    return unique


def load_references(path: str | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read references from {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError("references must be a JSON object keyed by instance")

    references: dict[str, dict[str, Any]] = {}
    for instance, raw in data.items():
        if not isinstance(instance, str) or not instance:
            raise ContractError("every reference key must be a non-empty instance string")
        if is_number(raw):
            references[instance] = {"quality": raw, "kind": "reference"}
            continue
        if not isinstance(raw, dict) or not is_number(raw.get("quality")):
            raise ContractError(
                f"reference for {instance!r} must be a number or an object with numeric quality"
            )
        reference = dict(raw)
        reference.setdefault("kind", "reference")
        references[instance] = reference
    return references


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ContractError(f"{path}:{line_number}: each JSONL record must be an object")
        records.append(record)
    if not records:
        raise ContractError(f"{path}: empty run record")
    return records


def require_nonempty_string(record: dict[str, Any], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location}: {field} must be a non-empty string")
    return value


def require_nonnegative_number(record: dict[str, Any], field: str, location: str) -> float:
    value = record.get(field)
    if not is_number(value) or value < 0:
        raise ContractError(f"{location}: {field} must be a finite non-negative number")
    return float(value)


def reference_gap_pct(quality: Any, reference_quality: Any, sense: str) -> float | None:
    if not is_number(quality) or not is_number(reference_quality) or reference_quality == 0:
        return None
    if sense == "min":
        return 100.0 * (float(quality) - float(reference_quality)) / abs(float(reference_quality))
    if sense == "max":
        return 100.0 * (float(reference_quality) - float(quality)) / abs(float(reference_quality))
    return None


def reference_difference(quality: Any, reference_quality: Any, sense: str) -> float | None:
    if not is_number(quality) or not is_number(reference_quality):
        return None
    if sense == "min":
        return float(quality) - float(reference_quality)
    if sense == "max":
        return float(reference_quality) - float(quality)
    return None


def threshold_key(value: float) -> str:
    return format(value, ".12g")


def parse_run(
    path: Path,
    references: dict[str, dict[str, Any]],
    gap_thresholds: list[float],
) -> dict[str, Any]:
    records = read_records(path)
    start = records[0]
    if start.get("type") != "run_start":
        raise ContractError(f"{path}: first record must have type run_start")
    if start.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"{path}: unsupported schema_version {start.get('schema_version')!r}")

    run_id = require_nonempty_string(start, "run_id", str(path))
    instance = require_nonempty_string(start, "instance", str(path))
    solver_version = require_nonempty_string(start, "solver_version", str(path))
    evaluator_version = require_nonempty_string(start, "evaluator_version", str(path))
    budget_seconds = require_nonnegative_number(start, "budget_seconds", str(path))
    if budget_seconds <= 0:
        raise ContractError(f"{path}: budget_seconds must be greater than zero")
    sense = start.get("sense")
    if sense not in {"min", "max", "custom"}:
        raise ContractError(f"{path}: sense must be min, max, or custom")
    if "seed" not in start:
        raise ContractError(f"{path}: run_start must include seed; use null for a deterministic run")
    if "resources" in start and not isinstance(start["resources"], dict):
        raise ContractError(f"{path}: optional resources must be a JSON object")
    if "command" in start:
        command = start["command"]
        if not (
            isinstance(command, str)
            or (
                isinstance(command, list)
                and all(isinstance(part, str) for part in command)
            )
        ):
            raise ContractError(
                f"{path}: optional command must be a string or an array of strings"
            )

    incumbents: list[dict[str, Any]] = []
    end: dict[str, Any] | None = None
    previous_elapsed = -math.inf
    previous_quality: float | None = None
    for index, record in enumerate(records[1:], start=2):
        location = f"{path}:{index}"
        record_type = record.get("type")
        if end is not None:
            raise ContractError(f"{location}: no records are allowed after run_end")
        if record_type == "incumbent":
            elapsed = require_nonnegative_number(record, "elapsed_seconds", location)
            if elapsed < previous_elapsed:
                raise ContractError(f"{location}: incumbent elapsed_seconds must be monotonic")
            if "quality" not in record:
                raise ContractError(f"{location}: incumbent must include quality")
            quality = record["quality"]
            if sense in {"min", "max"}:
                if not is_number(quality):
                    raise ContractError(
                        f"{location}: {sense} sense requires numeric incumbent quality; "
                        "use custom for evaluator-defined structured quality"
                    )
                numeric_quality = float(quality)
                if previous_quality is not None:
                    strictly_better = (
                        numeric_quality < previous_quality
                        if sense == "min"
                        else numeric_quality > previous_quality
                    )
                    if not strictly_better:
                        raise ContractError(
                            f"{location}: incumbent quality must strictly improve for {sense} sense"
                        )
                previous_quality = numeric_quality
            event = {
                "elapsed_seconds": elapsed,
                "quality": quality,
            }
            if "solution_ref" in record:
                if not isinstance(record["solution_ref"], str) or not record["solution_ref"]:
                    raise ContractError(
                        f"{location}: optional solution_ref must be a non-empty string"
                    )
                event["solution_ref"] = record["solution_ref"]
            incumbents.append(event)
            previous_elapsed = elapsed
        elif record_type == "run_end":
            end = record
        else:
            raise ContractError(f"{location}: unknown record type {record_type!r}")

    if end is None:
        raise ContractError(f"{path}: missing run_end record")
    end_to_end_seconds = require_nonnegative_number(end, "end_to_end_seconds", str(path))
    if incumbents and incumbents[-1]["elapsed_seconds"] > end_to_end_seconds + 1e-9:
        raise ContractError(f"{path}: incumbent occurs after end_to_end_seconds")
    status = require_nonempty_string(end, "status", str(path))
    stop_reason = require_nonempty_string(end, "stop_reason", str(path))
    if "telemetry" in end and not isinstance(end["telemetry"], dict):
        raise ContractError(f"{path}: optional telemetry must be a JSON object")

    reference = references.get(instance)
    trajectory: list[dict[str, Any]] = []
    numeric_gaps: list[tuple[float, float]] = []
    for event in incumbents:
        item = dict(event)
        gap = None
        difference = None
        if reference is not None:
            gap = reference_gap_pct(event["quality"], reference["quality"], sense)
            difference = reference_difference(event["quality"], reference["quality"], sense)
        if difference is not None:
            item["reference_difference"] = difference
        if gap is not None:
            item["reference_gap_pct"] = gap
            numeric_gaps.append((event["elapsed_seconds"], gap))
        trajectory.append(item)

    final_quality = incumbents[-1]["quality"] if incumbents else None
    first_feasible = incumbents[0]["elapsed_seconds"] if incumbents else None
    time_to_best = incumbents[-1]["elapsed_seconds"] if incumbents else None
    seconds_after_best = (
        max(0.0, end_to_end_seconds - time_to_best) if time_to_best is not None else None
    )
    time_after_best_fraction = (
        seconds_after_best / end_to_end_seconds
        if seconds_after_best is not None and end_to_end_seconds > 0
        else None
    )

    final_gap = numeric_gaps[-1][1] if numeric_gaps else None
    final_difference = (
        trajectory[-1].get("reference_difference") if trajectory else None
    )
    gap_area = None
    mean_gap_after_first = None
    if numeric_gaps:
        gap_area = 0.0
        for index, (elapsed, gap) in enumerate(numeric_gaps):
            next_elapsed = (
                numeric_gaps[index + 1][0]
                if index + 1 < len(numeric_gaps)
                else end_to_end_seconds
            )
            gap_area += gap * max(0.0, next_elapsed - elapsed)
        observed_duration = max(0.0, end_to_end_seconds - numeric_gaps[0][0])
        mean_gap_after_first = (
            gap_area / observed_duration if observed_duration > 0 else numeric_gaps[-1][1]
        )

    time_to_gap: dict[str, float | None] = {}
    for threshold in gap_thresholds:
        reached = next(
            (elapsed for elapsed, gap in numeric_gaps if gap <= threshold),
            None,
        )
        time_to_gap[threshold_key(threshold)] = reached

    result: dict[str, Any] = {
        "source_file": str(path),
        "run_id": run_id,
        "instance": instance,
        "solver_version": solver_version,
        "evaluator_version": evaluator_version,
        "seed": start["seed"],
        "sense": sense,
        "budget_seconds": budget_seconds,
        "status": status,
        "stop_reason": stop_reason,
        "feasible": bool(incumbents),
        "final_quality": final_quality,
        "first_feasible_seconds": first_feasible,
        "time_to_best_seconds": time_to_best,
        "end_to_end_seconds": end_to_end_seconds,
        "seconds_after_best": seconds_after_best,
        "time_after_best_fraction": time_after_best_fraction,
        "budget_used_fraction": end_to_end_seconds / budget_seconds,
        "incumbent_updates": len(incumbents),
        "trajectory": trajectory,
    }
    if "resources" in start:
        result["resources"] = start["resources"]
    if "command" in start:
        result["command"] = start["command"]
    if "violations" in end:
        result["violations"] = end["violations"]
    if "telemetry" in end:
        result["telemetry"] = end["telemetry"]
    if reference is not None:
        result["reference"] = reference
        result["final_reference_difference"] = final_difference
        result["final_reference_gap_pct"] = final_gap
        result["reference_gap_area_pct_seconds"] = gap_area
        result["mean_reference_gap_pct_after_first_feasible"] = mean_gap_after_first
        result["time_to_reference_gap_pct"] = time_to_gap
    return result


def finite_values(items: list[Any]) -> list[float]:
    return [float(item) for item in items if is_number(item)]


def formal_ineligibility_reasons(
    run: dict[str, Any],
    budget_seconds: float,
    solver_version: str | None,
    evaluator_version: str | None,
) -> list[str]:
    reasons: list[str] = []
    if run["status"] != "completed":
        reasons.append(f"status is {run['status']!r}, not 'completed'")
    if not math.isclose(
        float(run["budget_seconds"]),
        budget_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        reasons.append(
            f"budget_seconds is {run['budget_seconds']!r}, expected {budget_seconds!r}"
        )
    if solver_version is not None and run["solver_version"] != solver_version:
        reasons.append(
            f"solver_version is {run['solver_version']!r}, expected {solver_version!r}"
        )
    if evaluator_version is not None and run["evaluator_version"] != evaluator_version:
        reasons.append(
            "evaluator_version is "
            f"{run['evaluator_version']!r}, expected {evaluator_version!r}"
        )
    return reasons


def summarize_solver(
    version: str,
    evaluator_version: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    gaps = finite_values([run.get("final_reference_gap_pct") for run in runs])
    end_times = finite_values([run.get("end_to_end_seconds") for run in runs])
    infeasible = sorted({run["instance"] for run in runs if not run["feasible"]})
    missing_reference = sorted(
        {run["instance"] for run in runs if "reference" not in run}
    )
    failed_runs = sorted(
        run["run_id"] for run in runs if run["status"] != "completed"
    )
    positive_gap_runs = sorted(
        (
            {
                "run_id": run["run_id"],
                "instance": run["instance"],
                "gap_pct": run["final_reference_gap_pct"],
            }
            for run in runs
            if is_number(run.get("final_reference_gap_pct"))
            and float(run["final_reference_gap_pct"]) > 0
        ),
        key=lambda item: (item["instance"], item["run_id"]),
    )
    per_run_timing = sorted(
        (
            {
                "run_id": run["run_id"],
                "instance": run["instance"],
                "time_to_best_seconds": run["time_to_best_seconds"],
                "seconds_after_best": run["seconds_after_best"],
                "time_after_best_fraction": run["time_after_best_fraction"],
                "end_to_end_seconds": run["end_to_end_seconds"],
                "stop_reason": run["stop_reason"],
            }
            for run in runs
        ),
        key=lambda item: (item["instance"], item["run_id"]),
    )

    summary: dict[str, Any] = {
        "solver_version": version,
        "evaluator_version": evaluator_version,
        "runs": len(runs),
        "instances": len({run["instance"] for run in runs}),
        "completed_runs": sum(run["status"] == "completed" for run in runs),
        "feasible_runs": sum(run["feasible"] for run in runs),
        "infeasible_instances": infeasible,
        "failed_runs": failed_runs,
        "instances_without_reference": missing_reference,
        "positive_reference_gap_runs": positive_gap_runs,
        "per_run_timing": per_run_timing,
        "total_end_to_end_seconds": sum(end_times),
        "mean_end_to_end_seconds": statistics.fmean(end_times) if end_times else None,
    }
    if gaps:
        worst = max(
            (run for run in runs if is_number(run.get("final_reference_gap_pct"))),
            key=lambda run: float(run["final_reference_gap_pct"]),
        )
        summary.update(
            {
                "runs_with_reference_gap": len(gaps),
                "mean_final_reference_gap_pct": statistics.fmean(gaps),
                "median_final_reference_gap_pct": statistics.median(gaps),
                "worst_reference_gap": {
                    "run_id": worst["run_id"],
                    "instance": worst["instance"],
                    "gap_pct": worst["final_reference_gap_pct"],
                },
            }
        )
    else:
        summary["runs_with_reference_gap"] = 0
    return summary


def build_snapshot(
    files: list[Path],
    references: dict[str, dict[str, Any]],
    gap_thresholds: list[float],
    formal_budget_seconds: float | None = None,
    formal_solver_version: str | None = None,
    formal_evaluator_version: str | None = None,
    strict_formal: bool = False,
) -> dict[str, Any]:
    runs = [parse_run(path, references, gap_thresholds) for path in files]
    run_ids = [run["run_id"] for run in runs]
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicates:
        raise ContractError(f"duplicate run_id values: {', '.join(duplicates)}")
    solution_ref_sources: dict[str, list[str]] = {}
    for run in runs:
        for event in run["trajectory"]:
            solution_ref = event.get("solution_ref")
            if solution_ref is not None:
                solution_ref_sources.setdefault(solution_ref, []).append(run["run_id"])
    duplicate_solution_refs = {
        solution_ref: source_ids
        for solution_ref, source_ids in solution_ref_sources.items()
        if len(source_ids) > 1
    }
    if duplicate_solution_refs:
        details = "; ".join(
            f"{solution_ref} ({', '.join(source_ids)})"
            for solution_ref, source_ids in sorted(duplicate_solution_refs.items())
        )
        raise ContractError(
            "solution_ref must be unique per incumbent event across the supplied runs: "
            + details
        )
    runs.sort(key=lambda run: (run["solver_version"], run["instance"], str(run["seed"])))
    summary_keys = sorted(
        {(run["solver_version"], run["evaluator_version"]) for run in runs}
    )
    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "react-solve metrics.py",
        "run_count": len(runs),
        "solver_summaries": [
            summarize_solver(
                version,
                evaluator_version,
                [
                    run
                    for run in runs
                    if run["solver_version"] == version
                    and run["evaluator_version"] == evaluator_version
                ],
            )
            for version, evaluator_version in summary_keys
        ],
        "runs": runs,
    }
    if formal_budget_seconds is not None:
        ineligible: list[dict[str, Any]] = []
        eligible_run_ids: list[str] = []
        for run in runs:
            reasons = formal_ineligibility_reasons(
                run,
                formal_budget_seconds,
                formal_solver_version,
                formal_evaluator_version,
            )
            run["formal_coverage_eligible"] = not reasons
            if reasons:
                ineligible.append(
                    {
                        "run_id": run["run_id"],
                        "instance": run["instance"],
                        "source_file": run["source_file"],
                        "reasons": reasons,
                    }
                )
            else:
                eligible_run_ids.append(run["run_id"])
        snapshot["formal_coverage"] = {
            "criteria": {
                "status": "completed",
                "budget_seconds": formal_budget_seconds,
                "solver_version": formal_solver_version,
                "evaluator_version": formal_evaluator_version,
            },
            "eligible_run_ids": eligible_run_ids,
            "eligible_instances": sorted(
                {
                    run["instance"]
                    for run in runs
                    if run["formal_coverage_eligible"]
                }
            ),
            "ineligible_runs": ineligible,
        }
        if strict_formal and ineligible:
            details = "; ".join(
                f"{item['run_id']}: {', '.join(item['reasons'])}"
                for item in ineligible
            )
            raise ContractError(f"ineligible formal run records: {details}")
    return snapshot


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize common ReActSolve JSONL run records.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSONL run files or directories searched recursively for *.jsonl",
    )
    parser.add_argument(
        "--references",
        help="optional JSON object mapping instances to numbers or objects with numeric quality",
    )
    parser.add_argument(
        "--gap-threshold",
        action="append",
        type=float,
        default=[],
        help="report first time reference gap reaches this percentage; repeat as needed",
    )
    parser.add_argument(
        "--formal-budget-seconds",
        type=float,
        help="audit formal eligibility against this exact per-instance budget ceiling",
    )
    parser.add_argument(
        "--formal-solver-version",
        help="with --formal-budget-seconds, require this exact solver version",
    )
    parser.add_argument(
        "--formal-evaluator-version",
        help="with --formal-budget-seconds, require this exact evaluator version",
    )
    parser.add_argument(
        "--strict-formal",
        action="store_true",
        help="fail when any supplied run is ineligible for the requested formal check",
    )
    parser.add_argument("--output", help="write the snapshot to this JSON file")
    args = parser.parse_args(argv)
    formal_dependents = (
        args.formal_solver_version,
        args.formal_evaluator_version,
        args.strict_formal,
    )
    if args.formal_budget_seconds is None and any(formal_dependents):
        parser.error(
            "--formal-solver-version, --formal-evaluator-version, and "
            "--strict-formal require --formal-budget-seconds"
        )
    if (
        args.formal_budget_seconds is not None
        and (
            not math.isfinite(args.formal_budget_seconds)
            or args.formal_budget_seconds <= 0
        )
    ):
        parser.error("--formal-budget-seconds must be a finite positive number")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        files = discover_inputs(args.inputs)
        references = load_references(args.references)
        thresholds = sorted(set(args.gap_threshold), reverse=True)
        snapshot = build_snapshot(
            files,
            references,
            thresholds,
            formal_budget_seconds=args.formal_budget_seconds,
            formal_solver_version=args.formal_solver_version,
            formal_evaluator_version=args.formal_evaluator_version,
            strict_formal=args.strict_formal,
        )
    except ContractError as exc:
        print(f"metrics.py: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
