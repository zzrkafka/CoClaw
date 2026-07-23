# Solver interface

Write one self-contained Python program using the standard library unless an available dependency has clear value.

Accept exactly:

```text
python solver.py --manifest <batch.json> --output-dir <directory>
```

Resolve each instance path relative to the manifest directory. Process every entry with shared logic; never branch on filenames, benchmark identity, Oracle/BKS values, or expected answers.

For each entry, atomically write:

```text
<output-dir>/<instance_id>.solution.json
```

Candidate:

```json
{
  "version": 1,
  "instance_id": "tiny5",
  "status": "candidate",
  "objective": 44,
  "solution": {"tour": ["a", "b", "c", "e", "d"]},
  "telemetry": {}
}
```

Failure:

```json
{
  "version": 1,
  "instance_id": "tiny5",
  "status": "error",
  "message": "concise reason"
}
```

Allowed statuses are `candidate`, `no_candidate`, `error`, and `interrupted`. Objective and telemetry are solver claims; the runner and family validator recompute trusted values.

Respect every per-instance limit and the batch limit. Establish a safe candidate or honest failure for every instance before deep search can starve later entries. Reserve time to publish all final files.

Telemetry is optional and family-neutral. Report only mechanisms that actually ran, with useful counts such as proposals, accepted state changes, incumbent gains, and time. Do not emit algorithm aliases as evidence.
