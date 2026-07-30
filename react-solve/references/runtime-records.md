# Runtime record contract

Read this file only when first creating or repairing runtime records. It defines the minimum formats needed for recovery, comparison, and distillation. It does not define a solving workflow or require fields irrelevant to the current task.

## Stable layout

For a new task that starts from a statement and instances, use one stable `runtime/<problem>/` workspace:

```text
runtime/<problem>/
  src/                 # editable solver, evaluator, and runners
  solvers/<version>/   # immutable formal source snapshots and fingerprints
  runs/<version>/      # per-instance run trajectories
  metrics/             # script-generated derived facts
  incumbents/          # per-instance historical best feasible solutions
  output/              # final user-facing solutions and reports
  instance-analysis.md
  rounds.jsonl
  incumbent-provenance.jsonl
  FINAL_SOLVER         # points to or contains the final frozen version
```

Leave inputs and references in their user-provided locations; do not copy them into the solver or output. Keep these boundaries stable after creation: edit in `src/`, add new versions under `solvers/`, and add corresponding artifacts under runs, metrics, and incumbents. Do not move code and results into or out of the workspace root as stages change. When the user supplies an existing project layout, do not relocate its source, but still centralize newly generated run evidence, version snapshots, and deliverables in this runtime.

## General rules

- Encode JSONL as UTF-8, one JSON object per line, appended in event order. Fields listed below are required unless marked optional.
- The current `schema_version` is `1`. Task-specific fields may be added when they do not change existing field semantics.
- Resolve relative paths from `runtime/<problem>/`. Keep every solution file referenced by provenance immutable.
- The `solver_version` of formal v1, a round-closing candidate, or `FINAL_SOLVER` must resolve to an immutable source snapshot under `solvers/<version>/` plus a mechanically generated fingerprint or file manifest. Continue later experiments in `src/` under a new version identifier rather than overwriting an existing version.
- An existing run is eligible for formal coverage only when its `run_start` instance, solver snapshot, evaluator, seed semantics, formal budget ceiling, and materially comparable resource conditions all match the current request, and it has a valid completed `run_end`. File or path existence, a matching run ID, or a completed short-budget diagnostic does not establish eligibility. A runner's skip-existing logic must parse and verify the record rather than checking only its filename.
- Do not duplicate mechanical information: run conditions and trajectories belong to runs, derived facts to metrics, action judgments to rounds, and best-solution sources to provenance.
- Preserve the evaluator's actual JSON `quality`; do not rewrite comparison semantics for logging convenience.

## `instance-analysis.md`

This is compact Markdown for the main Agent, not a machine schema. It must contain:

- the problem/evaluator semantics, budget, and source of the authoritative instance inventory;
- every instance identifier, with design-relevant facts, risks, and unknowns;
- important cross-instance differences and outliers;
- a coverage check with `input_count`, `analyzed_count`, `missing`, and `extra`.

Use a per-instance table, sections, or shared notes to compress repeated facts, but keep every identifier checkable. Do not impose fixed feature columns or choose algorithms, implementation, scheduling, or a portfolio for the main Agent.

## `rounds.jsonl`

This is an append-only ReAct semantic trace. One open round may contain multiple `decision`/`outcome` pairs. Append one `round_close` only after comparable evidence covers the target batch. Round 0 is the v1 baseline and does not count as an improvement round. Round numbers increase monotonically within one runtime, but each solving or improvement request owns an independent additional-round target; historical rounds are evidence only.

At the start of a new request, read the highest current `round_close.round` (0 in a new runtime), then append:

```json
{"schema_version":1,"type":"round_target","target_id":"request-2","start_closed_round":10,"required_additional_rounds":5,"target_closed_round":15,"rounds_source":"default"}
```

Keep `target_id` unique within the file. Set `required_additional_rounds` to the user's count, or 5 when omitted; `target_closed_round` must equal start plus additional rounds. When resuming the same interrupted request, continue using its target if it has no `round_target_close`. A new request to improve again appends a new target and may not offset it with existing `round_close` records. For legacy runtime without targets, use the highest closed round immediately before the first new action as the start.

Do not rewrite legacy runtime. Existing `round_close` records that are otherwise checkable still determine the start snapshot. Begin using `round_target`, `local_qualification`, and the `attention_coverage` field below on the current request's new records.

When the request ends, append:

```json
{"schema_version":1,"type":"round_target_close","target_id":"request-2","actual_closed_round":15,"status":"completed","reason":"target_reached"}
```

For new records, use `completed` or `incomplete` for `status`. Write `completed` only after reaching the target round. Write `incomplete` only when a changed user requirement or a concrete non-resumable external hard constraint prevents continuation, with the cause in `reason`. Candidate failures, an open round without a promotable version, substantial wall time or context, a self-imposed work limit, and a resumable interruption do not close the target. Treat `early_completed` in legacy records as historical data only.

Before an action, append:

```json
{"schema_version":1,"type":"decision","round":1,"action_id":"r1-a1","trigger":"fact that triggered this action","judgment":"current revisable judgment","alternatives":[],"learned_consulted":[],"action":"planned action","change_rationale":"why it is worth doing","prediction":"expected observation","weaken_if":"result that would weaken the judgment"}
```

After the action, append:

```json
{"schema_version":1,"type":"outcome","round":1,"action_id":"r1-a1","artifact_refs":["solvers/v2","runs/v2"],"result":"actual observation","evidence_update":"supported","disposition":"keep, repair, or back off decision","unresolved":[],"next":"next judgment"}
```

After per-instance validation coverage is complete, append:

```json
{"schema_version":1,"type":"round_close","round":1,"candidate_solver_version":"v2","stable_solver_version":"v2","local_qualification":[{"instance":"a","aspects":["quality"],"criterion":"strictly improve feasible quality at a common 20-second cutoff","comparison_basis":"same evaluator, resources, and seed semantics; baseline incumbent at 20 seconds","baseline_ref":"metrics/local-v1-a.json","candidate_ref":"metrics/local-v2-a.json","status":"passed"}],"validation_coverage":[{"mode":"fresh","instances":["a"],"evidence_refs":["runs/v2/a.jsonl"]},{"mode":"carried_forward","instances":["b"],"source_solver_version":"v1","evidence_refs":["runs/v1/b.jsonl"],"reason":"changed branch is unreachable for b; shared path and run conditions are unchanged"}],"full_batch_metrics_ref":null,"disposition":"promote","attention_coverage":[{"instance":"a","aspects":["quality","efficiency"],"evidence_refs":["runs/v2/a.jsonl"],"status":"improved","next":"continue reducing positive gap"}],"unresolved_regressions":[]}
```

Constraints:

- Keep `action_id` unique within the file. Every `outcome` references an existing `decision`; interrupted and failed actions still receive an outcome using `inconclusive` or the value actually supported.
- `evidence_update` is one of `supported`, `weakened`, or `inconclusive`.
- `alternatives`, `learned_consulted`, `artifact_refs`, `unresolved`, and `unresolved_regressions` are arrays; use `[]` when empty.
- In an improvement round, `local_qualification` is nonempty and covers the target instances that determined the candidate direction. Each entry records the target aspects, the promotion condition stated before action, stable-baseline and candidate evidence references, `comparison_basis`, and `passed` status. Use a common cutoff and materially comparable conditions. Do not enter round-closing validation or write `round_close` unless every entry passes.
- `validation_coverage` lists every authoritative instance exactly once. `fresh` records new evidence. `carried_forward` records its source, evidence, and a reason supported by the actual diff and executed path. Use `fresh` when impact cannot be ruled out. Fill `full_batch_metrics_ref` only when every instance was freshly run; otherwise use `null`.
- Seed changes, mere time-limit or repeat-count changes, renaming, wrappers, reruns, backoffs, local failures, and report generation do not receive their own `round_close`. A small change may be substantive when it repairs a confirmed defect or creates a locally verified new capability.
- `attention_coverage` covers every known per-instance issue, including positive gaps when available, anomalies, and repeatedly unresolved items found by the formal-coverage review. Reaching a quality reference closes only a `quality` item, not an efficiency or scheduling issue. Carry newly exposed issues into the next round.
- Round 0 uses the same decision/outcome format: the v1 design is a decision and the formal baseline is an outcome. Close round 0 only after the full-batch baseline and per-instance review, recording the first attention set in `attention_coverage`; use `[]` only when no issue is known.

## `runs/*.jsonl`, `references.json`, and `metrics/*.json`

The top-level documentation and validation code in `scripts/metrics.py` are the only authority for these machine records. Run `python <skill-dir>/scripts/metrics.py --help` to inspect the contract:

- each run file describes one execution for one instance;
- the solver or a light wrapper writes `run_start`, strictly improving feasible `incumbent` events, and `run_end`;
- optional `references.json` comes from the evaluation side and must not be read by the solver;
- only the script generates `metrics/*.json`; do not edit snapshots manually.

Before a formal baseline or round close, use `--formal-budget-seconds` to declare the current formal ceiling and optionally add `--formal-solver-version` and `--formal-evaluator-version`. With `--strict-formal`, any supplied record with the wrong budget or version, or an incomplete status, fails the check instead of being counted toward formal coverage. The Agent still verifies resource comparability and the authoritative instance inventory from task facts.

`run_end.stop_reason` names the single primary cause that actually triggered the whole solver to return, such as `target_reached`, `proven_optimal`, `budget_exhausted`, `interrupted_for_diagnosis`, or `error`. A formal run may end only after reaching a trustworthy target, proving optimality, or exhausting the external budget; completion of a fixed iteration count, restart count, stage, or portfolio may trigger only internal scheduling. If a fixed work cap does truncate a diagnostic run, record `work_cap_reached`; do not present it as formal completion. A clearer task-specific reason is allowed, but do not combine unresolved explanations into a label such as `budget_or_stagnation`.

`run_end` may include a task-specific `telemetry` object for the minimum run facts needed by the current judgment and not recoverable from the incumbent trajectory, such as whether planned stages were entered, work completed and time spent per stage, accepted actions, or incumbent contributions. This is not a fixed monitoring checklist. Record it only when judging stage reachability, effective search throughput, or scheduling. `metrics.py` validates that it is an object and preserves it without interpreting task semantics.

Every `solution_ref` points to a distinct, immutable solution file; do not reuse an overwriteable path. Record in `run_start.resources` only the materially comparable conditions that actually took effect.

## `incumbent-provenance.jsonl` and incumbent solutions

Whenever the evaluator confirms a solution is strictly better than the previous task-wide best feasible solution for that instance, save an immutable solution file and append:

```json
{"schema_version":1,"instance":"a","solution_ref":"incumbents/a/v2-run7.sol","quality":100,"evaluator_version":"e1","source":{"kind":"run","run_id":"v2-a-seed1","solver_version":"v2","elapsed_seconds":1.4}}
```

Required fields are `instance`, `solution_ref`, `quality`, `evaluator_version`, and `source`. When `source.kind` is `run`, also record `run_id`, `solver_version`, and `elapsed_seconds` within that run. For a user-provided or externally imported initial solution, use another clear `kind` and `source_ref`; do not invent a run.

The incumbent solution format is problem-specific and must be independently readable and validatable by the current evaluator. Backing off a solver, changing references, or regenerating metrics must not delete or overwrite a recorded solution or its provenance.

Before complete delivery, every per-instance best solution must be traceable through `incumbent-provenance.jsonl` to a still-existing immutable `solution_ref`.
