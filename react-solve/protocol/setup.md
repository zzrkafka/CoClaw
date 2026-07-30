# First construction: build a capable v1

The focus of first construction is to produce a genuinely capable solver v1 early and establish a credible baseline for the full instance batch within budget. Do not treat the first runnable program as v1, and do not let preparation, recordkeeping, or qualification displace algorithm design and implementation.

Fix the `runtime/<problem>/` workspace before writing code; see `references/runtime-records.md` for its layout and record formats. When the user already has a solver project, respect its source layout and put only new evidence, version snapshots, and deliverables in runtime.

## Understand every instance and the real problem

Read the problem statement, inputs, and existing code first. Establish feasibility, the true quality-comparison semantics, the authoritative instance inventory, budget, execution entry point, and available dependencies.

For every multi-instance batch, invoke one subagent to analyze the complete instance set; only let the main Agent do this directly for a single-instance task or when subagents are unavailable. The subagent gathers evidence only and does not choose algorithms, design the solver, or decide a portfolio:

- check each instance's design-relevant semantics, scale, structure, constraints, risks, and unknowns, and record them in `instance-analysis.md` according to `references/runtime-records.md`;
- identify outlying, special, and potentially difficult instances, explaining their material differences and the supporting facts;
- use grouping only to compress completed per-instance analysis, never to replace it with a group portrait, average, or “typical instance”;
- lead its final response with anomalies and unknowns most likely to affect the evaluator or solver design, then report shared facts and the coverage check without making algorithm decisions for the main Agent.

The main Agent may read the statement and prepare the workspace in parallel, but must wait for and read the complete subagent result before fixing evaluator semantics or the v1 design. Revisit `instance-analysis.md` as needed for high-value anomalies. This evidence directs attention but does not replace the main Agent's judgment.

When the user gives no budget, use a default ceiling of 120 seconds per instance. An explicit per-instance budget applies independently to every instance and is not a shared batch total. General-purpose tools, libraries, and optimization components are allowed within the user's permissions, but do not directly invoke or thinly wrap LKH or another problem-specific turnkey end-to-end solver in place of constructing the task solver.

BKS, complete reference solutions, and numeric gap are not prerequisites for constructing the solver. If a trustworthy reference is available at low cost from the task materials or allowed tools, use it for evaluation and compute gap. Do not spend substantial solving budget searching for one, and never let the final solver read references, historical answers, or task trajectories.

## Build a trustworthy evaluator

Build the evaluator independently from the statement and input semantics. It must verify complete feasibility, return the true quality and violation reason, and compare two results. It must not certify the solver by calling the solver's own objective, feasibility, or search-acceptance implementation; internal acceptance, ranking, pruning, and incremental evaluation must agree with it.

Check instance-specific semantics, rounding, floating point, penalties, hierarchical objectives, and incremental updates according to the actual risks rather than creating a universal testing ritual. If a trustworthy reference is beaten, an unexplained negative gap appears, or the candidate uses freedom not allowed by the reference, immediately re-examine the task and evaluator; do not treat the result as evidence until it is explained.

## Design and implement the solver seriously

Let Codex choose algorithms, representations, data structures, implementation methods, within-instance scheduling, and any necessary portfolio from the problem structure, full instance batch, existing code, available tools, and its own knowledge. Search learned material according to the current design question; rely on Codex's own judgment when learned is empty, unmatched, or inapplicable.

If the task scope already contains a reproducible solver, run, or incumbent, use it as a construction anchor. A stronger existing implementation may be the v1 starting point; do not deliver a per-instance weaker version merely to “construct again.”

Concentrate on writing a capable solver:

- exploit problem structure, produce strong feasible solutions early, and keep searching effectively for better ones with the remaining budget;
- identify the solving decisions that dominate quality; unless construction is already well justified, later search should be able to reconsider them rather than permanently freezing key groupings, assignments, orders, or configurations;
- make budget-dominating operations, data structures, and execution methods appropriate for the target scale, and confirm that the largest, outlying, and potentially difficult instances actually enter the intended effective paths;
- when instances need different mechanisms, allow a minimal, clear portfolio, but route by observable properties that affect algorithm behavior rather than benchmark identity.

These are design judgments, not a fixed algorithm list. Numba, JIT, vectorization, general-purpose libraries, and a portfolio are optional means; do not add complexity without evidence that it is needed.

Use short runs first to rule out crashes, infeasibility, evaluator disagreement, no-op changes, obvious target-scale bottlenecks, and nominal improvement paths that are unreachable or cannot affect the incumbent. Fix such problems inside round 0, then move promptly to the formal baseline rather than turning every design judgment into an experiment.

Within the external time ceiling, continue search that has a chance to improve the incumbent. Repeating exhausted fixed candidates, stages, or configurations merely consumes time; completing an internal iteration, restart, or stage cap cannot return from a formal run.

## Establish and report the round 0 baseline

Freeze a candidate as formal v1 only when the task and evaluator are trustworthy, target-scale execution matches the design, and the solver produces strong initial solutions with a reachable path to further improvement.

Save immutable versioned v1 source and its invocation, then give every instance in the full batch the declared or default per-instance ceiling. Interpret instances as they finish; if partial evidence shows that v1 cannot be a credible baseline or exposes a repairable defect, stop remaining runs that no longer add key discrimination, repair it inside round 0, and form the formal v1 again. Except after reaching a trustworthy target or proving optimality, a formal run must not end early.

Save runs, incumbents, and provenance according to `references/runtime-records.md`; use `scripts/metrics.py` to generate and verify per-instance quality-time facts. Report feasibility, objective, trustworthy reference and gap when available, time from process start until that run's final best solution first appeared, incumbent source, and reproducible v1 for every instance. Keep other temporal facts in runtime for diagnosis. Aggregates are only an overview and must not hide any instance's problem.

After the complete formal baseline is available, invoke another subagent to review every per-instance result and compare instances with comparable scale, structure, constraints, or actual code paths. It first scans the full batch, then examines incumbent curves, improvement timing, end-of-budget trends, and available telemetry for anomalous or repeatedly unresolved instances. It reports evidence, comparability, and explanations that remain unresolved, but does not decide causes, design algorithms, or choose a portfolio. Keep every positive gap when available. The main Agent must read the result and establish the complete attention set before closing round 0.

Close round 0 only after the final v1 has complete, valid, comparable per-instance coverage, and preserve every instance's historical best feasible solution. If the user requested first construction only, report and stop. Otherwise pass v1, the evaluator, runs, metrics, incumbents, anomalies and unknowns from instance analysis, and the per-instance issues exposed by the baseline to `loop.md`.
