---
name: evolve-cop
description: Initialize an unknown combinatorial-optimization family or improve persistent family guidance, manifest, parser, validator, fixtures, and activation evidence through autonomous evidence-driven experiments with isolated solve-cop Agents. Use when a family is missing, initializing, stale, semantically uncovered, or repeatedly produces weak, incorrect, or inefficient solvers.
---

# Evolve COP

Improve what a fresh `solve-cop` Agent reliably builds. Generated solvers are disposable phenotypes; persistent knowledge is limited to family guidance, manifest, parser, validator, fixtures, and activation evidence.

## Keep the boundary simple

- Own family hypotheses, candidate edits, experiment choice, paired comparison, and accept/reject decisions.
- Give each new champion or candidate phenotype one fresh isolated Agent executing `solve-cop`.
- Never write, patch, or Debug a phenotype solver from evolve.
- Keep Oracle/BKS values, gaps, thresholds, prior solvers, and the other side hidden from the whole solve Agent.
- Keep all snapshots, candidates, solvers, solutions, and comparisons outside the published Skills.

Read [references/tools.md](references/tools.md) only for the tool you are about to use. Load only the target family and relevant fixtures.

## Run one real outer loop

Use the host Agent loop, not a custom state machine:

1. Observe the latest family, generated code, validation, per-instance results, and budget.
2. Form a falsifiable family-level explanation for the largest recurring failure or uncertainty.
3. Choose the cheapest useful action: inspect, change, call a phenotype, request another probe, compare, accept, reject, change strategy, or stop.
4. Act and let the new evidence determine the next action.

A round number is only a budget cap. Do not execute a fixed round template. Record accepted/rejected decisions and their evidence in the run directory; ordinary tool traces are sufficient for intermediate history.

Use only `time_sec`, `tokens`, and `full_evaluations`. Defaults are `tokens: 800000` and `full_evaluations: 5`; set a time budget. Solver generation and probes consume time/tokens. One complete matched champion/candidate comparison consumes one full evaluation.

## Resolve or initialize

Continue an existing family when objective, representation, and feasibility semantics match, even if it is initializing or stale. Initialize only when none matches.

A new family begins with a compact manifest and guide, parser, validator, and adversarial fixtures. Keep it `initializing` until recognition, parsing, validation, and cold solver effectiveness are credible. Do not create a generic `oracle.py`.

Before changing a family version, inspect what a cold solve Agent is likely to implement. Treat a one-off solver bug as phenotype-local; change persistent guidance only when the text or repeated independent failures support a family-level cause.

Write guidance as concise condition → action → observable feedback. Its purpose is to produce efficient correct code, not to teach the whole algorithmic field.

## Experiment fairly

Snapshot the current family and stage one coherent candidate without touching canonical files. A candidate may contain several mutually supporting high-confidence edits.

Before seeing candidate results, declare:

- correctness and full-evaluation admission gates;
- primary quality and efficiency criteria;
- per-instance and worst-tail regression limits;
- terminal success and stopping conditions;
- how available objective references will be used.

With complete references, use eligible absolute gaps. With none, use matched validated objectives, wins/losses, worst regression, time-to-match, anytime gain, robustness, and scaling. With partial references, combine both. Correctness is a gate; diversity is not a goal.

Create a minimal blinded phenotype request with `scripts/evolve.py request`, then start a fresh Agent and tell it to execute `solve-cop` against that request. If the same phenotype needs more solver-level evidence, continue with that Agent while its lineage remains valid.

Use cheap probes until the main uncertainty is resolved. Run `scripts/evolve.py compare` only when champion and candidate are validator-clean and mature enough to justify a matched full run.

## Accept only transferable improvement

Mechanism telemetry explains results but earns no diversity reward. Inspect generated solver code when needed to connect family wording to observed behavior.

If parser, validator, or objective semantics change, invalidate earlier comparisons and rerun both sides with the repaired evaluator.

Accept only when predeclared criteria pass with bounded per-instance regressions. Use `scripts/evolve.py decide` to copy the exact staged family atomically. Reject without changing canonical files. Continue while the goal is unmet, budget remains, and another action has credible value; otherwise stop.

Deliver accepted family changes, activation evidence, comparisons, budget use, decision history, and limitations. Keep phenotype solvers and solutions as run artifacts.
