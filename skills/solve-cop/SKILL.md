---
name: solve-cop
description: Generate, debug, run, and independently validate one complete batch solver for a recognized combinatorial-optimization family. Use for ordinary COP solving or for one blinded champion/candidate phenotype requested by evolve-cop; return solver code, solutions, validated objectives, runtime evidence, and limitations.
---

# Solve COP

Own one complete `solver.py` in the current Agent context. Do not delegate coding to another Agent and do not treat family assets as a solver codebase.

## Recognize before solving

Use `scripts/cop.py recognize` against the supplied inputs. Ordinary solving requires one unambiguous `active` family whose parser and validator cover the observed semantics. Hand unknown, ambiguous, initializing, or uncovered inputs to `evolve-cop`.

When called by `evolve-cop`, read the supplied phenotype request instead. Use only its isolated family root, anonymous batch, budget, and requested evidence. Do not seek Oracle/BKS values, gaps, acceptance thresholds, prior solvers, or the other phenotype.

Pass the request's `family_root` to `cop.py --skill-root`. If it supplies `evaluator_root`, pass that to `cop.py run --evaluator-root`; this lets both phenotypes use one repaired evaluator without exposing the other phenotype.

Read [references/solver-interface.md](references/solver-interface.md) and only the recognized family guide. Keep generated code and results in an independent work directory; never modify family assets.

## React to evidence

Use the host Agent loop directly:

1. Inspect the latest code, tool output, validation, and metrics.
2. Identify the largest current correctness, completeness, or effectiveness uncertainty.
3. Choose the cheapest action likely to resolve it or materially improve the solver.
4. Act, observe the result, and revise the hypothesis.

Do not predeclare a phase queue or manufacture a separate ReAct state machine. The conversation and tool results are working state. For long runs, leave a short `journal.jsonl` containing only observation, action, result, and budget used.

Write a protocol-complete executable early, then improve the same lineage. Preserve a recoverable best revision before risky edits. Start over only when no coherent executable formed, the context is unusable, or the architecture cannot support the required correction.

## Debug semantics, not only crashes

Distinguish:

- execution failures;
- infeasible or incorrectly scored solutions;
- incomplete algorithms whose claimed mechanisms are unreachable, starved, non-state-changing, or overwritten;
- complete but ineffective search with poor convergence, allocation, or basin recovery.

Use `scripts/cop.py run` for independent validation. Reproduce the smallest failing case, patch minimally, and rerun the informative probe. Solver self-reports are diagnostic claims, never validation.

## Spend evaluation deliberately

Use only time, tokens, and full evaluations. Defaults are `tokens: 800000` and `full_evaluations: 5`; require an explicit time budget.

Cheap microcases, shortened representative runs, forced mechanism probes, and local ablations do not consume a full evaluation. Before a full target run, establish that the solver compiles, obeys the interface, emits validator-accepted solutions, and has evidence against its largest quality risk.

Protect every instance and the validated weak tail. Do not exchange a material feasibility, crash, timeout, or worst-instance regression for a better mean unless the user explicitly permits that trade.

## Deliver

Return the best complete `solver.py`, its `run-result.json`, solution and validation files, per-instance objectives and compatible supplied-reference gaps, actual budget use, and unresolved limitations.

For an evolution phenotype, return validated objectives without reference-aware judgments. `evolve-cop` owns pairing and family acceptance.
