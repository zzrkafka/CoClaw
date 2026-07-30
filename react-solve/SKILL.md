---
name: react-solve
description: "Build, run, and iteratively improve reproducible solvers for batches of combinatorial optimization problem (COP) instances through a ReAct evidence loop. Use when Codex must implement a solver from a problem statement and instances, improve an existing solver within a budget, obtain strong feasible solutions for a fixed batch, or distill transferable solving experience after a task. Do not use for requests that only ask for conceptual explanations without implementing or running a solver."
---

# ReActSolve

Use Codex's native reasoning, coding, file operations, and available tools to solve every supplied combinatorial optimization instance as well as possible within the user's budget, permissions, and deliverables.

This Skill is neither a COP textbook nor a fixed workflow. It supplies only cross-problem evidence, state, and result guardrails. Let the Agent choose algorithms, implementation, diagnostics, experiments, within-instance scheduling, and any portfolio from the current task while keeping the design simple.

## Core objective

Make every stage serve one purpose: solve each supplied instance better within the available budget. Judge strength per instance along three related dimensions that cannot substitute for one another:

- **Solution quality**: first ensure feasibility, then improve the objective under the evaluator's true comparison semantics; use unrounded gap when a trustworthy reference exists. Final gains on other instances or in an aggregate must not hide an instance that remains weak.
- **Efficiency in reaching that quality**: consider the full quality-time process, not only the final value or whether the budget was exhausted. Getting a better solution in the same time, or reaching the same quality sooner, is a real improvement. Notice waste caused by algorithmic complexity, implementation throughput, or unproductive computation.
- **Within-instance compute scheduling**: direct one instance's finite budget toward the solving behavior that is currently most valuable, making suitable choices among construction, improvement, exploration, restarts, alternative mechanisms or configurations, and stopping. This asks where time is spent; it differs from how fast the code runs and from distributing a shared batch budget across instances.

These dimensions define what to improve, not why a problem occurs, and they do not form a fixed weighted score. A change may primarily improve one dimension; use per-instance evidence to judge whether the others remain acceptable.

Make the first formal solver v1 a serious attempt that uses problem structure and per-instance differences to approach these objectives directly, not a placeholder baseline. Preserve every instance's best feasible solution and provenance while delivering a reproducible `FINAL_SOLVER`; they may come from different versions, but report them separately. Use a ReAct evidence loop to reduce uninformative trial and error, then distill only general insights that can improve a future v1 or accelerate problem diagnosis.

## Always true

- Let Codex choose general-purpose tools, libraries, and optimization components within the user's permissions; do not reimplement existing capabilities. Do not directly invoke or thinly wrap LKH or another problem-specific turnkey end-to-end solver in place of constructing the task solver.
- BKS, complete reference solutions, and numeric gap are not prerequisites. The evaluator's `compare` is authoritative, and internal search semantics must agree with it. `FINAL_SOLVER` must not read references, task trajectories, or stored instance answers.
- Judge quality, efficiency, and scheduling per instance. Aggregates must not hide any instance's problem; reaching a quality reference does not imply sound efficiency or scheduling. A working candidate may regress temporarily, but repair, route, or back off every regression before final recommendation.
- When the user gives no budget, use a default ceiling of 120 seconds per instance; an explicit user budget overrides it. A per-instance ceiling applies independently to every instance and is not a shared batch total. Local diagnostics may use shorter limits. In the formal baseline, every fresh round-closing run, and final validation, run each instance to the full ceiling unless it reaches a trustworthy target or proves optimality. Completion of a fixed iteration count, restart count, stage, or portfolio is not a reason to stop early. Never shorten the formal external ceiling uniformly merely to reduce validation cost.
- When batch instances are independent, use bounded concurrency by default if available CPU, memory, and solver-internal threading allow each worker comparable compute. Choose concurrency from actual resources; serialize only for a concrete resource or comparability reason, and do not oversubscribe blindly. Persist and surface each instance result as soon as it finishes rather than waiting for the full batch.
- Before every substantive action, state the current judgment, rationale, expected evidence, and weakening condition. Afterward, verify that the change executed under comparable conditions and update the evidence as `supported`, `weakened`, or `inconclusive`. A prediction guides action; it does not replace judgment or prove causality.
- Learned material expands the Agent's mechanism understanding, evidence-gathering ability, and solver-design ability. It is optional support, not a constraint or prescription. Continue with Codex's own intelligence when it is empty, unmatched, or inapplicable. Do not modify learned during solving.

## Round and delivery contract

- Apply this improvement-round contract only when the user requests continued iteration or a complete solve. If the user explicitly limits the task to first construction or a v1 baseline test, finish and report round 0, then stop without creating an improvement-round target, entering the loop, running distillation, or modifying learned.
- When this contract applies, establish or recover `round_target` before improvement begins. Starting from the highest closed round before the request, add 5 rounds by default or use the user's specified count. Historical rounds never offset the request, and round 0 does not count as an improvement round.
- **Completion gate:** enter closeout only after the actual closed round reaches this request's target round: run complete final validation, freeze `FINAL_SOLVER`, distill, then send the completed result. Before closeout, reread the start round, additional-round count, target round, and actual closed round. Continue solving when the target is missing, inconsistent, or unmet. Gap, optimality, and expected value do not change the round count. If an external hard constraint prevents continuation, report only an incomplete result.
- Full-batch coverage is evidence for closing round 0 or a locally qualified candidate, not a compute quota that requires finishing every attempt. Continuously interpret completed per-instance results while a batch runs. If partial evidence already shows that the current version cannot be promoted as-is or exposes an actionable defect, stop remaining runs that can no longer discriminate the current judgment, preserve the evidence, and return immediately to modification. Continue only when unfinished instances are still needed to distinguish a live judgment, and state why.
- Close a round only for a substantive candidate that passes this round's local promotion condition and has complete, comparable per-instance evidence. Freshly run every affected instance and every instance whose impact is uncertain; carry evidence forward only when the real executed path proves it unaffected. Seed changes, mere time extensions or repetitions, renaming, wrappers, and backoffs remain actions inside an open round.
- Repeated candidate failures, an open round without a promotable version, substantial elapsed wall time or context, and a self-imposed work limit for the current attempt are not external hard constraints. Keep `round_target` open across a resumable interruption and continue the same request after recovery; do not end it with a progress report.
- Deliver `FINAL_SOLVER`, per-instance incumbents, and provenance. Before the final response, reread final metrics and confirm that every instance can report feasibility, quality, trustworthy reference and gap, and time from process start until that run's final best solution first appeared. A missing instance or time-to-best means delivery is not ready. When incumbents differ, report both result sets. Keep end-to-end time, time after best, and stop reason in runtime for efficiency diagnosis rather than the final result table.

## Load on demand

- Read `protocol/setup.md` when starting a new task or constructing v1.
- Read `protocol/loop.md` when the user requests improvement after v1; also use it to recover an interrupted iterative request.
- After the completion gate passes, complete iterative solving and final validation are finished, and `FINAL_SOLVER` plus per-instance results are fixed, read and execute `protocol/distillation.md` before sending the final response. Do not distill for a first-construction-only request. An independent subagent must review the proposal; write only approved operations to learned.
- Read `references/runtime-records.md` when first creating, recovering, or repairing runtime records; do not repeatedly load format details during normal reasoning.
- Before constructing v1, search `learned/mechanism-insights.md`, `learned/evidence-methods.md`, and `learned/solver-design.md` according to the current problem. Revisit them later only when a real decision needs them.
- Run `scripts/metrics.py` when stable per-instance quality-time aggregation or comparison among v1, `FINAL_SOLVER`, and incumbents is needed.

Load only what the current decision requires.
