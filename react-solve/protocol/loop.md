# Iterative improvement: keep making the solver stronger

The loop has one focus: use the real code and per-instance results to keep improving solution quality, efficiency in reaching that quality, and within-instance scheduling. The unit of improvement is a solver capability, not a code change or an arbitrary tiny objective decrease; rounds, records, and experiments only support better solutions.

Full-batch evaluation is only for closing a candidate that remains eligible for promotion, not a quota that every attempt must finish. Interpret instances as they complete; when partial evidence already shows that the current version cannot be promoted as-is or exposes an actionable defect, stop remaining runs that can no longer distinguish the current judgment, preserve the results, and return to modification.

When starting or resuming an iterative request, use the main `SKILL.md` and `references/runtime-records.md` to create or recover this request's round target. Confirm the stable solver, working candidate, evaluator, per-instance metrics, and incumbents. The round count determines how long to keep improving, not how to design the solver.

## Understand capability gaps from instance evidence

Keep the complete per-instance state in runtime rather than requiring all detail to occupy the main context. Maintain a complete attention set containing the special instances and unknowns emphasized by setup, weaknesses exposed by the baseline, and later problems. Let the main Agent deeply inspect a small set of the weakest, most informative, or path-distinct instances; an instance does not disappear or become solved merely because it is not loaded into the current context.

Read the implementation alongside current quality, gap when available, time-to-best, real incumbent trajectories, and stopping behavior. Determine what the target instances actually execute, which important decisions the solver can revise, where the budget goes, and whether each mechanism affects the incumbent. Form the most valuable current judgment from that evidence without assuming a closed set of causes.

An early final best followed by a long flat tail is evidence that search capability or within-instance scheduling remains unexplained; exhausting the budget does not prove effectiveness. As soon as evidence shows that the design cannot revise quality-dominating decisions, a key path is unreachable at target scale, or the remaining budget mainly repeats ineffective work, revisit the base implementation and design assumptions without waiting for repeated failures.

Quality, efficiency, and within-instance scheduling are improvement objectives, not cause categories, and need not be covered mechanically in every round. Reaching a quality reference closes only the quality issue; obvious efficiency or scheduling problems remain. Learned is only a reference; rely on Codex's own intelligence when it is empty or inapplicable. When instances respond reliably differently to mechanisms, repair the shared approach or form a simple portfolio routed by observable properties.

## Modify and validate with ReAct

Before acting, record the current judgment, rationale, expected evidence, and weakening condition according to `references/runtime-records.md`; update it immediately after observing the result rather than reconstructing it from memory at task end. Then modify the solver directly or first run a low-cost experiment that distinguishes an important explanation.

Test first on the weakest, most relevant, or most informative instances, adding controls that distinguish different responses as needed. Compare quality-time behavior at a common cutoff under materially comparable conditions; failure to beat the baseline's long-budget final value in a short test is not enough to reject a candidate.

An arbitrary strict improvement is not enough for promotion. Local evidence must show that the candidate materially reduces the target problem or unlocks a previously missing, actually reachable solver capability with a credible path to further improvement. Otherwise continue understanding and modifying within the same open round. When a result is poor, confirm that the change executed, the relevant path was reached, and the comparison was comparable; then use the run to revise the judgment instead of mechanically changing parameters.

Within the external time ceiling, continue search that has a chance to improve the incumbent; repeating exhausted fixed candidates or merely extending time is not an improvement. Completion of a fixed iteration, restart, stage, or portfolio cannot end a formal run. Except after reaching a trustworthy target or proving optimality, continue every instance to the external ceiling.

Keep the stable recommended version separate from the working candidate. A candidate may temporarily regress on some instances; use mixed responses to refine the direction, applicability boundary, or routing rather than accepting by average or immediately discarding the whole direction. Whenever the evaluator confirms a better feasible solution, update that instance's incumbent and provenance immediately.

## Close a round with per-instance evidence

After local promotion, determine impact scope from the actual diff and executed paths. Every instance that may reach the changed path is affected; do not decide scope from the locally tested instances or benchmark names. Freshly validate affected instances and any instance whose impact cannot be ruled out with the full ceiling. Carry forward the latest comparable formal evidence only when code paths prove an instance unaffected.

Whenever complete formal per-instance coverage is available, invoke one subagent to review every instance and compare those with comparable scale, structure, constraints, or actual code paths. It first scans the full batch, then examines incumbent curves, improvement timing, end-of-budget trends, and available telemetry for anomalous or repeatedly unresolved instances. It reports evidence, comparability, and explanations that remain unresolved, but does not decide causes or design the solver. Keep every positive gap when available. The main Agent must read the result and update the complete attention set before closing the round; a candidate already disqualified during the batch does not trigger this review.

Before closing, revisit the complete attention set and decide for every known issue whether the candidate affects it, whether existing evidence remains valid, and what comes next. Issues touched by the candidate require direct evidence. Unresolved quality, efficiency, or scheduling issues may carry forward, but must remain visible rather than being cleared by improvements elsewhere. Continue updating the judgment as full-batch results arrive instead of waiting until the batch ends to notice disqualification.

Only a candidate that creates and validates a substantive solver capability closes a round. Parser, evaluator, record, or evidence-validity repairs may be necessary to continue solving but do not close an improvement round; repetition, time extension, renaming, wrapping, backoff, and report generation also remain actions within the open round.

Replace the stable recommended solver only after every per-instance regression is repaired, reliably routed, or backed off. Continue the next round from all unresolved instance evidence rather than executing a predetermined algorithm route.

After the completion gate passes, fully validate `FINAL_SOLVER` and run the same per-instance subagent review. If it exposes no issue requiring repair, freeze the solver, fix the per-instance results and incumbents, and proceed to `distillation.md`; otherwise return to the loop.
