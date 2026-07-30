# End-of-task distillation: help future Agents start stronger and repair faster

The purpose of distillation is to:

- bring a future Agent's first v1 closer to the best performance attainable within the task;
- help it understand and repair defects in quality, efficiency, or within-instance scheduling faster when problems remain;
- improve its ability to form solver-design judgments from the problem statement and per-instance evidence.

The third purpose runs through the first two: learned should help the Agent identify which instance differences affect solving, understand why, and use that understanding to compare algorithms, representations, implementations, search, and within-instance scheduling. It must not map features directly to actions or predetermine a shared approach, conditional configuration, or portfolio; let the Agent choose a sufficiently simple design from current evidence.

Learned is not a list of this task's successful changes, algorithm options, or parameters. Concrete changes are only evidence sources; distillation must extract general insights that will change a future Agent's judgment. If the conclusion is mainly “try the same algorithm next time and route it per instance,” distillation is still incomplete even when the wording includes conditions.

Read this only after the completion gate passes, final validation finishes, and `FINAL_SOLVER` plus per-instance results are fixed. First confirm that the final report, run references, and solver version agree. Return to the loop or repair the result records when deliverables are inconsistent or the solver still has an unresolved problem. Normal mixed experimental results are not delivery errors; retain them as support or counterevidence during distillation. Distillation itself must not change the solver or results.

## Work backward from a future Agent's decisions

Do not summarize “what happened” round by round or begin by selecting successful changes. Think from the two moments when a future Agent must decide, always asking how instance evidence changes the design:

1. Given only the statement and full instance batch, which per-instance facts and differences truly affect the solver, and why? What non-obvious prior insight would make the first design stronger in quality, efficiency, or scheduling?
2. When the solver has a problem, which instance behavior, code behavior, or quality-time evidence distinguishes the important explanations? What prior insight would help the Agent find the cause and form an effective repair sooner?

Use the current conversation and task context still retained by the Agent to recover intent and changes in judgment. Then read the instance analysis, round 0, v1, final per-instance results, and the episodes that materially changed a judgment or consumed substantial effort. In particular, compare how v1 understood the instances with later evidence showing that an instance difference mattered, was misread, or was omitted, and how that changed the solver. Successful choices, failed attempts, mixed responses, counterevidence from final validation, and initial decisions that were already correct can all be sources. First form candidates that may change a future decision, then inspect only the related code, diffs, runs, metrics, and provenance; do not recount the full trajectory for completeness.

Enumerate every independent candidate that is genuinely useful. Do not limit the result to one candidate or require every task to produce content. Learned may remain unchanged when there is no sufficiently new insight.

## Decide whether a candidate belongs in learned

Every candidate must withstand all of these judgments:

- **Purpose fit**: it must actually improve a future v1 or accelerate problem repair, while improving the Agent's ability to form design judgments from instance evidence. Being correct, evidence-backed, or cautiously worded is not enough.
- **Evidence to design**: it identifies which observable instance facts or run behavior matter, how they may affect the solving mechanism, and which design choices they make worth comparing. Do not store feature profiles unrelated to design or write “feature → fixed action.”
- **COP transferability**: the core claim can be reconsidered in another COP from observable structure, mechanism, or evidence. Specific problems, instances, and algorithms may appear in the supporting evidence; if removing those names leaves no useful insight, the abstraction remains insufficient. An entry need not apply to every COP, but it must not depend on instance identity or cached answers.
- **Complete evidence**: actively seek and include the strongest counterevidence, final validation, and alternative explanations. A historical incumbent proves only that one run found that solution, not that the mechanism is reliably stronger. When later comparable runs do not reproduce the result, narrow the claim and retain uncertainty instead of citing only the earlier win.
- **Genuine novelty**: search the main `SKILL.md`, setup, loop, and relevant learned material selectively. Do not restate an existing protocol guardrail, knowledge Codex already recalls reliably, or a synonymous entry merely because this task confirmed it again.

A mixed response may motivate a shared repair, conditional design, or portfolio, but it does not automatically imply “route per instance.” Learned should preserve why responses may differ, how to judge them, and which alternatives remain—not turn this task's instance branches into a future prescription.

## Write an insight that can be judged anew

As the content requires, connect instance evidence, its implications for solving behavior, and the resulting design judgment; also include a low-cost retest, credible alternative explanations, evidence source, and applicability boundaries when useful. These are not fixed fields or a required order. When evidence comes from one task or a small number of runs, let the future Agent see its scope and uncertainty.

Keep abstracting until the body no longer depends on a problem name, instance answer, bare parameter, final configuration, or task-specific code while retaining enough substance to affect a real judgment. Do not write round summaries, score stories, algorithm lists, unconditional prescriptions, deterministic “symptom → action” rules, causal conclusions from one fluctuation, or empty maxims such as “analyze before acting.”

Choose one entry point according to the future Agent's primary retrieval question:

- `learned/mechanism-insights.md`: understand structure, sources of complexity, or algorithm behavior;
- `learned/evidence-methods.md`: obtain or interpret low-cost discriminating evidence;
- `learned/solver-design.md`: construct or modify representations, data structures, search, scheduling, stopping, or a portfolio.

Place each insight at its primary entry point and keep it self-contained. Do not split one insight across all three files or require all three files to change.

## Propose, review, and write

Write candidates as independent operations in `runtime/<problem>/distillation-proposal.md`. Explain each operation in the clearest form: the target learned file, add/revise/retire intent, which purpose it serves, how it helps the Agent form design judgments from instance evidence, checkable supporting and weakening evidence, and the proposed body or diff. Do not use top-1 selection or a fixed quota.

After the proposal is complete, invoke one independent subagent. Give the reviewer the three purposes at the top of this page first, then have it read the proposal, cited task artifacts, final validation, the main Skill, relevant protocols, and relevant learned material. The reviewer must first decide whether each operation genuinely serves the purpose, especially whether it improves the Agent's ability to form solver-design judgments from instance evidence. It must not pass an operation merely because the content is correct, cites runs, or includes applicability conditions. Then check the evidence-to-design connection, COP transferability, complete evidence, and genuine novelty.

For every operation, require `pass`, `revise`, or `reject` with a brief reason that explicitly states how it would improve a future v1 or repair and how it strengthens the judgment from instance understanding to design. Use `revise` or `reject` when this cannot be explained. A bare `pass` is not a review. The reviewer should also identify an obviously omitted instance difference, initial choice, failure lesson, counterexample, or judgment transition, but must not draft the entry for the main Agent.

Revise the proposal from the feedback and return affected operations to the reviewer. Write every operation that ultimately receives `pass`, and do not write rejected or unresolved operations. When no candidate qualifies, explicitly record that learned remains unchanged. Append the per-operation verdicts and actual write results to the proposal so one concise, checkable audit record remains. If the user requires approval before writing, wait after reviewer approval; otherwise apply the approved operations and summarize them briefly in the final response.
