# TSP

## Semantic boundary

Use this family only for a minimum-cost symmetric closed tour that visits every declared node exactly once. Reject optional visits, multiple vehicles, side constraints, asymmetric costs, or another objective.

Normalized data contains ordered unique `data.node_ids`. When `data.representation` is `coordinates_2d`, coordinates are in `data.coordinates`; when it is `explicit_matrix`, the complete symmetric matrix is in `data.distance_matrix`. Emit `{"tour": [...]}` with every ID exactly once and no repeated start; closure is implicit. EUC_2D distance is `floor(hypot(dx, dy) + 0.5)`, never Python `round()`.

The solver interface supplies CLI and artifact rules. Spend this document's attention on a fast, strong tour search. Never branch on filenames, benchmark identity, reference values, gaps, BKS, or optima.

## Default strong search

Unless instance evidence supports a better design, implement one coherent iterated local search:

1. Build a bounded candidate graph for every node.
2. Create two or three inexpensive but structurally different initial tours.
3. Reach a fast 2-opt and short-segment relocate/Or-opt local optimum without repeatedly rescanning the whole candidate neighborhood after each move.
4. On a nontrivial medium/large-instance budget, intensify the resulting basin with a legal variable-depth alternating exchange, then return to the cheaper neighborhoods after every accepted strong move. Restricted 3-opt is a useful bridge and probe, but is not the default terminal neighborhood for a high-quality regime unless shortened evidence shows it meets the requested tail target.
5. Use perturb → full re-descent cycles while protecting the global incumbent; escalate or change the intensifier when many distinct basins reproduce the same quality floor.

This is a policy, not a mechanism checklist. A smaller complete implementation is better than several nominal operators. But construction plus 2-opt/Or-opt and kicks is only a fallback unless shortened evidence shows it already meets the requested quality regime. For medium or large Euclidean instances, repeatedly reaching distinct 2-opt/Or-opt minima without a stronger exchange is an incomplete effectiveness policy, even when every implemented neighborhood is locally exhaustive.

### Candidate graph

For coordinates, prefer an available `scipy.spatial.cKDTree`; otherwise implement a correct sparse spatial fallback such as a balanced k-d tree or multi-projection/grid candidates. For explicit matrices, select low-cost entries per row without geometric assumptions. Do not install dependencies.

Start around 24–40 neighbors per node and adapt to size, memory, build time, and sampled candidate recall. Symmetrize candidates, retain every node, and always add current-tour neighbors. Approximate candidates are acceptable when their recall is measured; an unprocessed suffix or empty rows are not. On a small random sample, compare candidate edges with an exact row scan. Widen or repair when strong missed edges remain common.

Use dense distances only when both time and memory clearly fit (typically small/medium instances). Large coordinate instances need on-demand or sparse cached distances; candidate construction must not consume the search budget merely to prove exact nearest-neighbor order.

### Initial tours

Publish a trivial feasible tour immediately, then improve the starting pool. Useful contrasting constructors are nearest-neighbor/insertion from dispersed starts and degree-2 multi-fragment greedy construction. A multi-fragment builder must enforce degree at most two, forbid a premature cycle, and finish one Hamiltonian cycle; abandon it for the safe tour if its completion becomes doubtful.

Deduplicate tours by undirected edge set. Give each surviving construction a short equal descent probe and retain it only if its objective or early gain rate is competitive. Avoid dozens of cheap starts that steal time from local search.

### 2-opt and Or-opt descent

For 2-opt replacing `(a,b)` and `(c,d)`, use
`d(a,c) + d(b,d) - d(a,b) - d(c,d)`. Exclude adjacent edges and whole-cycle reversal. Maintain a correct position map after every reversal.

Use complete 2-opt on small instances when it fits. Otherwise enumerate candidate endpoints from both incident tour edges and default to first improvement with a persistent/resumable cursor or correct don't-look scheme; reactivate affected vertices after a move. A pass is complete only when every active edge has been reconsidered against its current candidates since relevant state changes. Repeatedly restarting first improvement at index zero is an incompleteness bug. Recomputing the global best candidate move after every accepted move is usually a throughput bug in Python when it leaves too little time for independent basins or stronger exchanges; retain it only where measured scan cost is small.

Relocate contiguous segments of length 1–3 into candidate insertion edges. Exclude overlapping/identity moves, compute the cyclic remove/insert delta including all boundary edges, update positions, then re-enter 2-opt. If Or-opt repeatedly receives time but proposes or accepts nothing, inspect enumeration and scheduling before declaring the neighborhood weak.

After every accepted move during Debug, compare incremental objective with a full tour recomputation. Always fully rescore before promoting a global incumbent.

### Stronger intensification

Treat a verified variable-depth exchange as part of the default completed search when 2-opt/Or-opt plateaus above a demanding quality target and useful time remains. Candidate-restricted 3-opt may precede it, but do not wait for more identical 3-opt moves or kicks to supply an improvement that the active neighborhood cannot express.

A practical 3-opt enumerator anchors one removed tour edge, reaches the other removed edges through bounded candidate endpoints, and evaluates every legal reconnection pattern it claims. Exclude adjacent/overlapping cuts, unchanged tours, and patterns equivalent to an already exhausted 2-opt move. Keep enumeration near candidate scale rather than cubic all-edge search. Accept only a negative full delta, rebuild or update the cyclic order unambiguously, fully rescore, then re-enter 2-opt and Or-opt because the strong move invalidates their local-optimum state.

For variable depth, alternate removed and added edges, maintain degree feasibility and one recoverable Hamiltonian path throughout every partial state, and promote only a legal single-cycle closure after full rescoring. Keep several high-potential partial paths when a greedy choice could destroy a later closure. A fixed reconnection, random three-edge shuffle, or sequence of ordinary 2-opt moves is not truthful 3-opt/LK evidence.

Do not blindly accept the first tiny positive closure, but do not exhaust the whole beam to prove a global best move either. Within a bounded search window, compare legal closures by fully verified gain and accept immediately only when an aspiration rule says the gain is material; otherwise apply the best verified closure found before the window ends. Candidate width, beam and depth must respond to measured closure coverage and gain per second. Exact nearest-neighbor recall only proves candidate construction correctness; it does not prove that the candidate set exposes useful strong exchanges.

Separate direct gain on the current tour from gain of the protected incumbent. Many accepted deep exchanges with little incumbent gain and high wall time are an effectiveness plateau, not success. Reallocate only after repeated comparable low-return windows in the same basin and enough legal closures to make the observation meaningful; reacting to one noisy window can oscillate between over-searching and over-kicking. Preserve a minimum strong-search opportunity, then widen candidate reach, change the partial-path selection rule, restart from another retained construction, or escape according to the observed failure.

Report strong-neighborhood proposals, legal patterns reached, accepted state changes, fully rescored incumbent gain, and time separately. Zero accepted moves on forced microcases is a correctness/reach failure. Many independently re-descended basins with zero incumbent gain is evidence to strengthen or repair intensification, not evidence that simple perturbation needs more budget.

### Perturb and re-descend

The default escape is a seeded double-bridge or another verified non-2-opt kick followed by the same complete descent. Choose four well-separated cyclic cuts, reject degenerate short segments, reconnect into one tour, and verify that its undirected edge set differs materially from the source. A reversal or ordinary 2-opt under another name is not a kick.

Keep two states: a protected best tour and the current local optimum. A perturbed/re-descended tour may become the new current basin even when it does not beat the best; only a fully rescored improvement replaces the best. Vary cut positions and strength when repeated kicks return duplicate basins. If many independently different basins yield no gain, escalate the neighborhood or restart from another retained construction instead of repeating a sterile kick forever.

Restricted 3-opt is the usual next intensifier when 2-opt/Or-opt are exhausted. Unit-test every reconnection pattern actually claimed; prefer verified 3-opt plus iterated kicks over a fake LK label.

## Allocate time from measured return

First make every batch entry survivable. Then allocate active-time slices by instance size, remaining limit, current objective, recent gain per second, and uncertainty. Per-instance timing excludes pauses while other entries run.

Use ceilings rather than rigid quotas:

- feasible construction and publication reserve should be small;
- candidate building must leave most time for improvement;
- initial-basin probes should quickly eliminate weak starts;
- perturb/re-descent or a stronger exchange should receive the majority once the descent floor works;
- finalization time is never spendable search time.

Measure proposals, accepted state changes, incumbent gains, last-improvement time, distinct basins, and time by mechanism. Reallocate when a mechanism's comparable windows have much lower gain/information rate. A fixed pass count, proposal cap, or failed kick ends a chunk, not the instance.

Larger fixed-seed time must preserve the shorter run's incumbent. Early exit is credible only after the applicable descent is complete and multiple viable basin-changing attempts show measured stagnation. Otherwise run until the reserved finalization boundary and report a budget cutoff.

## Diagnose quality failures causally

Treat a valid but poor tour as a Debug target. Use the cheapest probe that distinguishes these cases:

- weak initial objective before descent: compare constructors after equal short descent;
- 2-opt gains stop suspiciously early: sample exact non-candidate edges, inspect edge-cursor coverage, and force a known improving reversal;
- Or-opt/3-opt has zero state changes: test exact deltas and forced moves on tiny tours;
- many kicks but no incumbent gain: check whether basins are actually distinct and receive complete re-descent;
- time rises without more proposals or basin changes: inspect Python hot loops, cursor restarts, distance cost, and oversized preprocessing;
- small instances remain poor: use complete neighborhoods and more exhaustive re-descent before adding scale-oriented machinery;
- large instances remain poor: inspect candidate recall, search throughput, and whether preprocessing or one basin monopolizes time.

When a shortened controller probe shows a large quality deficit, fix the earliest causal layer that is weak: construction, descent completeness, candidate recall, basin change, or time control. Do not respond by adding unrelated operator names.

## Correctness and truthful telemetry

A kick changes only a working copy. A timed-out, infeasible, or numerically inconsistent state never replaces the protected incumbent. Recompute the closing edge and full objective for every published tour.

In optional telemetry, report only policy-defining construction, descent, and escape/intensification mechanisms. A reached mechanism must produce a valid proposal or state expansion, not a wrapper call. Attribute accepted moves and incumbent gain to the mechanism that produced the fully rescored improvement. Report backend, candidate width/coverage sample, proposals, state changes, gains, distinct basins, time, and cutoffs when available.

The independent validator is authoritative. A mechanism profile explains why search worked or stalled; it cannot turn a weak tour into a successful result.
