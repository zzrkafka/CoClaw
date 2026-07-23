# Evolution tools

Use `scripts/evolve.py` only after the Agent has selected the corresponding action.

## Preserve or stage a family

```text
python scripts/evolve.py snapshot --solve-root ../solve-cop --family tsp --output <snapshot-dir>
python scripts/evolve.py stage --snapshot <snapshot-dir> --output <candidate-dir>
```

`snapshot` captures only the target manifest, guide, parser, validator, fixtures, and activation evidence. `stage` creates a writable candidate copy. Edit the candidate, never canonical files.

## Prepare a blinded phenotype call

```text
python scripts/evolve.py request \
  --side candidate --family tsp --family-root <candidate-dir>/solve-cop \
  --manifest <anonymous-batch.json> --time-sec 120 --tokens 80000 \
  --output <phenotype-request.json>
```

Give the request to one fresh Agent and instruct it to execute `solve-cop`. Do not attach references, thresholds, prior solvers, or the other side.

If parser, validator, or objective semantics changed, pass the same staged evaluator root to both phenotype requests with `--evaluator-root`; `compare` rejects results produced by different validator bytes.

## Compare mature results

```text
python scripts/evolve.py compare \
  --champion <champion-run-result.json> \
  --candidate <candidate-run-result.json> \
  --output <comparison.json>
```

Add `--references <objective-references.json>` only in the evolve context. The tool checks matched family, batch bytes, validator bytes, instances, objective sense, and validation before computing per-instance changes. The evolve Agent is responsible for giving both solve Agents comparable authoring budgets.

## Accept or reject

```text
python scripts/evolve.py decide \
  --solve-root ../solve-cop --family tsp \
  --candidate-root <candidate-dir>/solve-cop \
  --comparison <comparison.json> \
  --decision accept --reason "<evidence-based reason>" \
  --output <decision.json>
```

`accept` replaces only the target family assets and rolls back if copying fails. `reject` records the decision without touching canonical files.
