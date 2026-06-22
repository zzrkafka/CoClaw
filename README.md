# CoClaw

A **frozen** LLM agent that solves combinatorial-optimization (COP) routing instances
end-to-end and **reverse-induces a reusable, typed skill library** from its own
`(solution, exact-gap)` pairs. The model weights never change — what evolves is the
**skill library (code)**, curated by an **exact optimization objective** (LKH gap),
not by an LLM judge.

The scientific question: does that judgment **compound** (the library is genuinely
reused and its value rises across a stream of instances) or **collapse** (the library is
dead weight and all gains come from per-instance self-correction)? CoClaw is built to
*measure* this against an exact verifier rather than assert it.

## How it works

- **Solve loop** (`agent/solver.py`, `sandbox/`) — for each instance a multi-step
  **CodeAct** agent runs in a stateful sandbox: it observes, emits code actions, checks
  feasibility, measures the true objective, and submits a rolling-best tour. The sandbox
  **auto-repairs** invalid tours (`sanitize_tour`) and captures a **hang traceback**
  (SIGALRM stack dump) instead of silently failing.
- **Evolve loop** (`agent/evolve.py`) — **agentic induction**: a planner reflects on
  solve/teacher trajectories and proposes which typed operators to add; each is authored
  in a focused context — **breadth** (diverse personas, grounded-picked) then **depth**
  (a separate trace-reading *debugger* critic returns a minimal patch). Curation is a
  **step-wise grounded judge with partial credit**, and the judge is the **exact LKH gap
  on a dev set — never an LLM**.
- **Typed operators** (`skills/schema.py`) — the construct step is atomized into
  `diagnose/detect → order → build` so each atom earns independent credit, plus
  `local_search`, `repair`, `destroy`, and a `debug` skill. Leave-one-out **lesion**
  attribution (`analysis/lesion.py`) assigns each operator an exact marginal value.
- **Lessons + supervisor** (`agent/lessons.py`) — a minimal generic rule seed, plus
  **agent-distilled lessons** from grounded failures, plus a static `lint` for known
  anti-patterns. Specific bug rules are *not* hand-fed; the system rediscovers them.
- **Analysis / curation stack** (`analysis/`) — leave-one-out value (`lesion`),
  two-dimensional reuse×marginal pruning with merges (`curation`), quality-diversity
  niches (`qd`), a discriminative instance set (`discriminative`), and transfer/scale
  probes that yield `frac_general` (`probe`).

## Testbeds

- **`F`** (`configs/default.yaml`) — passes all three validity gates
  (toolkit-far, oracle-near, `LKH == exact`).
- **`F_hard`** (`configs/default_hard.yaml`) — the main testbed for *quality*
  compounding: inter-cluster visit order dominates cost, so generic self-discovery
  stalls (~28% gap) while the structural solution is near-optimal (~2%) — real headroom
  for an induced strategy to capture.

On `F_hard` the agent reverse-induces the intended strategy from teacher trajectories —
*detect cost clusters → brute-force the cheap directed cluster order → build a
cluster-contiguous tour → bounded 2-opt* — and reaches near-optimal gaps. The
**with-library vs no-library vs self-judge** comparison is the object of study;
`configs/default_hard_v2.yaml` enables the full curation/growth/probe stack (otherwise
config-gated off).

## Setup & run (WSL2 / Ubuntu)

CoClaw runs in WSL2 Ubuntu for POSIX sandbox isolation (`setrlimit`) and LKH-3 built
from source.

1. **Enable WSL2** (admin PowerShell, then reboot), if it isn't already:
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   Restart-Computer
   # after reboot:  wsl --update;  wsl --install -d Ubuntu
   ```
2. **Build the env** — venv + LKH-3, from the repo root inside Ubuntu:
   ```bash
   bash scripts/setup_wsl.sh
   source ~/.venvs/coclaw/bin/activate
   ```
3. **Add a key** — `cp .env.example .env`, fill in `DEEPSEEK_API_KEY`, then
   `set -a && source .env && set +a`.
4. **Run**:
   ```bash
   python -m experiments.gates --config configs/default.yaml          # validity gates
   python scripts/_smoke_judge.py                                     # offline judge smoke (no LLM)
   python -m experiments.run_arm --arm grounded_induce_mem \
       --config configs/default_hard.yaml --stream-len 4 --warm-k 2   # one arm (needs key)
   ```
   Run records (JSONL) land in `runs/<run-id>/` (`solve_record`, `evolve_record`,
   `metric_record`, `lessons.jsonl`, `skills/`). Output is block-buffered through the WSL
   pipe — monitor the record files, not stdout.

## Backend

Provider-agnostic; default **DeepSeek** (`configs/default.yaml → model:`). Swap to
Anthropic/OpenAI by editing that block and setting the matching API key. The model is
frozen — no training, no GPU; cost is tokens only.

## Layout

| path | what |
|------|------|
| `lcar/` | LCAR testbed generator + structural oracle |
| `agent/` | solver, evolve, featurize, lessons, playbook, harness, seeds, growth |
| `sandbox/` | stateful CodeAct executor + primitives |
| `skills/` | typed skill schema + library |
| `solvers/` | LKH-3, exact (Held–Karp), reference gaps |
| `analysis/` | lesion, curation, qd, discriminative, probe, curves |
| `experiments/` | arms, run_arm, gates, records |
| `llm/` | provider-agnostic client + prompt templates |
| `problems/` | problem-adapter interface (cross-type, kept behind the Stage-B boundary) |
