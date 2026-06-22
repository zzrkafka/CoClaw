# CoClaw

A **frozen** LLM agent that solves combinatorial-optimization (COP) routing instances
end-to-end and **reverse-induces a reusable library of typed operators**, composed into
**strategies**, from its own `(solution, exact-gap)` pairs. The model weights never
change; what evolves is the **library**, curated by an **exact optimization objective**
(LKH gap), not by an LLM judge.

> **Terminology.** A **skill** is a top-level **strategy** — an executable **`scaffold`**
> composing **operators** (the typed code atoms: `construct` / `order` / `local_search` /
> `repair` / `destroy` / `diagnose` / `debug`), plus a declarative **`plan`** (whose
> injection is config-gated — see below). **Lessons** are cross-cutting rules. The code
> keeps the identifiers `Skill` / `SkillLibrary` for any library entry.

The scientific question: does that judgment **compound** (the library is genuinely
reused and its value rises across a stream of instances) or **collapse** (the library is
dead weight and all gains come from per-instance self-correction)? CoClaw is built to
*measure* this against an exact verifier rather than assert it.

## How it works

Two loops that must not be conflated: the **harness** — a *fixed*, problem-agnostic
**inner** engine that drives one instance to a submitted solution — and the **evolve
loop**, the **outer** loop that grows and curates the library across instances. The
harness is frozen *code*; what evolves is the library, not the engine.

- **Harness — the fixed inner engine** (`agent/solver.py`, `agent/harness.py`, `sandbox/`)
  — for each instance a multi-step **CodeAct** agent runs in a stateful sandbox: observe →
  act → check feasibility → measure the true objective → **recover** on error → track the
  rolling-best tour → submit. The sandbox **auto-repairs** invalid tours (`sanitize_tour`)
  and captures a **hang traceback** (SIGALRM stack dump) instead of silently failing; the
  harness — not the agent — measures the exact LKH gap (the agent never sees it).
- **Evolve loop** (`agent/evolve.py`) — **agentic induction**: a planner reflects on
  solve/teacher trajectories and proposes which typed operators to add; each is authored in
  a focused context — **breadth** (diverse personas, grounded-picked) then **depth** (a
  separate *debugger* critic reads the execution evidence and returns a minimal patch).
  Curation is a **step-wise grounded judge with partial credit**, and the judge is the
  **exact LKH gap on a dev set — never an LLM**.
- **Atomized operators & lesion credit** (`skills/schema.py`, `analysis/lesion.py`) — the
  monolithic constructor is split into `detect → order → build` so each **operator** is
  judged and credited independently (alongside `local_search`, `repair`, `destroy`,
  `debug`). Leave-one-out **lesion** attribution then assigns each operator an exact
  marginal value; dead weight is pruned.
- **Lessons + supervisor** (`agent/lessons.py`) — a minimal generic rule seed, plus
  **agent-distilled lessons** from grounded failures, plus a static `lint` for known
  anti-patterns. Specific bug rules are *not* hand-fed; the system rediscovers them.
- **Analysis / curation stack** (`analysis/`) — leave-one-out value (`lesion`),
  two-dimensional reuse×marginal curation that prunes dead weight and flags merges
  (`curation`), quality-diversity niche dedup (`qd`), a discriminative instance set
  (`discriminative`), and post-hoc transfer/scale probes that yield `frac_general`
  (`probe`).

> **Config-gated layer.** An explicit controller **playbook**, injection of a strategy's
> **`plan`** into the solve context, and a **debug**-operator recovery seam
> (`agent/playbook.py`, `agent/harness.py`) are implemented but **off by default** in
> `configs/default_hard.yaml` (the profile behind the current results) and **on** in
> `configs/default_hard_v2.yaml`, so old-vs-new can be run as a clean A/B.

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
| `skills/` | skill (strategy) + operator schema; library |
| `solvers/` | LKH-3, exact (Held–Karp), reference gaps |
| `analysis/` | lesion, curation, qd, discriminative, probe, curves |
| `experiments/` | arms, run_arm, gates, records |
| `llm/` | provider-agnostic client + prompt templates |
| `problems/` | problem-adapter interface (cross-type, kept behind the Stage-B boundary) |
