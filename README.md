# CoClaw — Pilot v2 (LCAR)

End-to-end LLM agent that solves one routing instance at a time, and reverse-induces a
**reusable typed skill library** from its own (solution, exact gap) pairs. The model is
**frozen**; what evolves is the skill library (code), not weights. The scientific question:
does that judgment **compound** (library genuinely reused, value rises) or **collapse**
(library is dead weight, gains all come from per-instance self-correction)?

## Status (2026-06-21)

Testbed + full agent stack built and exercised end-to-end. Current architecture:

- **Testbeds**: `F` (`configs/default.yaml`) passes all three validity gates; **`F_hard`
  (`configs/default_hard.yaml`)** is the main testbed for *quality* compounding (cluster-visit
  order dominates cost; self-discovery stalls ~28%, structural solution ~2%).
- **Solve loop**: per-instance multi-step CodeAct in a stateful sandbox (`sandbox/`,
  `agent/solver.py`). The sandbox **auto-repairs** invalid tours (`sanitize_tour`) and captures a
  **hang traceback** (SIGALRM stack dump) instead of just failing.
- **Evolve loop** (`agent/evolve.py`): **agentic induction** — a planner picks which typed skills
  to add; each is authored in a focused context with **breadth (diverse personas, grounded-picked)
  then depth (a separate trace-reading *debugger* critic → minimal patch)**. Curation is a
  **step-wise grounded judge with partial credit** (construct scored with local-search off;
  local-search judged for regression; repair/destroy/diagnose on probation) — the judge is the
  **exact LKH gap on a dev set**, never an LLM.
- **Lessons + supervisor** (`agent/lessons.py`): a minimal generic rule seed plus
  **agent-distilled lessons** from grounded failures, plus a static `lint` that flags known
  anti-patterns. Specific bug rules are *not* hand-fed — the system rediscovers them.

In progress (Stage 2b): two-tier quality-diversity library (archive → active with counterfactual
promotion), full lesson lifecycle (attribution voting + context retrieval), diversity/collapse
monitoring.

## Running (WSL2)

```bash
cd /mnt/c/Users/81309/OneDrive/Desktop/Idea/CoClaw && source ~/.venvs/coclaw/bin/activate
python -m experiments.gates --config configs/default.yaml          # validity gates
python scripts/_smoke_judge.py                                     # offline judge smoke (no LLM)
python -m experiments.run_arm --arm grounded_induce_mem \
    --config configs/default_hard.yaml --stream-len 4 --warm-k 2   # one arm (needs DEEPSEEK_API_KEY)
```
Run records (JSONL) land in `runs/<run-id>/` (`solve_record`, `evolve_record`, `metric_record`,
`lessons.jsonl`, `skills/`). Output is block-buffered through the WSL pipe — monitor the record
files, not stdout.

## Setup (WSL2 / Ubuntu)

This project runs in WSL2 Ubuntu (POSIX sandbox isolation + LKH-3 from source).

1. Get WSL2 working (admin PowerShell, then reboot):
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   Restart-Computer
   ```
   After reboot: `wsl --update; wsl --install -d Ubuntu` (creates a UNIX user).
2. Inside Ubuntu, from the project root:
   ```bash
   cd /mnt/c/Users/81309/OneDrive/Desktop/Idea/CoClaw
   bash scripts/setup_wsl.sh
   ```
3. Provide your DeepSeek key: `cp .env.example .env`, edit, then `set -a && source .env && set +a`.

## Backend

Provider-agnostic; default **DeepSeek** (`configs/default.yaml` → `model:`). Swap to
Anthropic/OpenAI by editing that block and setting the matching API key. The model is frozen
(no training, no GPU); cost is tokens only.
