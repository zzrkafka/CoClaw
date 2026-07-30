# ReActSolve

## 1. Install the Skill in Codex

Codex loads user-level Skills from `$HOME/.agents/skills` and follows linked Skill directories. Run the commands for your operating system from the repository root.

### Windows

Run in PowerShell:

```powershell
$source = (Resolve-Path ".\react-solve").Path
$skillsRoot = Join-Path $HOME ".agents\skills"
$destination = Join-Path $skillsRoot "react-solve"

New-Item -ItemType Directory -Force -Path $skillsRoot | Out-Null

if (Test-Path -LiteralPath $destination) {
    Write-Host "react-solve is already installed at $destination"
} else {
    New-Item -ItemType Junction -Path $destination -Target $source | Out-Null
    Write-Host "Installed react-solve at $destination"
}
```

### macOS

Run in Terminal:

```bash
source_dir="$(cd react-solve && pwd)"
skills_root="$HOME/.agents/skills"
destination="$skills_root/react-solve"

mkdir -p "$skills_root"

if [ -e "$destination" ] || [ -L "$destination" ]; then
    echo "react-solve is already installed at $destination"
else
    ln -s "$source_dir" "$destination"
    echo "Installed react-solve at $destination"
fi
```

The Windows junction or macOS symbolic link keeps the installed Skill synchronized with the repository's `react-solve/` directory. Codex detects Skill changes automatically; restart Codex if `$react-solve` does not appear in a new task.

## 2. Prompt Template

Replace the bracketed values, open the intended project in Codex, and submit:

```text
/goal Use $react-solve to solve the combinatorial optimization task in the current workspace. Solve every supplied instance as well as possible.

Task mode: complete solve
Problem statement: [TASK.md or a concise problem description]
Instance directory: [instances/]
Reference values: [references.json; delete this line when unavailable]
Existing runtime: [runtime/<problem>/; delete this line for a new task]

Additional improvement rounds: [5 by default, or a user-specified count]
Per-instance time budget: [120s by default, or a user-specified ceiling]
Allowed tools and dependencies: [no additional restrictions; choose autonomously]

Requirements:
- Understand the problem and every instance, then construct a capable solver v1.
- Round 0 does not count toward the additional improvement rounds.
- Establish or recover this request's round_target before improvement begins.
- Complete exactly the specified number of additional improvement rounds. Do not stop early because of gap, optimality, expected value, failed candidates, context length, or a single execution limit.
- If an execution is interrupted, preserve the runtime state, keep the Goal active, and continue from the same round_target and open round.
- Preserve every instance's best feasible solution and provenance.
- After reaching the target round, run complete final validation, freeze FINAL_SOLVER, perform distillation, and only then send the final report.
- In the final report, list every instance's solution value, gap when available, and time to best.

Unless new permission, external information, or a material user decision is required, continue autonomously until every completion condition above is satisfied.
```
