"""Run the 3 hard_split contrast arms sequentially: grounded mem / grounded nomem / selfjudge mem.
Validates arm names up front, then runs each via experiments.run_arm; stops on first failure."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.arms import get_arm   # validate names before any long run

ARMS = [
    ("grounded_induce_mem", "hard_split"),            # H2 mem (library accumulates)
    ("grounded_induce_nomem", "hard_split_nomem"),    # H2 nomem (library reset each round)
    ("selfjudge_induce_mem", "hard_split_selfjudge"),  # H3 self-judge curation
]

for arm, _ in ARMS:
    get_arm(arm)   # raises on a bad name -> fail fast, before any 20-min run
print("VALID arms; starting batch", flush=True)

for arm, run_id in ARMS:
    print(f"\n=== RUN {arm} -> {run_id} ===", flush=True)
    r = subprocess.run([sys.executable, "-u", "-m", "experiments.run_arm",
                        "--arm", arm, "--config", "configs/default_hard.yaml",
                        "--stream-len", "20", "--warm-k", "2", "--budget", "5",
                        "--run-id", run_id], cwd=str(ROOT))
    if r.returncode != 0:
        print(f"ARM FAILED: {arm} (exit {r.returncode}) -- stopping batch", flush=True)
        sys.exit(r.returncode)

print("\nALL_ARMS_DONE", flush=True)
