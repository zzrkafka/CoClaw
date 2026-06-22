"""Unattended run queue: v2 A/B (A1) -> self-judge control (B1) -> multi-seed variance (B2).

Continue-on-failure (one bad run does not kill the queue), per-run log under runs/<id>.log, final
manifest. B1/B2 use the BASELINE config (all gates OFF = original code path), so they produce the
headline-control data even if the v2 path (A1) has trouble.

Run inside the WSL venv with LKH on PATH, detached:
  cd /mnt/c/.../CoClaw
  PATH=~/.local/bin:$PATH nohup /home/zzr/.venvs/coclaw/bin/python -u scripts/_run_v2_queue.py \
      > runs/_v2_queue.log 2>&1 &
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.arms import get_arm   # validate arm names before any long run

V2 = "configs/default_hard_v2.yaml"
BASE = "configs/default_hard.yaml"

# (arm, config, seed, run_id)  -- order = priority
QUEUE = [
    ("grounded_induce_mem",   V2,   1234, "hard_split_v2"),         # A1: v2 A/B vs hard_split
    ("selfjudge_induce_mem",  BASE, 1234, "hard_split_selfjudge"),  # B1: H3 grounded-vs-selfjudge
    ("grounded_induce_mem",   BASE, 2234, "hard_mem_s2234"),        # B2: multi-seed variance...
    ("grounded_induce_nomem", BASE, 2234, "hard_nomem_s2234"),
    ("grounded_induce_mem",   BASE, 3234, "hard_mem_s3234"),
    ("grounded_induce_nomem", BASE, 3234, "hard_nomem_s3234"),
]

for arm, *_ in QUEUE:
    get_arm(arm)
print(f"QUEUE of {len(QUEUE)} runs validated; starting", flush=True)

results = []
for i, (arm, cfg, seed, rid) in enumerate(QUEUE, 1):
    log = ROOT / "runs" / f"{rid}.log"
    print(f"\n=== [{i}/{len(QUEUE)}] {rid}: {arm} seed={seed} cfg={cfg} -> {log.name} ===", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        try:
            rc = subprocess.run(
                [sys.executable, "-u", "-m", "experiments.run_arm",
                 "--arm", arm, "--config", cfg, "--stream-len", "20", "--warm-k", "2",
                 "--budget", "5", "--seed", str(seed), "--run-id", rid],
                cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=3 * 3600).returncode
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT(3h)"      # a hung run is killed so the queue moves on
            f.write("\n[queue] run exceeded 3h wall limit -> killed, continuing queue\n")
    dt = (time.time() - t0) / 60.0
    results.append((rid, status, round(dt, 1)))
    print(f"=== {rid}: {status} in {dt:.1f} min ===", flush=True)

print("\n==== QUEUE DONE ====", flush=True)
for rid, status, mins in results:
    print(f"  {rid:26} {status:12} {mins} min", flush=True)
