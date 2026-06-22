"""Integration: run the FULL run_arm orchestration with EVERY §1-§6 switch ON, using a FAKE LLM
(no API). Catches wiring errors the per-feature smokes can't: seed -> solve -> evolve -> lesion ->
leave-one-in -> prune -> QD -> discriminative -> probe -> V(t) -> metric, with plans/playbook/debug
prompts rendered. Asserts it completes and the new metric fields are emitted. Needs LKH (refs)."""
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.run_arm as RA            # noqa: E402
from experiments.records import RunRecorder  # noqa: E402


class _Usage:
    prompt_tokens = 0
    completion_tokens = 0
    calls = 0
    def usd(self, model):  # noqa: D401
        return 0.0


class FakeLLM:
    """Deterministic stand-in: returns a valid CodeAct action for solves, and EMPTY ops for the
    reflect planner (so no induction) -- the library stays the seed strategy, which still exercises
    the entire analysis pipeline downstream."""
    def __init__(self, model_cfg, budget_usd=5.0):
        self.model = model_cfg.get("model", "fake")
        self.usage = _Usage()

    def complete(self, msgs, **kw):
        text = " ".join(c for _, c in msgs) if isinstance(msgs, list) else str(msgs)
        if "FINAL_TOUR" in text:                       # CodeAct solve prompt
            return "```python\nFINAL_TOUR = list(range(n))\nDONE = True\n```"
        if "ops" in text or "DECOMPOSE" in text:       # reflect planner -> no ops
            return '{"analysis": "none", "ops": []}'
        return "{}"


RA.LLMClient = FakeLLM                                  # inject the fake into the orchestrator

cfg = yaml.safe_load((ROOT / "configs" / "default_hard.yaml").read_text())
# turn EVERYTHING on
cfg["harness"] = {"inject_plans": True, "use_playbook": True, "debug_recovery": True}
cfg["library"] = {"seed_strategy": True, "fill_holes": False}   # fill_holes needs real authoring
cfg["curation"] = {"prune": True, "tenure_k": 3, "prune_mv_eps": 0.001, "prune_reuse_max": 0.34,
                   "leave_one_in": True, "qd_niche": True}
cfg["discriminative"] = {"enabled": True, "persist_gap_matrix": True, "pool_k": 4,
                         "reselect_every": 1, "anchor_frac": 0.34}
cfg["problem"] = {"family": "F"}

RUN = "_smoke_e2e"
shutil.rmtree(ROOT / "runs" / RUN, ignore_errors=True)
out = RA.run_arm("grounded_induce_mem", cfg, stream_len=2, n=cfg["instances"]["n_main"],
                 budget_usd=1.0, warm_k=1, run_id=RUN, verbose=True)
print("run_arm returned:", {k: out[k] for k in ("n_skills", "usd")})

# the metric record must exist and carry the new fields
import json
metrics = [json.loads(l) for l in (ROOT / "runs" / RUN / "metric_record.jsonl").read_text().splitlines()]
assert metrics, "no metric records emitted"
last = metrics[-1]
for field in ("curation", "slot_fill_ratio", "library_value", "frac_general"):
    assert field in last, f"metric record missing {field}"
assert "qd" in last["curation"], "curation summary should include the QD result"
print(f"metrics rounds={len(metrics)} | last: n_active={last['n_active_skills']} "
      f"slot_fill={last['slot_fill_ratio']} V(t)={last['library_value']} "
      f"curation_enabled={last['curation'].get('enabled')}")
shutil.rmtree(ROOT / "runs" / RUN, ignore_errors=True)
print("OK e2e: full run_arm pipeline runs with all switches on; new metric fields emitted.")
