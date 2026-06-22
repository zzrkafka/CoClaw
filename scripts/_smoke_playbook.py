"""Smoke: §1.5 top-level controller playbook -- gating + prompt injection + the 3 artifacts.
No LLM / no LKH."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.playbook import (DEBUG_SKILL_SPEC, HARNESS_PLAYBOOK,            # noqa: E402
                           PER_TYPE_SKILL_TEMPLATE, harness_playbook)
from llm.prompt_loader import render_messages                             # noqa: E402

# 1) gating: off by default, on only when harness.use_playbook
assert harness_playbook(None) == "", "playbook must be off when no cfg"
assert harness_playbook({}) == "", "playbook must be off by default"
assert harness_playbook({"harness": {"use_playbook": False}}) == ""
assert harness_playbook({"harness": {"use_playbook": True}}) == HARNESS_PLAYBOOK

# 2) minimal-seed discipline: the general playbook must NOT bake in domain-specific answers
low = HARNESS_PLAYBOOK.lower()
for banned in ("cluster", "2-opt", "asymmetric", "lkh"):
    assert banned not in low, f"playbook leaks a problem-specific answer: {banned!r} (minimal-seed)"

# 3) prompt injection is gated on the rendered text
def render(pb):
    return "\n".join(c for _, c in render_messages(
        "codeact_system", n=60, c_stats="x", x_preview="[]", skills_block="(none)",
        plans_block="", playbook_block=pb, max_steps=8, eval_budget=None))

assert "general playbook" not in render("").lower(), "empty -> baseline prompt unchanged"
assert "general playbook" in render(HARNESS_PLAYBOOK).lower(), "on -> playbook appears in prompt"

# 4) the §1.5 split artifacts exist with their documented shape
assert set(PER_TYPE_SKILL_TEMPLATE) == {"dispatch", "strategize", "pitfalls", "debug"}
assert set(DEBUG_SKILL_SPEC) == {"inputs", "output", "grounded_by"}

print("OK §1.5: playbook gated off by default; minimal-seed (no domain answers); 3 artifacts present.")
