"""Arm configurations (spec section 11) -- mapping to hypotheses H1-H5.

| arm                     | curation  | reflect   | memory | tests                               |
| grounded_induce_mem     | grounded  | induce    | on     | main line (compounding upper bound) |
| selfjudge_induce_mem    | selfjudge | induce    | on     | H3 grounded vs self-judge (killer)  |
| grounded_induce_nomem   | grounded  | induce    | off    | H2 cross-instance compounding       |
| grounded_scoreonly_mem  | grounded  | scoreonly | on     | H3' induce vs score-only (vs ReEvo) |

Build order: science core first (grounded/self-judge, mem/no-mem), then vs prior work.
"""
from __future__ import annotations

from agent.evolve import Arm

ARMS: dict[str, Arm] = {
    "grounded_induce_mem":    Arm("grounded_induce_mem",    "grounded",  "induce",    "on"),
    "selfjudge_induce_mem":   Arm("selfjudge_induce_mem",   "selfjudge", "induce",    "on"),
    "grounded_induce_nomem":  Arm("grounded_induce_nomem",  "grounded",  "induce",    "off"),
    "grounded_scoreonly_mem": Arm("grounded_scoreonly_mem", "grounded",  "scoreonly", "on"),
}


def get_arm(name: str) -> Arm:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; choose from {list(ARMS)}")
    return ARMS[name]
