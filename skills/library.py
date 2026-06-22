"""Typed skill library with versioned lifecycle + reuse accounting (spec section 7.2).

AutoSkill-style lifecycle (add/modify/merge/prune) but curated by the EXACT objective
(section 9), not an LLM judge. The ReuseStats are the scientific core: they let us measure
whether the library compounds (reuse value rises) or collapses (dead weight).

Note: the lifecycle methods take a fully-materialized candidate Skill (the evolve layer
builds it from an LLM op); modify/merge deprecate the source(s) and add the candidate as a
new version with lineage. This is the practical form of the section 7.2 signatures.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .schema import Skill


class SkillLibrary:
    def __init__(self, store_dir: str | Path | None = None):
        self.skills: dict[str, Skill] = {}
        self.version: int = 0
        self._counter: int = 0
        self._instances_per_skill: dict[str, set] = {}
        self._seed_state: dict | None = None
        self.store_dir = Path(store_dir) if store_dir else None
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)

    # -- ids / persistence -----------------------------------------------------
    def _new_id(self, kind: str, name: str | None = None) -> str:
        self._counter += 1
        base = (name or kind).strip().lower().replace(" ", "_")
        return f"{base}_{self._counter}"

    def _persist(self, skill: Skill) -> None:
        if not self.store_dir:
            return
        (self.store_dir / f"{skill.id}.py").write_text(skill.code)
        with (self.store_dir / f"{skill.id}.jsonl").open("a") as f:
            f.write(json.dumps({
                "library_version": self.version, "status": skill.status,
                "version": skill.version, "parent_ids": skill.parent_ids,
                "reuse": skill.reuse.__dict__,
            }) + "\n")

    def _bump(self) -> int:
        self.version += 1
        return self.version

    # -- lifecycle -------------------------------------------------------------
    def add(self, skill: Skill, by_round: int = 0) -> str:
        if not skill.id or skill.id in self.skills:
            skill.id = self._new_id(skill.kind, skill.name)
        skill.status = "active"
        if not skill.created_round:
            skill.created_round = by_round
        skill.reuse.created_round = skill.created_round
        self.skills[skill.id] = skill
        self._instances_per_skill.setdefault(skill.id, set())
        self._bump()
        self._persist(skill)
        return skill.id

    def modify(self, target_id: str, new_skill: Skill, by_round: int) -> str:
        """Replace target with a new version; keep lineage, deprecate the old one."""
        old = self.skills.get(target_id)
        new_skill.parent_ids = [target_id] + list(new_skill.parent_ids or [])
        new_skill.version = (old.version + 1) if old else new_skill.version
        if old:
            old.status = "deprecated"
            self._persist(old)
        return self.add(new_skill, by_round)

    def merge(self, parent_ids: list[str], new_skill: Skill, by_round: int) -> str:
        """Combine several skills into a new one; deprecate the parents."""
        new_skill.parent_ids = list(parent_ids) + list(new_skill.parent_ids or [])
        for pid in parent_ids:
            if pid in self.skills:
                self.skills[pid].status = "deprecated"
                self._persist(self.skills[pid])
        return self.add(new_skill, by_round)

    def prune(self, skill_id: str, by_round: int) -> None:
        if skill_id in self.skills:
            self.skills[skill_id].status = "pruned"
            self._bump()
            self._persist(self.skills[skill_id])

    # -- retrieval (section 7.3) ----------------------------------------------
    def active(self, kind: str | None = None) -> list[Skill]:
        out = [s for s in self.skills.values() if s.status == "active"]
        return [s for s in out if (kind is None or s.kind == kind)]

    def retrieve(self, features: dict | None = None, kind: str | None = None,
                 k: int = 6) -> list[Skill]:
        """Small library (<= k or < 20): return all active. Larger: rank by reuse value.

        Embedding similarity over description_nl (sentence-transformers) is the planned
        upgrade; the pilot library stays small, so the return-all fallback is faithful.
        """
        cand = self.active(kind)
        if len(cand) <= k:
            return cand
        cand.sort(key=lambda s: (s.reuse.reuse_rate, s.reuse.marginal_value, s.created_round),
                  reverse=True)
        return cand[:k]

    # -- reuse accounting (scientific core) -----------------------------------
    def record_invocation(self, skill_id: str, instance_id: str, round_idx: int) -> None:
        s = self.skills.get(skill_id)
        if not s:
            return
        s.reuse.invocations += 1
        seen = self._instances_per_skill.setdefault(skill_id, set())
        seen.add(instance_id)
        s.reuse.instances_used_on = len(seen)
        s.reuse.last_used_round = round_idx
        span = max(1, round_idx - s.reuse.created_round)
        s.reuse.reuse_rate = s.reuse.instances_used_on / span

    def set_marginal_value(self, skill_id: str, value: float) -> None:
        if skill_id in self.skills:
            self.skills[skill_id].reuse.marginal_value = value

    # -- snapshots / seed ------------------------------------------------------
    def snapshot(self, round_idx: int) -> int:
        """Freeze the current library version (full history is retained for replay)."""
        return self.version

    def skill_states(self) -> list[dict]:
        """Serializable per-skill states for skill_snapshot.jsonl records (section 12)."""
        states = []
        for s in self.skills.values():
            states.append({
                "skill_id": s.id, "kind": s.kind, "version": s.version, "status": s.status,
                "parent_ids": s.parent_ids, "created_round": s.created_round,
                "reuse": dict(s.reuse.__dict__), "meta": dict(s.meta),
                "description_nl": s.description_nl, "plan": s.plan,
                "origin_family": s.origin_family, "tags": list(s.tags),
            })
        return states

    def set_seed(self) -> None:
        """Mark the current state as the seed (for the no-memory ablation arm)."""
        self._seed_state = copy.deepcopy(
            (self.skills, self.version, self._counter, self._instances_per_skill))

    def reset_to_seed(self) -> None:
        if self._seed_state is None:
            self.skills, self.version, self._counter, self._instances_per_skill = {}, 0, 0, {}
            return
        skills, version, counter, inst = copy.deepcopy(self._seed_state)
        self.skills, self.version, self._counter, self._instances_per_skill = \
            skills, version, counter, inst
