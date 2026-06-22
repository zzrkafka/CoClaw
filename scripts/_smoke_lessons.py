"""Validate item 3: AutoGuide conditional retrieval (by kind) + ExpeL voting/retirement."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.lessons import (append_lesson, load_lessons, retrieve_lessons, vote_lessons)

p = Path(tempfile.mkdtemp()) / "lessons.jsonl"
append_lesson(p, {"id": "c1", "rule": "construct rule", "condition": {"kind": "construct"}})
append_lesson(p, {"id": "ls1", "rule": "ls rule", "condition": {"kind": "local_search"}})
append_lesson(p, {"id": "u1", "rule": "universal rule"})   # no condition -> universal

got = [l["id"] for l in retrieve_lessons(load_lessons(p), "construct")]
print("retrieve(construct) ->", got, "(expect c1 + u1, NOT ls1)")

for _ in range(3):
    vote_lessons(p, ["c1"], -1)          # 3 downvotes -> votes -3 -> retire
vote_lessons(p, ["u1"], 2)               # upvote universal
after = {l["id"]: l.get("votes") for l in load_lessons(p)}
print("after voting ->", after, "(expect c1 retired/gone, u1 votes=2)")

ok = ("c1" in got and "u1" in got and "ls1" not in got
      and "c1" not in after and after.get("u1") == 2)
print("VERDICT item3:", "PASS" if ok else "FAIL")
