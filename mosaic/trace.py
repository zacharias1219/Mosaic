"""Tracer: every call is a span; question -> plan / filter / map / reduce (PRD 9.12)."""
from __future__ import annotations

import itertools
from typing import Optional

from mosaic.types import Span


class Tracer:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.spans: dict[str, list[Span]] = {}
        self._ids = itertools.count(1)

    def add(self, qid: str, role: str, parent: Optional[str], start: float, end: float, **kw) -> Span:
        s = Span(run_id=self.run_id, question_id=qid, span_id=f"{qid}:{next(self._ids)}", parent_id=parent, role=role,
                 start_ms=start, end_ms=end, total_ms=kw.pop("total_ms", end - start), **kw)
        self.spans.setdefault(qid, []).append(s)
        return s

    def pop(self, qid: str) -> list[Span]:
        return self.spans.pop(qid, [])


def question_cost(spans: list[Span]) -> float:
    return sum(s.cost for s in spans)


def critical_path_ms(spans: list[Span]) -> float:
    """Latency of a question from its spans: plan + filter + slowest map wave + reduce, which,
    because each stage starts when the previous one ends, is last end minus first start."""
    root = [s for s in spans if s.role == "question"]
    if root:
        children = [s for s in spans if s.role != "question"]
        if children:
            return max(s.end_ms for s in children) - root[0].start_ms
        return root[0].end_ms - root[0].start_ms
    return max(s.end_ms for s in spans) - min(s.start_ms for s in spans) if spans else 0.0


def validate_nesting(spans: list[Span]) -> bool:
    ids = {s.span_id: s for s in spans}
    roots = [s for s in spans if s.parent_id is None]
    if len(roots) != 1 or roots[0].role != "question":
        return False
    for s in spans:
        if s.parent_id is not None:
            p = ids.get(s.parent_id)
            if p is None or s.start_ms < p.start_ms - 1e-6 or s.end_ms > p.end_ms + 1e-6:
                return False
    return True
