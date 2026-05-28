"""Pareto-front extraction over polyalign Vertices.

A Vertex `v1` dominates `v2` if both:
- v1.joint_probability >= v2.joint_probability, and
- v1.coverage_lower_bound >= v2.coverage_lower_bound,
with at least one strict. Non-dominated vertices form the Pareto front.
"""

from __future__ import annotations

from collections.abc import Sequence

from polyalign._types import ParetoFront, Vertex


def _dominates(a: Vertex, b: Vertex) -> bool:
    ge = (
        a.joint_probability >= b.joint_probability
        and a.coverage_lower_bound >= b.coverage_lower_bound
    )
    gt = (
        a.joint_probability > b.joint_probability or a.coverage_lower_bound > b.coverage_lower_bound
    )
    return ge and gt


def pareto_front(vertices: Sequence[Vertex]) -> ParetoFront:
    """Return the Pareto front of vertices (non-dominated by any other vertex)."""
    survivors: list[Vertex] = []
    for v in vertices:
        if any(_dominates(other, v) for other in vertices if other is not v):
            continue
        survivors.append(v)
    survivors.sort(key=lambda x: (-x.joint_probability, -x.coverage_lower_bound))
    return ParetoFront(vertices=tuple(survivors))
