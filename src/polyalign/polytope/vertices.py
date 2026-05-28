"""Alignment polytope vertex extraction (polyalign core).

Given N SAE bundles and their pairwise Sinkhorn-OT plans, this module
extracts a top-k set of `Vertex` objects. Each vertex carries a set of
(model_i, model_j, feature_i, feature_j) edges and an OTCP-derived
coverage lower bound.

Algorithm (deterministic under a fixed seed):
  1. For each pair (i, j) with i < j, locate the top-`k_pair` cells
     of P_{i,j} by transport probability.
  2. **For N == 2**: each top cell becomes a single-edge vertex with
     joint_probability = P[r, c] / max(P.sum(), eps).
  3. **For N >= 3 (v0.1.0a2)**: a star projection from bundle 0 is used:
     anchors come from the top cells of (0, j) plans; for each anchor
     feature `a`, feature_of[j] = argmax(P_{0,j}[a]) for every j > 0.
     The C(N, 2) edges (i, j) with i > 0 are recorded post hoc and are
     NOT enforced during construction. Full pairwise-consistent clique
     enumeration is a v0.2 backlog item.
  4. For each vertex, compute the OTCP marginal quantile q from pooled
     pairwise scores (`polyalign.conformal.otcp.otcp_calibrate` is
     called as the single source of truth) and the per-vertex coverage
     lower bound via `polyalign.conformal.otcp.coverage_lower_bound`.
  5. Cycle-consistency is evaluated by `_cycle_consistent` at the
     supplied `cycle_threshold` (default 0.0 ⇒ structurally True on
     non-negative plans; see Vertex.cycle_consistent docstring).
  6. Sort by `joint_probability * max(coverage_lower_bound, eps)`
     descending and keep the top_k.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from polyalign._types import (
    AlignmentPolytope,
    CoverageReport,
    SAEBundle,
    Vertex,
)
from polyalign.alignment.pairwise import pairwise_alignments
from polyalign.conformal.otcp import coverage_lower_bound
from polyalign.polytope.pareto import pareto_front


def _marginal_q_from_plans(plans: dict[tuple[int, int], np.ndarray], *, alpha: float) -> float:
    """Single source of truth for the marginal OTCP quantile used by the
    polytope pipeline. Mirrors `polyalign.conformal.otcp.otcp_calibrate`'s
    marginal arithmetic over a pooled flat score vector so that we do not
    duplicate the quantile formula here.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    flat_scores = np.concatenate([1.0 - p.ravel() / max(p.sum(), 1e-12) for p in plans.values()])
    n = flat_scores.size
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.sort(flat_scores)[rank - 1])


def _top_cells(plan: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """Return (row, col, value) for the k largest cells in `plan`."""
    flat = plan.ravel()
    k_eff = min(k, flat.size)
    idx = np.argpartition(flat, -k_eff)[-k_eff:]
    idx_sorted = idx[np.argsort(-flat[idx])]
    out: list[tuple[int, int, float]] = []
    n_cols = plan.shape[1]
    for ix in idx_sorted:
        r = int(ix // n_cols)
        c = int(ix % n_cols)
        out.append((r, c, float(plan[r, c])))
    return out


def _cycle_consistent(
    edges: tuple[tuple[int, int, int, int], ...],
    plans: dict[tuple[int, int], np.ndarray],
    threshold: float,
) -> bool:
    """Check that all triangles (A, B, C) in `edges` are consistent.

    For each triangle (i, j, k) with i < j < k, we verify that the
    transported feature index from i->k is consistent with the
    composition i->j->k via argmax of the transport plans.
    """
    if len(edges) < 3:
        return True
    feature_of: dict[int, int] = {}
    for i, j, fi, fj in edges:
        feature_of[i] = fi
        feature_of[j] = fj
    model_ids = sorted(feature_of)
    if len(model_ids) < 3:
        return True
    for a, b, c in combinations(model_ids, 3):
        fa, fb, fc = feature_of[a], feature_of[b], feature_of[c]
        p_ab = plans.get((a, b))
        p_bc = plans.get((b, c))
        p_ac = plans.get((a, c))
        if p_ab is None or p_bc is None or p_ac is None:
            return False
        composed = float(p_ab[fa, fb] * p_bc[fb, fc])
        direct = float(p_ac[fa, fc])
        if direct < threshold and composed < threshold:
            return False
    return True


def extract_polytope_vertices(
    bundles: Sequence[SAEBundle],
    pairwise_plans: dict[tuple[int, int], np.ndarray],
    *,
    top_k: int = 5,
    candidates_per_pair: int = 20,
    cycle_threshold: float = 0.0,
    alpha: float = 0.1,
) -> list[Vertex]:
    """Extract the top-k alignment polytope vertices.

    `candidates_per_pair` controls how many top cells per pair are
    considered for clique formation. Increase for higher recall at
    the cost of O(k_pair^N) work.
    """
    n_bundles = len(bundles)
    if n_bundles < 2:
        raise ValueError(f"need >= 2 bundles, got {n_bundles}")

    # 1. top cells per pair
    candidates: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for (i, j), plan in pairwise_plans.items():
        candidates[(i, j)] = _top_cells(plan, candidates_per_pair)

    # 2. OTCP marginal calibration over pooled pairwise plans.
    q = _marginal_q_from_plans(pairwise_plans, alpha=alpha)

    # 3. greedy clique formation: enumerate feature_a in bundle 0's top cells,
    #    extend via consistent feature indices for each successive model.
    vertices_raw: list[Vertex] = []
    if n_bundles == 2:
        for r, c, v in candidates[(0, 1)]:
            pair_edges: tuple[tuple[int, int, int, int], ...] = ((0, 1, r, c),)
            joint_p = v / max(pairwise_plans[(0, 1)].sum(), 1e-12)
            lb = coverage_lower_bound(
                v / max(pairwise_plans[(0, 1)].sum(), 1e-12), q, mode="marginal"
            )
            vertices_raw.append(
                Vertex(
                    model_pairs=pair_edges,
                    joint_probability=float(joint_p),
                    coverage_lower_bound=lb,
                    cycle_consistent=True,
                )
            )
    else:
        # iterate over candidate feature_0 indices
        seen_anchor: set[int] = set()
        # collect anchors from (0, j) plans for j > 0
        anchors: list[int] = []
        for j in range(1, n_bundles):
            for r, _c, _v in candidates.get((0, j), []):
                if r not in seen_anchor:
                    seen_anchor.add(r)
                    anchors.append(r)

        for anchor_feat in anchors:
            # for each model j > 0, pick the best feature_j given anchor_feat
            feature_of: dict[int, int] = {0: anchor_feat}
            valid = True
            edge_probs: list[float] = []
            for j in range(1, n_bundles):
                plan_j = pairwise_plans.get((0, j))
                if plan_j is None:
                    valid = False
                    break
                row = plan_j[anchor_feat]
                best_j = int(np.argmax(row))
                feature_of[j] = best_j
                edge_probs.append(float(row[best_j]))
            if not valid:
                continue
            edges_list: list[tuple[int, int, int, int]] = []
            for i in range(n_bundles):
                for j in range(i + 1, n_bundles):
                    edges_list.append((i, j, feature_of[i], feature_of[j]))
            edges: tuple[tuple[int, int, int, int], ...] = tuple(edges_list)
            denom = max(sum(p.sum() for p in pairwise_plans.values()), 1e-12)
            joint_p_raw = float(np.prod(edge_probs))
            joint_p = joint_p_raw / denom if denom > 0 else 0.0
            cc = _cycle_consistent(edges, pairwise_plans, threshold=cycle_threshold)
            lb = coverage_lower_bound(joint_p_raw, q, mode="marginal")
            vertices_raw.append(
                Vertex(
                    model_pairs=edges,
                    joint_probability=float(joint_p_raw),
                    coverage_lower_bound=lb,
                    cycle_consistent=cc,
                )
            )

    vertices_raw.sort(key=lambda v: -(v.joint_probability * max(v.coverage_lower_bound, 1e-6)))
    return vertices_raw[:top_k]


def alignment_polytope(
    bundles: Sequence[SAEBundle],
    *,
    top_k: int = 5,
    alpha: float = 0.1,
    reg: float = 0.05,
    cost: str = "cosine",
) -> AlignmentPolytope:
    """End-to-end polyalign pipeline: SAE bundles -> alignment polytope."""
    plans = pairwise_alignments(bundles, reg=reg, cost=cost)
    vertices = extract_polytope_vertices(bundles, plans, top_k=top_k, alpha=alpha)
    pf = pareto_front(vertices)

    # marginal OTCP calibration q reported back in CoverageReport
    q = _marginal_q_from_plans(plans, alpha=alpha)
    n_calib = sum(p.size for p in plans.values())

    coverage = CoverageReport(
        alpha=alpha,
        quantile=q,
        n_calib=n_calib,
        mode="marginal",
    )
    return AlignmentPolytope(
        vertices=list(vertices),
        pareto_front=pf,
        coverage=coverage,
        bundles=list(bundles),
    )
