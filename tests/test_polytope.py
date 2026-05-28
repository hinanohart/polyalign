"""Alignment polytope vertices + Pareto front tests (polyalign novel core)."""

from __future__ import annotations

import pytest

from polyalign import SAEBundle, Vertex, alignment_polytope
from polyalign.alignment.pairwise import pairwise_alignments
from polyalign.polytope.pareto import pareto_front
from polyalign.polytope.vertices import extract_polytope_vertices


def test_polytope_2_bundles_produces_vertices(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    plans = pairwise_alignments([bundle_a, bundle_b])
    verts = extract_polytope_vertices([bundle_a, bundle_b], plans, top_k=3)
    assert 1 <= len(verts) <= 3
    for v in verts:
        assert 0.0 <= v.joint_probability <= 1.0
        assert 0.0 <= v.coverage_lower_bound <= 1.0


def test_polytope_3_bundles_cycle_consistency(
    bundle_a: SAEBundle, bundle_b: SAEBundle, bundle_ssm: SAEBundle
) -> None:
    plans = pairwise_alignments([bundle_a, bundle_b, bundle_ssm])
    verts = extract_polytope_vertices([bundle_a, bundle_b, bundle_ssm], plans, top_k=5)
    assert len(verts) >= 1
    # At the default cycle_threshold=0 the predicate is structurally True on
    # non-negative plans (documented in docs/CLAIM.md [non-CLAIM]); this test
    # only asserts the predicate is well-formed.
    assert all(v.cycle_consistent for v in verts)


def test_polytope_3_bundles_cycle_threshold_discriminates(
    bundle_a: SAEBundle, bundle_b: SAEBundle, bundle_ssm: SAEBundle
) -> None:
    """At a non-trivial threshold > 0, the cycle predicate must discriminate.

    Pinned by the v0.1.0a2 post-/compact audit (Agent A MAJOR-1 + Meta MINOR-8):
    cycle_consistent at default 0.0 is structurally True; a positive threshold
    sufficiently above pooled-plan cell mass should drive the predicate to
    return False for star-projection vertices on random Gaussian decoders.
    """
    plans = pairwise_alignments([bundle_a, bundle_b, bundle_ssm])
    # threshold much larger than any single plan cell on doubly-stochastic
    # uniform plans (n_a * n_b cells, mass ~ 1/(n_a*n_b)) — predicate must
    # reject at least one vertex.
    verts = extract_polytope_vertices(
        [bundle_a, bundle_b, bundle_ssm], plans, top_k=5, cycle_threshold=1.0
    )
    assert len(verts) >= 1
    assert not all(v.cycle_consistent for v in verts), (
        "cycle predicate failed to discriminate at threshold=1.0; "
        "test_polytope_3_bundles_cycle_threshold_discriminates regression"
    )


def test_polytope_too_few_bundles(bundle_a: SAEBundle) -> None:
    plans: dict = {}
    with pytest.raises(ValueError, match=">= 2"):
        extract_polytope_vertices([bundle_a], plans)


def test_alignment_polytope_endtoend(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    result = alignment_polytope([bundle_a, bundle_b], top_k=3)
    assert len(result.vertices) >= 1
    assert result.coverage.alpha == 0.1
    assert result.coverage.mode == "marginal"


def test_alignment_polytope_deterministic(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    r1 = alignment_polytope([bundle_a, bundle_b], top_k=3)
    r2 = alignment_polytope([bundle_a, bundle_b], top_k=3)
    assert len(r1.vertices) == len(r2.vertices)
    for v1, v2 in zip(r1.vertices, r2.vertices, strict=True):
        assert v1.model_pairs == v2.model_pairs
        assert v1.joint_probability == pytest.approx(v2.joint_probability)


def test_pareto_front_filters_dominated() -> None:
    v_strong = Vertex(((0, 1, 0, 0),), 0.9, 0.9, True)
    v_dominated = Vertex(((0, 1, 0, 1),), 0.5, 0.5, True)
    v_orth = Vertex(((0, 1, 1, 0),), 0.4, 0.95, True)
    pf = pareto_front([v_strong, v_dominated, v_orth])
    # v_dominated dominated by v_strong
    assert v_strong in pf.vertices
    assert v_orth in pf.vertices
    assert v_dominated not in pf.vertices


def test_pareto_front_keeps_all_when_no_dominance() -> None:
    v_a = Vertex(((0, 1, 0, 0),), 0.9, 0.1, True)
    v_b = Vertex(((0, 1, 1, 1),), 0.1, 0.9, True)
    pf = pareto_front([v_a, v_b])
    assert len(pf.vertices) == 2


def test_polytope_top_k_caps_count(
    bundle_a: SAEBundle, bundle_b: SAEBundle, bundle_ssm: SAEBundle
) -> None:
    plans = pairwise_alignments([bundle_a, bundle_b, bundle_ssm])
    verts = extract_polytope_vertices([bundle_a, bundle_b, bundle_ssm], plans, top_k=2)
    assert len(verts) <= 2


def test_polytope_4_bundle_includes_hybrid(
    bundle_a: SAEBundle,
    bundle_b: SAEBundle,
    bundle_ssm: SAEBundle,
    bundle_hybrid: SAEBundle,
) -> None:
    plans = pairwise_alignments([bundle_a, bundle_b, bundle_ssm, bundle_hybrid])
    verts = extract_polytope_vertices(
        [bundle_a, bundle_b, bundle_ssm, bundle_hybrid], plans, top_k=3
    )
    assert len(verts) >= 1
    for v in verts:
        # for N=4 bundles, every clique has C(4, 2) = 6 edges
        assert len(v.model_pairs) == 6


def test_polytope_models_attached_to_result(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    result = alignment_polytope([bundle_a, bundle_b], top_k=2)
    assert len(result.bundles) == 2
    assert result.bundles[0].model_id == "synth_a"
