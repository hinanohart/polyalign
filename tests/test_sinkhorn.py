"""Sinkhorn-OT alignment tests (POT reference + properties)."""

from __future__ import annotations

import numpy as np
import ot
import pytest

from polyalign import SAEBundle
from polyalign.alignment.pairwise import pairwise_alignments
from polyalign.alignment.sinkhorn import build_cost_matrix, sinkhorn_align


def test_cost_matrix_cosine_shape(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    cm = build_cost_matrix(bundle_a, bundle_b, cost="cosine")
    assert cm.shape == (bundle_a.n_features, bundle_b.n_features)
    assert cm.min() >= 0.0


def test_cost_matrix_l2_nonnegative(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    cm = build_cost_matrix(bundle_a, bundle_b, cost="l2")
    assert cm.min() >= 0.0


def test_cost_matrix_unknown_raises(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    with pytest.raises(ValueError, match="unknown cost"):
        build_cost_matrix(bundle_a, bundle_b, cost="manhattan")


def test_cost_matrix_d_model_mismatch_raises() -> None:
    a = SAEBundle("a", "transformer", 0, np.zeros((4, 8)))
    b = SAEBundle("b", "transformer", 0, np.zeros((4, 6)))
    with pytest.raises(ValueError, match="d_model mismatch"):
        build_cost_matrix(a, b, cost="cosine")


def test_cost_matrix_poincare_stub_falls_back(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    with pytest.warns(UserWarning, match="v0.2 stub"):
        cm = build_cost_matrix(bundle_a, bundle_b, cost="poincare")
    cm_cos = build_cost_matrix(bundle_a, bundle_b, cost="cosine")
    np.testing.assert_allclose(cm, cm_cos)


def test_sinkhorn_align_marginals_uniform(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    plan = sinkhorn_align(bundle_a, bundle_b, reg=0.05)
    row_sums = plan.sum(axis=1)
    col_sums = plan.sum(axis=0)
    np.testing.assert_allclose(row_sums, 1.0 / bundle_a.n_features, atol=1e-4)
    np.testing.assert_allclose(col_sums, 1.0 / bundle_b.n_features, atol=1e-4)


def test_sinkhorn_align_pot_reference_match() -> None:
    rng = np.random.default_rng(7)
    dec_a = rng.standard_normal((5, 3)).astype(np.float32)
    dec_b = rng.standard_normal((5, 3)).astype(np.float32)
    a = SAEBundle("a", "transformer", 0, dec_a)
    b = SAEBundle("b", "transformer", 0, dec_b)
    cm = build_cost_matrix(a, b, cost="cosine")
    n = 5
    ref = ot.sinkhorn(
        np.full(n, 1.0 / n),
        np.full(n, 1.0 / n),
        cm,
        reg=0.1,
        method="sinkhorn_stabilized",
    )
    ours = sinkhorn_align(a, b, reg=0.1, cost="cosine")
    np.testing.assert_allclose(ours, np.asarray(ref), atol=1e-4)


def test_sinkhorn_align_reg_must_be_positive(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    with pytest.raises(ValueError, match="reg must be"):
        sinkhorn_align(bundle_a, bundle_b, reg=0.0)


def test_sinkhorn_cross_architecture_no_nan(bundle_a: SAEBundle, bundle_ssm: SAEBundle) -> None:
    plan = sinkhorn_align(bundle_a, bundle_ssm, reg=0.05)
    assert not np.any(np.isnan(plan))
    assert plan.min() >= 0.0


def test_sinkhorn_reg_decay_concentrates(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    p_hi = sinkhorn_align(bundle_a, bundle_b, reg=0.5)
    p_lo = sinkhorn_align(bundle_a, bundle_b, reg=0.01)
    # entropy of small reg < entropy of large reg
    h_hi = -(p_hi * np.log(p_hi + 1e-12)).sum()
    h_lo = -(p_lo * np.log(p_lo + 1e-12)).sum()
    assert h_lo < h_hi


def test_pairwise_alignments_count(
    bundle_a: SAEBundle, bundle_b: SAEBundle, bundle_ssm: SAEBundle
) -> None:
    plans = pairwise_alignments([bundle_a, bundle_b, bundle_ssm])
    assert set(plans.keys()) == {(0, 1), (0, 2), (1, 2)}
    for plan in plans.values():
        assert plan.shape == (bundle_a.n_features, bundle_b.n_features)


def test_pairwise_alignments_requires_two(bundle_a: SAEBundle) -> None:
    with pytest.raises(ValueError, match=">= 2"):
        pairwise_alignments([bundle_a])
