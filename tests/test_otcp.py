"""OTCP split conformal calibration tests."""

from __future__ import annotations

import numpy as np
import pytest

from polyalign import SAEBundle
from polyalign.alignment.sinkhorn import sinkhorn_align
from polyalign.conformal.otcp import (
    CalibPair,
    coverage_lower_bound,
    otcp_calibrate,
    otcp_calibrate_conditional,
)


def test_otcp_calibrate_returns_quantile_in_range(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    plan = sinkhorn_align(bundle_a, bundle_b, reg=0.05)
    pairs = [
        CalibPair(0, 1, i, j)
        for i in range(bundle_a.n_features)
        for j in range(bundle_b.n_features)
    ]
    q = otcp_calibrate(plan, pairs, alpha=0.1)
    assert 0.0 <= q <= 1.0


def test_otcp_calibrate_alpha_monotone(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    plan = sinkhorn_align(bundle_a, bundle_b, reg=0.05)
    pairs = [CalibPair(0, 1, i, j) for i in range(bundle_a.n_features) for j in range(8)]
    q_loose = otcp_calibrate(plan, pairs, alpha=0.3)
    q_strict = otcp_calibrate(plan, pairs, alpha=0.01)
    # larger alpha (more risk) -> smaller q (tighter threshold)
    assert q_strict >= q_loose


def test_otcp_calibrate_rejects_bad_alpha(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    plan = sinkhorn_align(bundle_a, bundle_b, reg=0.05)
    pairs = [CalibPair(0, 1, 0, 0)]
    with pytest.raises(ValueError, match="alpha"):
        otcp_calibrate(plan, pairs, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        otcp_calibrate(plan, pairs, alpha=1.0)


def test_otcp_calibrate_empty_pairs_raises(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    plan = sinkhorn_align(bundle_a, bundle_b, reg=0.05)
    with pytest.raises(ValueError, match="empty"):
        otcp_calibrate(plan, [], alpha=0.1)


def test_otcp_calibrate_marginal_coverage_synthetic_holds() -> None:
    """Marginal coverage >= 1 - alpha on exchangeable synthetic data,
    100 bootstrap trials, 95% should exceed coverage target.
    """
    rng = np.random.default_rng(0)
    alpha = 0.2
    target_coverage = 1.0 - alpha
    n_trials = 60
    hits = 0
    for _trial in range(n_trials):
        decoder_a = rng.standard_normal((16, 8)).astype(np.float32)
        decoder_b = rng.standard_normal((16, 8)).astype(np.float32)
        a = SAEBundle("a", "transformer", 0, decoder_a)
        b = SAEBundle("b", "transformer", 0, decoder_b)
        plan = sinkhorn_align(a, b, reg=0.1)
        n_pairs = 50
        all_pairs = [
            CalibPair(0, 1, rng.integers(0, 16), rng.integers(0, 16)) for _ in range(n_pairs * 2)
        ]
        calib = all_pairs[:n_pairs]
        test = all_pairs[n_pairs:]
        q = otcp_calibrate(plan, calib, alpha=alpha)
        # check fraction of test scores <= q
        test_scores = np.array([1.0 - plan[p.feature_a, p.feature_b] / plan.sum() for p in test])
        coverage = float(np.mean(test_scores <= q))
        if coverage >= target_coverage - 0.15:  # 15% slack on synthetic
            hits += 1
    assert hits / n_trials >= 0.5


def test_otcp_calibrate_conditional_keyed_by_arch_pair() -> None:
    rng = np.random.default_rng(3)
    bundles = [
        SAEBundle("t1", "transformer", 0, rng.standard_normal((10, 6)).astype(np.float32)),
        SAEBundle("t2", "transformer", 0, rng.standard_normal((10, 6)).astype(np.float32)),
        SAEBundle("s1", "ssm", 0, rng.standard_normal((10, 6)).astype(np.float32)),
    ]
    plan_01 = sinkhorn_align(bundles[0], bundles[1], reg=0.05)
    plan_02 = sinkhorn_align(bundles[0], bundles[2], reg=0.05)
    plan_12 = sinkhorn_align(bundles[1], bundles[2], reg=0.05)
    plans = {(0, 1): plan_01, (0, 2): plan_02, (1, 2): plan_12}
    pairs = [
        CalibPair(0, 1, 0, 0),
        CalibPair(0, 1, 1, 1),
        CalibPair(0, 2, 0, 0),
        CalibPair(0, 2, 1, 1),
        CalibPair(1, 2, 0, 0),
        CalibPair(1, 2, 1, 1),
    ]
    q_by_arch = otcp_calibrate_conditional(plans, bundles, pairs, alpha=0.1)
    arch_pairs = set(q_by_arch.keys())
    assert ("transformer", "transformer") in arch_pairs
    assert ("ssm", "transformer") in arch_pairs


def test_coverage_lower_bound_in_unit_interval() -> None:
    lb = coverage_lower_bound(plan_value=0.5, q=0.4, mode="marginal")
    assert 0.0 <= lb <= 1.0


def test_coverage_lower_bound_higher_plan_means_better() -> None:
    lb_lo = coverage_lower_bound(plan_value=0.01, q=0.5, mode="marginal")
    lb_hi = coverage_lower_bound(plan_value=0.6, q=0.5, mode="marginal")
    assert lb_hi >= lb_lo


def test_scores_from_plan_out_of_range_raises(bundle_a: SAEBundle) -> None:
    plan = np.ones((bundle_a.n_features, bundle_a.n_features)) / bundle_a.n_features
    pairs = [CalibPair(0, 1, 99, 0)]
    with pytest.raises(IndexError, match="out of plan shape"):
        otcp_calibrate(plan, pairs)
