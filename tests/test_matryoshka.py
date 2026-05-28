"""Matryoshka prefix-nested SAE alignment tests."""

from __future__ import annotations

import numpy as np
import pytest

from polyalign import SAEBundle
from polyalign.matryoshka import (
    MatryoshkaWrapper,
    default_prefix_schedule,
    multi_granularity_alignment,
    prefix_recon_error,
)


def test_default_prefix_schedule_4_stages() -> None:
    schedule = default_prefix_schedule(32)
    assert schedule == (4, 8, 16, 32)


def test_default_prefix_schedule_small_input_clamps() -> None:
    schedule = default_prefix_schedule(8)
    assert schedule[0] >= 2
    assert max(schedule) == 8


def test_default_prefix_schedule_rejects_tiny() -> None:
    with pytest.raises(ValueError, match="n_features >= 2"):
        default_prefix_schedule(1)


def test_matryoshka_wrapper_prefix_decoder(bundle_a: SAEBundle) -> None:
    wrapper = MatryoshkaWrapper(bundle_a)
    sub = wrapper.prefix(8)
    assert sub.n_features == 8
    assert sub.d_model == bundle_a.d_model
    np.testing.assert_allclose(sub.decoder, bundle_a.decoder[:8])


def test_matryoshka_wrapper_prefix_bounds(bundle_a: SAEBundle) -> None:
    wrapper = MatryoshkaWrapper(bundle_a)
    with pytest.raises(ValueError, match="out of range"):
        wrapper.prefix(1)
    with pytest.raises(ValueError, match="out of range"):
        wrapper.prefix(bundle_a.n_features + 1)


def test_matryoshka_wrapper_rejects_degenerate_prefix(bundle_a: SAEBundle) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        MatryoshkaWrapper(bundle_a, prefix_lengths=[1, 2, 4])


def test_multi_granularity_alignment_shapes(bundle_a: SAEBundle, bundle_b: SAEBundle) -> None:
    result = multi_granularity_alignment(bundle_a, bundle_b)
    assert len(result.prefix_lengths) >= 1
    for p, plan in zip(result.prefix_lengths, result.plans, strict=True):
        assert plan.shape == (p, p)


def test_prefix_recon_error_monotone() -> None:
    rng = np.random.default_rng(11)
    decoder = rng.standard_normal((16, 8)).astype(np.float32)
    activations = rng.standard_normal((20, 16)).astype(np.float32)
    bundle = SAEBundle("m", "transformer", 0, decoder)
    err = prefix_recon_error(bundle, activations)
    schedule = sorted(err.keys())
    # longer prefix => lower (or equal) reconstruction error
    from itertools import pairwise

    for s_prev, s_next in pairwise(schedule):
        assert err[s_next] <= err[s_prev] + 1e-6


def test_prefix_recon_error_shape_mismatch_raises() -> None:
    bundle = SAEBundle("m", "transformer", 0, np.zeros((8, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="!="):
        prefix_recon_error(bundle, np.zeros((20, 5), dtype=np.float32))


def test_multi_granularity_cross_arch_no_nan(bundle_a: SAEBundle, bundle_ssm: SAEBundle) -> None:
    result = multi_granularity_alignment(bundle_a, bundle_ssm)
    for plan in result.plans:
        assert not np.any(np.isnan(plan))
