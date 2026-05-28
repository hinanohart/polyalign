"""Shared pytest fixtures for polyalign."""

from __future__ import annotations

import numpy as np
import pytest

from polyalign import SAEBundle


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture()
def bundle_a() -> SAEBundle:
    rng = np.random.default_rng(0)
    return SAEBundle(
        model_id="synth_a",
        architecture="transformer",
        layer=0,
        decoder=rng.standard_normal((24, 12)).astype(np.float32),
    )


@pytest.fixture()
def bundle_b() -> SAEBundle:
    rng = np.random.default_rng(1)
    return SAEBundle(
        model_id="synth_b",
        architecture="transformer",
        layer=0,
        decoder=rng.standard_normal((24, 12)).astype(np.float32),
    )


@pytest.fixture()
def bundle_ssm() -> SAEBundle:
    rng = np.random.default_rng(2)
    return SAEBundle(
        model_id="synth_ssm",
        architecture="ssm",
        layer=0,
        decoder=rng.standard_normal((24, 12)).astype(np.float32),
    )


@pytest.fixture()
def bundle_hybrid() -> SAEBundle:
    rng = np.random.default_rng(3)
    return SAEBundle(
        model_id="synth_hybrid",
        architecture="hybrid",
        layer=0,
        decoder=rng.standard_normal((24, 12)).astype(np.float32),
    )
