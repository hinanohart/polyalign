"""SAEBundle / Vertex / ParetoFront constructor and invariant tests."""

from __future__ import annotations

import numpy as np
import pytest

from polyalign import AlignmentPolytope, CoverageReport, ParetoFront, SAEBundle, Vertex


def test_sae_bundle_construction_basic() -> None:
    b = SAEBundle(
        model_id="m",
        architecture="transformer",
        layer=3,
        decoder=np.zeros((4, 2), dtype=np.float32),
    )
    assert b.n_features == 4
    assert b.d_model == 2
    assert b.architecture == "transformer"


def test_sae_bundle_rejects_1d_decoder() -> None:
    with pytest.raises(ValueError, match="2D"):
        SAEBundle(model_id="m", architecture="transformer", layer=0, decoder=np.zeros(8))


def test_sae_bundle_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        SAEBundle(
            model_id="m",
            architecture="quantum",  # type: ignore[arg-type]
            layer=0,
            decoder=np.zeros((2, 2)),
        )


def test_sae_bundle_rejects_inconsistent_encoder_shape() -> None:
    with pytest.raises(ValueError, match="encoder shape"):
        SAEBundle(
            model_id="m",
            architecture="transformer",
            layer=0,
            decoder=np.zeros((4, 2)),
            encoder=np.zeros((3, 4)),  # should be (d_model=2, n_features=4)
        )


def test_sae_bundle_accepts_consistent_encoder_shape() -> None:
    b = SAEBundle(
        model_id="m",
        architecture="ssm",
        layer=2,
        decoder=np.zeros((4, 2)),
        encoder=np.zeros((2, 4)),
    )
    assert b.encoder is not None
    assert b.encoder.shape == (2, 4)


def test_vertex_dataclass_immutable() -> None:
    from dataclasses import FrozenInstanceError

    v = Vertex(
        model_pairs=((0, 1, 0, 0),),
        joint_probability=0.5,
        coverage_lower_bound=0.9,
        cycle_consistent=True,
    )
    with pytest.raises(FrozenInstanceError):
        v.joint_probability = 0.1  # type: ignore[misc]


def test_coverage_report_construction() -> None:
    cr = CoverageReport(alpha=0.1, quantile=0.42, n_calib=50, mode="marginal")
    assert cr.alpha == 0.1
    assert cr.mode == "marginal"


def test_pareto_front_packs_tuple_of_vertices() -> None:
    v = Vertex(((0, 1, 0, 0),), 0.5, 0.5, True)
    pf = ParetoFront(vertices=(v,))
    assert len(pf.vertices) == 1
    assert pf.vertices[0].joint_probability == 0.5


def test_alignment_polytope_top_k() -> None:
    v = Vertex(((0, 1, 0, 0),), 0.5, 0.5, True)
    pf = ParetoFront(vertices=(v,))
    cr = CoverageReport(alpha=0.1, quantile=0.4, n_calib=10, mode="marginal")
    ap = AlignmentPolytope(vertices=[v, v], pareto_front=pf, coverage=cr)
    assert ap.top_k == 2
