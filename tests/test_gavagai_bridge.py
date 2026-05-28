"""gavagai HookedSAEBundle adapter tests."""

from __future__ import annotations

import numpy as np
import pytest

from polyalign.backends import from_decoder_matrix, from_gavagai


def test_from_decoder_matrix_basic() -> None:
    dec = np.eye(4, dtype=np.float32)
    b = from_decoder_matrix(dec, model_id="m", architecture="transformer", layer=0)
    assert b.n_features == 4
    assert b.d_model == 4
    assert b.architecture == "transformer"


def test_from_decoder_matrix_with_encoder() -> None:
    dec = np.ones((4, 2), dtype=np.float32)
    enc = np.ones((2, 4), dtype=np.float32)
    b = from_decoder_matrix(dec, model_id="m", architecture="ssm", layer=1, encoder=enc)
    assert b.encoder is not None
    assert b.encoder.shape == (2, 4)


def test_from_gavagai_accepts_raw_ndarray() -> None:
    dec = np.ones((6, 3), dtype=np.float32)
    b = from_gavagai(dec, model_id="raw", architecture="hybrid", layer=2)
    assert b.architecture == "hybrid"
    assert b.layer == 2
    assert b.decoder.shape == (6, 3)


def test_from_gavagai_w_dec_dict() -> None:
    bundle = {"W_dec": np.ones((6, 3), dtype=np.float32)}
    b = from_gavagai(bundle, model_id="dict", architecture="transformer")
    assert b.decoder.shape == (6, 3)


def test_from_gavagai_w_dec_object_attr() -> None:
    class FakeBundle:
        def __init__(self) -> None:
            self.W_dec = np.ones((6, 3), dtype=np.float32)
            self.W_enc = np.ones((3, 6), dtype=np.float32)

    b = from_gavagai(FakeBundle(), model_id="obj", architecture="transformer")
    assert b.encoder is not None
    assert b.encoder.shape == (3, 6)


def test_from_gavagai_decoder_attr_priority() -> None:
    class FakeBundle:
        def __init__(self) -> None:
            self.decoder = np.full((4, 2), 7.0, dtype=np.float32)
            self.W_dec = np.zeros((4, 2), dtype=np.float32)

    b = from_gavagai(FakeBundle(), model_id="prio", architecture="transformer")
    assert np.allclose(b.decoder, 7.0)


def test_from_gavagai_rejects_unknown_object() -> None:
    class Empty:
        pass

    with pytest.raises(TypeError, match="cannot extract decoder"):
        from_gavagai(Empty(), model_id="x", architecture="transformer")


def test_from_gavagai_roundtrip_dtype() -> None:
    dec = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float64)
    b = from_gavagai(dec, model_id="m", architecture="transformer")
    assert b.decoder.dtype == np.float32
