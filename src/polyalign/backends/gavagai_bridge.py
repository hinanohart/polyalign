"""Bridge gavagai HookedSAEBundle / saelens / raw decoder -> polyalign.SAEBundle.

gavagai (MIT) is the upstream pairwise SAE indeterminacy scorer that
polyalign generalizes to N x M. We import gavagai lazily so that
polyalign installs cleanly without a hard dependency on gavagai.

The adapter mirrors `gavagai.backends.saelens_adapter.extract_decoder`
in spirit but returns a polyalign SAEBundle.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from polyalign._types import SAEBundle


def from_gavagai(
    bundle: Any,
    *,
    model_id: str,
    architecture: Literal["transformer", "ssm", "hybrid"] = "transformer",
    layer: int = 0,
) -> SAEBundle:
    """Adapt a gavagai HookedSAEBundle to a polyalign SAEBundle.

    Accepts any object with `.decoder` or `.W_dec` attribute (the gavagai
    convention as of v0.2.1). Falls back to numpy ndarray when bundle is
    already a decoder matrix.
    """
    if isinstance(bundle, np.ndarray):
        return SAEBundle(
            model_id=model_id,
            architecture=architecture,
            layer=layer,
            decoder=np.asarray(bundle, dtype=np.float32),
        )
    if hasattr(bundle, "decoder"):
        dec = np.asarray(bundle.decoder, dtype=np.float32)
    elif hasattr(bundle, "W_dec"):
        dec = np.asarray(bundle.W_dec, dtype=np.float32)
    elif isinstance(bundle, dict) and "W_dec" in bundle:
        dec = np.asarray(bundle["W_dec"], dtype=np.float32)
    else:
        raise TypeError(
            f"cannot extract decoder from object of type {type(bundle).__name__}; "
            f"expected ndarray, gavagai bundle, or dict with key 'W_dec'"
        )
    enc: np.ndarray | None = None
    if isinstance(bundle, dict):
        if "W_enc" in bundle:
            enc = np.asarray(bundle["W_enc"], dtype=np.float32)
    else:
        if hasattr(bundle, "encoder"):
            enc = np.asarray(bundle.encoder, dtype=np.float32)
        elif hasattr(bundle, "W_enc"):
            enc = np.asarray(bundle.W_enc, dtype=np.float32)
    return SAEBundle(
        model_id=model_id,
        architecture=architecture,
        layer=layer,
        decoder=dec,
        encoder=enc,
    )


def from_decoder_matrix(
    decoder: np.ndarray,
    *,
    model_id: str,
    architecture: Literal["transformer", "ssm", "hybrid"] = "transformer",
    layer: int = 0,
    encoder: np.ndarray | None = None,
) -> SAEBundle:
    """Construct an SAEBundle directly from a numpy decoder matrix.

    Useful for tests, synthetic data, and quickstart examples.
    """
    return SAEBundle(
        model_id=model_id,
        architecture=architecture,
        layer=layer,
        decoder=np.asarray(decoder, dtype=np.float32),
        encoder=None if encoder is None else np.asarray(encoder, dtype=np.float32),
    )
