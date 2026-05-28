"""polyalign-lint CLI (typer)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import typer

from polyalign._version import __version__

app = typer.Typer(
    name="polyalign-lint",
    help="N-model M-architecture SAE alignment polytope.",
    no_args_is_help=True,
)


@app.command("version")
def version() -> None:
    """Print polyalign version."""
    typer.echo(__version__)


@app.command("smoke")
def smoke() -> None:
    """Run a tiny in-memory smoke test against synthetic SAE bundles.

    Returns exit code 0 on success. Used by S10 clean-clone audit and by
    `polyalign-lint smoke` for downstream-quickstart sanity.
    """
    import numpy as np

    from polyalign import SAEBundle, alignment_polytope

    rng = np.random.default_rng(0)
    bundle_a = SAEBundle(
        model_id="synth_a",
        architecture="transformer",
        layer=0,
        decoder=rng.standard_normal((32, 16)).astype(np.float32),
    )
    bundle_b = SAEBundle(
        model_id="synth_b",
        architecture="transformer",
        layer=0,
        decoder=rng.standard_normal((32, 16)).astype(np.float32),
    )
    result = alignment_polytope([bundle_a, bundle_b], top_k=3)
    typer.echo(
        f"smoke ok: vertices={len(result.vertices)} "
        f"alpha={result.coverage.alpha:.2f} q={result.coverage.quantile:.4f}"
    )


@app.command("align")
def align(
    bundles: str = typer.Option(..., "--bundles", help="Comma-separated .npz paths"),
    top_k: int = typer.Option(5, "--top-k", min=1, max=100),
    alpha: float = typer.Option(0.1, "--alpha", min=0.0, max=1.0),
    threshold_coverage: float = typer.Option(0.9, "--threshold-coverage"),
) -> None:
    """Align N SAE bundles and emit a polytope summary.

    Each .npz must contain `decoder` (n_features x d_model). Optional keys:
    `encoder`, `architecture`, `layer`, `model_id`, `feature_ids`.
    """
    import numpy as np

    from polyalign import SAEBundle, alignment_polytope

    paths = [Path(p.strip()) for p in bundles.split(",") if p.strip()]
    if len(paths) < 2:
        raise typer.BadParameter("--bundles requires at least 2 .npz paths")

    loaded: list[SAEBundle] = []
    for i, p in enumerate(paths):
        if not p.exists():
            raise typer.BadParameter(f"bundle file not found: {p}")
        data = np.load(p, allow_pickle=False)
        arch_str = str(data["architecture"]) if "architecture" in data.files else "transformer"
        if arch_str not in ("transformer", "ssm", "hybrid"):
            raise typer.BadParameter(
                f"unsupported architecture in {p}: {arch_str!r} (expected transformer|ssm|hybrid)"
            )
        arch = cast(Literal["transformer", "ssm", "hybrid"], arch_str)
        loaded.append(
            SAEBundle(
                model_id=str(data["model_id"]) if "model_id" in data.files else f"model_{i}",
                architecture=arch,
                layer=int(data["layer"]) if "layer" in data.files else 0,
                decoder=np.asarray(data["decoder"], dtype=np.float32),
                encoder=(
                    np.asarray(data["encoder"], dtype=np.float32)
                    if "encoder" in data.files
                    else None
                ),
            )
        )

    result = alignment_polytope(loaded, top_k=top_k, alpha=alpha)
    typer.echo(f"vertices: {len(result.vertices)}")
    typer.echo(f"coverage: q={result.coverage.quantile:.4f} mode={result.coverage.mode}")
    for v in result.vertices:
        typer.echo(
            f"  joint_p={v.joint_probability:.4f}  "
            f"coverage_lb={v.coverage_lower_bound:.3f}  "
            f"cycle={v.cycle_consistent}  "
            f"edges={len(v.model_pairs)}"
        )

    threshold_passed = all(v.coverage_lower_bound >= threshold_coverage for v in result.vertices)
    raise typer.Exit(code=0 if threshold_passed else 1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
