# CLAIM.md — what polyalign claims and what it does not

Every assertion below begins at column 0 with either `[CLAIM]` (polyalign asserts this; the assertion is falsifiable by running the cited test) or `[non-CLAIM]` (polyalign explicitly does NOT claim this; deferred to a future release). CI greps `^\[(CLAIM|non-CLAIM)\]` to enforce line-start tags.

## Core algorithm correctness

[CLAIM] Sinkhorn-OT implementation in `polyalign.alignment.sinkhorn` matches POT `ot.sinkhorn` reference within `rtol=1e-4` for known small problems (`tests/test_sinkhorn.py::test_sinkhorn_align_pot_reference_match`).

[CLAIM] Matryoshka prefix-nested alignment recovers prefix-monotonic reconstruction-error trend on synthetic seeds, mirroring Bussmann et al. 2025 (`tests/test_matryoshka.py::test_prefix_recon_error_monotone`).

[CLAIM] OTCP split conformal in `polyalign.conformal.otcp` achieves marginal coverage >= 1 - alpha on synthetic exchangeable data over the bootstrap trials in `tests/test_otcp.py::test_otcp_calibrate_marginal_coverage_synthetic_holds`.

[CLAIM] PAVA monotone isotonic regression in `polyalign.calibration.pava` is vendored from foldconsensus (Apache-2.0) and inherits the foldconsensus PAVA invariants (`tests/test_calibration.py`).

[CLAIM] Alignment polytope vertex extraction is deterministic under a fixed seed (`tests/test_polytope.py::test_alignment_polytope_deterministic`).

[CLAIM] Cycle-consistency property holds for the polytope-vertex output on synthetic 3-model inputs (`tests/test_polytope.py::test_polytope_3_bundles_cycle_consistency`).

## Cross-architecture

[CLAIM] Sinkhorn-OT cost matrix builder supports `cosine` (default) and `l2` and exposes a `poincare` stub that warns and falls back to cosine in v0.1.0a1; the actual hyperbolic implementation is deferred to v0.2.

[CLAIM] Transformer-vs-SSM bundle pairs produce NaN-free Sinkhorn-OT plans at default reg on synthetic high-dimensional inputs (`tests/test_sinkhorn.py::test_sinkhorn_cross_architecture_no_nan`).

[non-CLAIM] polyalign v0.1.0a1 does NOT claim real-world SSM-vs-Transformer concept alignment fidelity. Cross-architecture validity at scale will be evaluated in v0.1.1 with real SAE weights.

## OTCP coverage mode

[CLAIM] Marginal coverage is the default returned by `otcp_calibrate(...)` and is verified on synthetic exchangeable inputs.

[CLAIM] Per-architecture-pair conditional coverage is implemented (`otcp_calibrate_conditional`) and tested for keying by sorted arch-pair tuples (`tests/test_otcp.py::test_otcp_calibrate_conditional_keyed_by_arch_pair`).

[non-CLAIM] Conditional-coverage empirical validity at scale (with real cross-architecture SAEs) is deferred to v0.1.1.

## Live model integration

[non-CLAIM] Live SAE extraction from Llama-3, Gemma-3, or Mamba-2 is NOT included in v0.1.0a1. The `polyalign[llama3]` / `polyalign[mamba]` / `polyalign[saelens]` optional extras install the dependencies, but v0.1.0a1 only consumes pre-extracted decoder / encoder arrays via `SAEBundle`. Live extraction lands in v0.1.1.

## Ground truth (Case C activated at S0)

[non-CLAIM] polyalign v0.1.0a1 does NOT claim performance against real hand-labeled cross-model concept pairs. The S0 OQ3 verdict was `degrade` because live cross-model SAE inference is infeasible inside a single build session. All metrics in `results/v0.1.0a1_ablation.json` are computed against synthetic `[DEMO]`-prefixed pairs in `datasets/ground_truth_v0.1.0a1.jsonl`.

[CLAIM] Every synthetic feature pair in `datasets/ground_truth_v0.1.0a1.jsonl` carries the `[DEMO]` prefix on the `concept_label` and the `prompt` fields. CI grep in `.github/workflows/audit.yml` enforces this.

[CLAIM] The ablation runner ships two synthetic regimes (`tests/test_ablation_runner.py` verifies both run): a `random` regime (random Gaussian decoders, no planted alignment, expected top-k ~ 0) and a `planted` regime (bundle B is bundle A's decoder permuted by a fixed-seed permutation; the synthetic ground truth points to that permutation). The `planted` regime returns non-trivial top-k > 0 on the same `[DEMO]` rows, and the JSON cell records `regime` and the planted permutation seed.

## Honest-marketing

[CLAIM] README, NOTICE, CHANGELOG, and docs do NOT contain any of the banned phrases listed in `.github/banned-phrases.txt`. The grep job in `.github/workflows/audit.yml` enforces this via `grep -rEni "$banned" README.md docs/ NOTICE CHANGELOG.md SECURITY.md`.

[CLAIM] README contains explicit differentiation against gavagai / AlignSAE / Anthropic crosscoder (the `.github/workflows/audit.yml` `diff-disclaimer` job enforces presence of these three terms).

## Deferred to v0.2+

[non-CLAIM] Poincare ball / hyperbolic cost matrix is a `poincare` stub in `polyalign.alignment.sinkhorn`; the actual hyperbolic implementation is deferred to v0.2.

[non-CLAIM] OKLab perceptually-uniform alignment visualization is deferred to v0.2.

[non-CLAIM] Muon (Newton-Schulz orthogonalization) optimizer integration is deferred to v0.2+.

## License

[CLAIM] polyalign is released under Apache License 2.0 (see `LICENSE`).

[CLAIM] All vendored / referenced upstream projects are listed in `NOTICE` with their license.
