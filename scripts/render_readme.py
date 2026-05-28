"""Render polyalign README ablation block from results/v0.1.0a1_ablation.json.

S7.5 honest-marketing rule: any number that appears in README MUST come
from the ablation JSON (no hand-written placeholders). This script
inserts a block between the two markers in README.md:

  <!-- ABLATION:BEGIN -->
  ...auto-generated table...
  <!-- ABLATION:END -->

Run after `python scripts/run_ablation.py` regenerates the JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
# v0.1.0a2 keeps the v0.1.0a1 JSON file path as the primary results artifact
# for byte-stability; the file is regenerated under the new schema in v0.1.0a2.
ABLATION = ROOT / "results" / "v0.1.0a1_ablation.json"

BEGIN = "<!-- ABLATION:BEGIN -->"
END = "<!-- ABLATION:END -->"


def render_block(payload: dict) -> str:
    env = payload["env_stamp"]
    rows = payload["metrics_by_setting"]
    lines = [
        BEGIN,
        "",
        f"### Ablation (synthetic, `mode={env['mode']}`, dataset_n={env['dataset_n']})",
        "",
        f"> All values reproduced from `results/v0.1.0a1_ablation.json` (seed={env['seed']},",
        f"> Python {env['python']} on {env['os']}, polyalign {env['version']}).",
        f"> `[DEMO]` prefix is mechanically enforced on every ground-truth row (Case {env['case']}).",
        "> These numbers reflect **synthetic alignment quality** on Gaussian-decoder bundles, NOT real cross-model concept matching - see [docs/CLAIM.md](docs/CLAIM.md).",
        ">",
        "> Column notes:",
        "> - `top5` is bounded above by `min(top_k, n_target_pairs)/n_target_pairs`. For `top_k=5` and `n_target_pairs=n_feat=24` the cap is `5/24 ≈ 0.208`; the observed `0.208` in the planted regime reflects **100% precision over the 5 emitted vertices**, not 20.8% accuracy on the full target set. `top5_precision = hits / min(5, n_target_pairs)` is recorded in the JSON.",
        "> - `otcp_q` is the split-conformal threshold quantile (NOT empirical coverage). On doubly-stochastic Sinkhorn plans the nonconformity score `1 - p` with `p ~ 1/(n_a * n_b)` drives q to ≈1.0 by construction in every cell; a measured empirical-coverage column is deferred to v0.1.1.",
        "> - `ece` on the `planted` regime uses `correct = (edge matches planted permutation)`. On the `random` regime it uses `correct = ones` (Case C); see CLAIM.md `[non-CLAIM]` on ECE.",
        "> - `cycle_consistency_rate` is `1.000` everywhere because the default `cycle_threshold=0.0` is structurally tautological on non-negative plans (see CLAIM.md and `tests/test_polytope.py::test_polytope_3_bundles_cycle_threshold_discriminates`).",
        "",
        "| setting | top1 | top5 | top5_precision | ece | otcp_q | n_vertices | cycle_consistency_rate |",
        "|---------|------|------|----------------|-----|--------|------------|------------------------|",
    ]
    for key, m in rows.items():
        lines.append(
            f"| `{key}` | {m['top1']:.3f} | {m['top5']:.3f} | {m['top5_precision']:.3f} | "
            f"{m['ece']:.4f} | {m['otcp_q']:.4f} | {m['n_vertices']} | "
            f"{m['cycle_consistency_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"> Reproduce: `uv run python scripts/run_ablation.py --seed {env['seed']}`",
            "",
            END,
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if not ABLATION.exists():
        print(f"ERROR: {ABLATION} missing; run scripts/run_ablation.py first", file=sys.stderr)
        return 1
    payload = json.loads(ABLATION.read_text())
    block = render_block(payload)

    text = README.read_text()
    if BEGIN in text and END in text:
        head, _, tail = text.partition(BEGIN)
        _, _, rest = tail.partition(END)
        text = head + block + rest
    else:
        text = text.rstrip() + "\n\n## Reproducibility\n\n" + block + "\n"
    README.write_text(text)
    print(f"updated {README} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
