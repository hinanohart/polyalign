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
        "> These numbers reflect **synthetic alignment quality on random Gaussian decoders**, not real cross-model concept matching - see [docs/CLAIM.md](docs/CLAIM.md).",
        "",
        "| setting | top1 | top5 | ece | otcp_coverage | n_vertices | cycle_consistency_rate |",
        "|---------|------|------|-----|---------------|------------|------------------------|",
    ]
    for key, m in rows.items():
        lines.append(
            f"| `{key}` | {m['top1']:.3f} | {m['top5']:.3f} | {m['ece']:.4f} | "
            f"{m['otcp_coverage']:.4f} | {m['n_vertices']} | {m['cycle_consistency_rate']:.3f} |"
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
