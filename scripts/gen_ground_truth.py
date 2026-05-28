"""Generate v0.1.0a1 synthetic ground truth feature pairs (Case C activated at S0).

S0 OQ3 verdict = degrade: live cross-model SAE inference is infeasible inside
the build session. We emit 36 synthetic feature pairs, each carrying the
`[DEMO]` prefix mechanically enforced by .github/workflows/audit.yml.

Schema (per JSONL line):
  {prompt, model_A_id, arch_A, layer_A, feature_id_A,
   model_B_id, arch_B, layer_B, feature_id_B,
   concept_label, source_url}
"""

from __future__ import annotations

import json
from pathlib import Path

CONCEPTS = [
    "color_blue",
    "color_red",
    "number_two",
    "negation_marker",
    "subject_pronoun_he",
    "subject_pronoun_she",
    "location_paris",
    "location_tokyo",
    "tense_past",
    "tense_future",
    "polarity_positive",
    "polarity_negative",
]

MODEL_PAIRS = [
    ("gpt2-small", "transformer", "pythia-160m", "transformer"),
    ("gpt2-small", "transformer", "mamba-2-130m", "ssm"),
    ("pythia-160m", "transformer", "mamba-2-130m", "ssm"),
]


def emit_rows() -> list[dict]:
    """36 synthetic rows: 12 concepts x 3 cross-arch pairs."""
    rows: list[dict] = []
    for k, concept in enumerate(CONCEPTS):
        for pair_idx, (m_a, arch_a, m_b, arch_b) in enumerate(MODEL_PAIRS):
            rows.append(
                {
                    "prompt": f"[DEMO] synthetic prompt for {concept} (template {k:02d})",
                    "model_A_id": m_a,
                    "arch_A": arch_a,
                    "layer_A": 6,
                    "feature_id_A": (k * 11 + pair_idx * 3) % 64,
                    "model_B_id": m_b,
                    "arch_B": arch_b,
                    "layer_B": 6,
                    "feature_id_B": (k * 7 + pair_idx * 5 + 17) % 64,
                    "concept_label": f"[DEMO] {concept}",
                    "source_url": "synthetic://polyalign/v0.1.0a1/Case-C",
                }
            )
    return rows


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "datasets" / "ground_truth_v0.1.0a1.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = emit_rows()
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"emitted {len(rows)} synthetic rows -> {out}")


if __name__ == "__main__":
    main()
