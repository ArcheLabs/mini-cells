"""Public markdown/report generation for the CLM-0.3 release benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _pct_improvement(ratio: float) -> str:
    return f"{100.0 * (1.0 - ratio):.2f}%"


def _pct_gap(ratio: float) -> str:
    gap = 100.0 * (ratio - 1.0)
    return f"{gap:+.2f}%"


def write_public_release_summary(
    path: Path,
    *,
    historical: dict[str, Any],
    bridge: dict[str, Any],
    capability: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    dense = bridge["arms"]["textnca_continuation"]
    clm = bridge["arms"]["clm_fixed4"]
    quality = decision["language_quality"]
    runtime = decision["reference_runtime"]
    foundation_006 = historical["experiment_006"]
    foundation_007 = historical["experiment_007"]
    promoted = capability.get("promoted_replicates", [])
    promoted_text = ", ".join(
        f"r{row['replicate']} {row['selected_expert']} ({row['ppl_improvement_percent']:.2f}% lower PPL)"
        for row in promoted
    ) or "none"

    overall = decision["overall"]["status"]
    content = f"""# CLM-0.3 Public Release Benchmark

**Release recommendation:** `{overall}`

CLM-0.3 adds a developmental capability to the TextNCA language-model substrate: a trained model can create function-preserving shadow lineages, let them develop under future experience, reject unnecessary structure, and promote persistent capacity only when the added lineage demonstrates sustained utility.

This release benchmark keeps two questions separate:

1. **Does the model remain a competitive language model?**
2. **Does it gain a capability that a fixed model does not have?**

## 1. Language-model foundation

Earlier matched-Transformer experiments are retained as immutable foundation evidence rather than retrained for this release.

| Evidence | TextNCA PPL | Transformer PPL | PPL ratio |
| --- | ---: | ---: | ---: |
| Experiment 006 — ~1.17M params, 10M tokens | {foundation_006['textnca_ppl']:.4f} | {foundation_006['transformer_ppl']:.4f} | {foundation_006['ppl_ratio_textnca_over_transformer']:.4f}× |
| Experiment 007 — ~30M params, 100M tokens | {foundation_007['textnca_ppl']:.4f} | {foundation_007['transformer_ppl']:.4f} | {foundation_007['ppl_ratio_textnca_over_transformer']:.4f}× |

At ~30M parameters / 100M training tokens, TextNCA was within **{_pct_gap(float(foundation_007['ppl_ratio_textnca_over_transformer']))}** PPL of its parameter-matched Transformer.

![Language-model quality chain](figures/figure-1-language-quality.png)

## 2. Cost of becoming a CLM

The release bridge starts both arms from the exact same trained Experiment-006 10M TextNCA checkpoint and continues them for the same 1M unseen-suffix training-token budget.

| Metric | TextNCA continuation | CLM fixed4 | CLM / TextNCA |
| --- | ---: | ---: | ---: |
| Final validation PPL | {dense['final_ppl']:.4f} | {clm['final_ppl']:.4f} | {quality['final_ppl_ratio_clm_over_textnca']:.4f}× |
| Total stored parameters | {dense['parameters']['total_parameters']:,} | {clm['parameters']['total_parameters']:,} | {clm['parameters']['total_parameters'] / dense['parameters']['total_parameters']:.2f}× |
| Active-parameter proxy | {dense['parameters']['active_parameter_proxy']:,} | {clm['parameters']['active_parameter_proxy']:,} | {runtime['active_parameter_proxy_ratio_clm_over_textnca']:.2f}× |
| Training throughput | {dense['runtime']['train_tokens_per_second']:.0f} tok/s | {clm['runtime']['train_tokens_per_second']:.0f} tok/s | {clm['runtime']['train_tokens_per_second'] / dense['runtime']['train_tokens_per_second']:.2f}× |
| Inference throughput | {dense['runtime']['inference_tokens_per_second']:.0f} tok/s | {clm['runtime']['inference_tokens_per_second']:.0f} tok/s | {runtime['inference_throughput_ratio_clm_over_textnca']:.2f}× |
| Inference peak VRAM | {dense['runtime']['inference_peak_vram_bytes'] / (1024**2):.1f} MiB | {clm['runtime']['inference_peak_vram_bytes'] / (1024**2):.1f} MiB | {runtime['inference_vram_ratio_clm_over_textnca']:.2f}× |

Language-quality status: `{quality['status']}`  
Reference-runtime status: `{runtime['status']}`

![Same-checkpoint machinery bridge](figures/figure-2-machinery-bridge.png)

![Reference runtime and structural cost](figures/figure-4-reference-cost.png)

The active-parameter number is a structural proxy: shared parameters + one active expert per stage + router parameters. It is **not** a measured FLOP count. The current CLM inference measurement uses the repository's `sparse_dispatch` correctness/reference backend, not a fused production kernel.

## 3. Developmental capability

CLM-0.3d supplies the formal capability evidence used by this release:

- function-preserving shadow births: **{capability['births_equivalent']}/{capability['births_checked']}**;
- stationary continuation rejected persistent growth: **{capability['stationary_rejected']}/{capability['stationary_total']}** replicates;
- controlled capability shift promoted persistent growth: **{capability['shift_promoted']}/{capability['shift_total']}** replicates;
- independently confirmed promoted lineages: **{promoted_text}**.

![Developmental selectivity](figures/figure-3-developmental-selectivity.png)

The central result is selectivity rather than unconditional expansion: under stationary continuation the probationary mechanism rejected all three persistent births, while a controlled capability shift produced independently confirmed promotion in two of three replicates.

## What CLM-0.3 supports

- A trained TextNCA can be converted into a fixed hierarchical CLM without changing its function at the conversion boundary.
- CLM language quality remains `{quality['status']}` relative to the same-checkpoint TextNCA continuation under the preregistered bridge.
- A trained CLM can create temporary shadow lineages and evaluate them over future data.
- The probationary controller can reject unnecessary persistent structure under stationary continuation.
- Under the formal controlled capability shift, persistent lineage promotion occurred in 2/3 replicates and survived an independent holdout.
- Persistent capacity can therefore be allocated after training according to demonstrated future utility rather than a fixed pretraining architecture alone.

## What CLM-0.3 does not claim

- It does not claim production-optimized sparse inference.
- It does not claim measured FLOP savings from the active-parameter proxy.
- It does not establish arbitrary-domain or arbitrary-capability generality.
- It does not establish repeated lifelong mitosis in one continuously developing organism; that is a CLM-0.4 question.
- It does not establish 30M-scale probationary growth. Experiment 007 is foundation evidence for the TextNCA substrate, not a 30M CLM growth experiment.

## Provenance

- Release-bridge training commit: `{bridge['training_commit']}`
- Release-bridge training tree: `{bridge['training_tree_sha']}`
- Source TextNCA checkpoint SHA-256: `{bridge['source_checkpoint_sha256']}`
- CLM-0.3d capability source ref: `{capability['source_ref']}`
- CLM-0.3d capability source commit: `{capability['source_commit']}`
- CLM-0.3d training commit: `{capability['training_code_commit']}`

Machine-readable evidence is in `decision.json`, `bridge-summary.json`, `historical-evidence.json`, and `capability-evidence.json`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
