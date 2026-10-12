# Core 008 Preflight — Functional Equivalence Bridge

Status: **ARTIFACT_AUDIT_COMPLETE / EXACT_COUNTERFACTUAL_PENDING_REHYDRATION**

Scientific decision: **false**.

This preflight is a non-decision bridge analysis between Core Validation 007 and the design of Core Validation 008. It must not modify, supersede, or reinterpret the frozen Core 007 protocol or scientific status.

Primary question:

> When Core 007 deploy routing disagrees with oracle functional-mode identity, is the near-zero heldout NLL gap evidence that the modes are functionally redundant/equivalent, or is it merely a consequence of Cell effects being too small relative to the frozen foundation model?

## Artifact audit result

The already-published Core 007 artifacts are **insufficient to establish functional equivalence**. They preserve aggregate NLL/routing metrics and per-Cell causal-ablation records, but not the final Cell matrices `A`, per-evaluation projected states, or counterfactual logits required to measure route-level functional regret.

See `ARTIFACT_AUDIT.md`.

The original Kaggle hidden-state cache is **not required**. The Core 007 data path is reproducible from pinned model/dataset revisions and the frozen manifest identity; missing hidden states are regenerated when the cache is absent.

## Frozen bridge protocol

`protocol.json` freezes the diagnostic-only measurement plan on the already-observed completed seeds:

```text
80721
80722
```

Seed `80723` is excluded because it never completed Core 007 candidate evaluation.

The bridge first requires deterministic rehydration to reproduce the published Core 007 candidate NLL and routing metrics. It then measures same-owner label redundancy, hidden/logit route distance, symmetric logit KL, and NLL route regret normalized by the Cell's own effect relative to the foundation path.

## Artifact-only audit

```bash
python scripts/research/analyze_core008_preflight_functional_equivalence.py
```

This requires no model download or GPU. It writes:

```text
artifacts/experiments/core-008-preflight-functional-equivalence/artifact-audit.json
```

## Exact counterfactual bridge

A fresh GPU session can regenerate the frozen inputs; the lost Kaggle cache is not needed.

Recommended one-command path:

```bash
python scripts/research/orchestrate_core008_preflight_functional_equivalence.py \
  --branch codex/core-008-preflight-functional-equivalence \
  --secret-name GITHUB_TOKEN \
  --device cuda \
  --batch-size 1 \
  --push-results
```

The orchestrator performs GitHub write preflight before GPU work, runs `80721` and `80722` in fresh processes, hydrates already-published completed bridge seeds after a session restart, generates `decision.json`/`RESULTS.md`, excludes the hidden-state cache from publication, commits, and pushes the bridge artifacts.

For Kaggle use the one-cell notebook:

```text
research/notebooks/04-continual-learning-core/core-008-preflight-functional-equivalence.ipynb
```

The seed runner validates the same data-manifest SHA used by Core 007:

```text
d098f9172083b8de9f825b66de5277dde5b6ea0581b3a950b8f76e4f443546cc
```

The resulting bridge status is diagnostic only and must not be presented as a new formal confirmation. Core 008, if opened, requires a new protocol and fresh formal seeds.
