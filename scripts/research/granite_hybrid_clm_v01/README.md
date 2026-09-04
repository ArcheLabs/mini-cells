# Granite Hybrid CLM v0.1 runner

This directory contains the executable engineering milestone for a frozen Granite MoE plus persistent CLM evolution layer.

## Kaggle smoke

Use the single orchestration entrypoint first:

```bash
python scripts/research/granite_hybrid_clm_v01/run_kaggle.py \
  --mode smoke \
  --device cuda:0
```

Smoke runs 3 controlled facts, then starts a fresh Granite process-equivalent overlay reconstruction from the emitted Cell artifacts and requires reload semantic retention.

## Full Milestone 1

After smoke is stable:

```bash
python scripts/research/granite_hybrid_clm_v01/run_kaggle.py \
  --mode full \
  --device cuda:0
```

The full flow runs 50 sequential facts, contextual child update, fresh-runtime artifact reload, and the canonical acceptance validator.

Equivalent explicit commands are:

```bash
python scripts/research/granite_hybrid_clm_v01/run_milestone.py \
  --device cuda:0 \
  --facts 50 \
  --output-dir results/granite-hybrid-clm-v0.1

python scripts/research/granite_hybrid_clm_v01/verify_reload.py \
  --device cuda:0 \
  --result-dir results/granite-hybrid-clm-v0.1

python scripts/research/granite_hybrid_clm_v01/validate_result.py \
  --result-dir results/granite-hybrid-clm-v0.1
```

## Publish an accepted full result

Publication is deliberately downstream of the canonical validator. The publisher refuses a result without successful fresh-runtime reload evidence.

```bash
python scripts/research/granite_hybrid_clm_v01/publish.py \
  --preflight-only \
  --branch codex/granite-hybrid-clm-v0.1

python scripts/research/granite_hybrid_clm_v01/publish.py \
  --branch codex/granite-hybrid-clm-v0.1
```

The runner writes each committed Cell as an independent `.pt` artifact plus `manifest.json`, `progress.json`, and `result.json`. A failed mutation remains uncommitted and therefore never changes production behavior.

The implementation deliberately separates:

1. address training on frozen cached read-layer features;
2. address freeze;
3. shadow transform training;
4. target + history safety selection;
5. commit.

For the main 50-fact path, final evaluation paraphrases are not used to train the applicability gate. Production candidate-choice therefore tests both address generalization and transform acquisition.

The public JAM demonstration should reuse this lifecycle only after the controlled 50-fact milestone is accepted.
