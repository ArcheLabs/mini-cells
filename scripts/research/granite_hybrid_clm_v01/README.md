# Granite Hybrid CLM v0.1 runner

This directory contains the executable engineering milestone for a frozen Granite MoE plus persistent CLM evolution layer.

## Smoke

```bash
python scripts/research/granite_hybrid_clm_v01/run_milestone.py \
  --device cuda:0 \
  --facts 3 \
  --address-steps 80 \
  --transform-steps 12 \
  --output-dir results/granite-hybrid-clm-v0.1-smoke
```

## Full Milestone 1

```bash
python scripts/research/granite_hybrid_clm_v01/run_milestone.py \
  --device cuda:0 \
  --facts 50 \
  --output-dir results/granite-hybrid-clm-v0.1
```

The runner writes each committed Cell as an independent `.pt` artifact plus `manifest.json`, `progress.json`, and `result.json`. A failed mutation remains uncommitted and therefore never changes production behavior.

The implementation deliberately separates:

1. address training on frozen cached read-layer features;
2. address freeze;
3. shadow transform training;
4. target + history safety selection;
5. commit.

The public JAM demonstration should reuse this lifecycle only after the controlled 50-fact milestone is stable.
