# CLM MoE Conversion 001

This package is the execution surface for the frozen protocol at:

`research/validations/moe-conversion-001/PROTOCOL.md`

The experiment deliberately keeps the original Granite MoE execution path unchanged. The CLM representation adds provenance, byte/tensor identities, and expert-slice mutation addresses around an immutable canonical substrate.

## Stage A — tiny deterministic smoke

```bash
pip install -e '.[lm,dev]'
python scripts/research/moe_conversion_001/run.py \
  --stage tiny \
  --device cpu \
  --dtype float32 \
  --work-dir artifacts/moe-conversion-001-tiny
```

## Stage B — real Granite 1B-A400M

For a Kaggle T4-class GPU:

```bash
python scripts/research/moe_conversion_001/run.py \
  --stage real \
  --model-id ibm-granite/granite-3.1-1b-a400m-base \
  --device cuda \
  --dtype float16 \
  --tolerance 1e-5 \
  --work-dir artifacts/moe-conversion-001-real
```

The default `hardlink` copy mode avoids tripling local checkpoint disk use when source, bundle and materialized output share a filesystem. It automatically falls back to ordinary copying when hard links are unavailable. Use `--copy-mode copy` when a physically independent bundle is required.

Each run writes `result.json`. A formal Stage B PASS requires the resolved Hugging Face revision SHA and every frozen gate to pass.
