# MINICELLS-T4-1B-CAPACITY-001

## Status

The operator approved the pinned MoM 1.3B checkpoint. The
[Phase 1 audit](../../artifacts/experiments/minicells-t4-1b-capacity-001/phase-1-kaggle/README.md)
passed on Kaggle CPU. The [compatibility report](NATIVE_MOM_13B_COMPATIBILITY.md)
records the local Meta and software checks. Actual T4 gates are pending.

## Single-GPU preflight

The [Kaggle Script](../../apps/mom-native-cells/kaggle_t4_1b_capacity.py)
runs only the independent preflight. Enable Internet and select the T4
accelerator. On a T4 ×2 session the launcher selects a physical T4 by UUID and exposes it as
`cuda:0`. NVML samples the same UUID. The existing `GITHUB_TOKEN` secret is used for result publication
and the publisher verifies the result branch commit.

The independent runner can also be invoked on a compatible host from a
checked-out source commit with the `mom` extras, NVML and psutil available:

```sh
# Copy the full UUID for one T4 from nvidia-smi -L.
CUDA_VISIBLE_DEVICES="$T4_GPU_UUID" python apps/mom-native-cells/preflight_t4_1b.py --output ./t4-preflight
```

The runner reports actual GPU detection, strict weight loading and placement,
a short native FLA forward, logit sanity, 1024-token prefill, 128-token
generation, and memory measurements in `preflight_result.json` and
`T4_SINGLE_GPU_PREFLIGHT.md`. The FP16 prefill uses the SDK's existing
64-token recurrent execution path. The evidence applies to the recurrent
kernel and does not certify the BF16 Chunk kernel on T4.

## Capacity runbook

The capacity runner's `--stage all` runs its independent preflight before
any A–F worker. A standalone capacity stage requires a complete passing
preflight receipt in the same result directory for the exact target and
source commit. On failure, the capacity gates remain NOT_RUN. The capacity
matrix and acceptance thresholds below are unchanged.

## Measurement protocol

- Read All measures four registry sizes (1, 2, 4, 8) and three exact encoded
  prompt lengths (128, 512, 1024), using the SDK standard scorer and selector.
- Generation measures 47 configurations. Representative cases repeat three
  times after a warm-up. Normal EOS behavior is checked in the no-memory smoke.
  Capacity trials record EOS positions and continue to the requested token
  count. Logits, transient recurrent values, generated IDs and raw decoded
  text are retained or validated as appropriate.
- CUDA-synchronized timings include independent reading, Cell load, reduction
  of Shared Read Output Norm scores, selection, fused forward, prefill, and
  autoregressive generation. Fusion forward time overlaps with prefill and
  generation. The first short smoke and the per-case warm-up are recorded
  apart from measured repetitions.
- NVML samples device usage at nominal 50 ms intervals alongside PyTorch
  allocated and reserved memory. Each memory view has its own baseline and
  increment. The gate applies `min(13 GiB, device total − 3 GiB)` to total
  device use and treats other processes using at least 1 GiB as contamination.
- A fresh recovery worker imports all eight Cells, checks exact ZIP hashes,
  queries, appends, verifies sibling isolation, and queries again.
- An OOM keeps its requested configuration and memory evidence in the report.
  Later cases proceed only if the same model passes a short health check.
  The maximum is computed from successfully repeated configurations per mode.

The current evidence is an audit and CPU regression coverage of the runner.
No model loading time, VRAM ceiling, context limit, generation operability
or recovery result has been measured for the designated 1B model.
