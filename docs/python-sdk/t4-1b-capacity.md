# MINICELLS-T4-1B-CAPACITY-001

## Status

The operator approved the pinned MoM 1.3B checkpoint. The
[Phase 1 audit](../../artifacts/experiments/minicells-t4-1b-capacity-001/phase-1-kaggle/README.md)
passed on Kaggle CPU. The [compatibility report](NATIVE_MOM_13B_COMPATIBILITY.md)
records the local Meta and software checks. The single-T4 preflight passed
all nine gates on Kaggle. [Published preflight receipt](https://github.com/ArcheLabs/mini-cells/blob/3d068d1f80e99756c4cd7e50ba3f0ab868079195/artifacts/experiments/minicells-t4-1b-capacity-001/4230429be5ba/preflight_result.json).

| Operation | Seconds | Peak device memory (MiB) |
| --- | ---: | ---: |
| Download and strict load to cuda:0 | 64.17 | 3,529.2 |
| First 16-token forward | 69.15 | 4,127.2 |
| 1024-token prefill | 4.44 | 4,133.2 |
| 128-token greedy generation | 20.72 | 4,133.2 |

The validated configuration is one Tesla T4, FP16, 1,552,884,000 parameters,
and no external Cell. Decode throughput was 6.18 tokens/s. These results
establish runtime feasibility for the tested configuration.

## Single-GPU preflight

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

Upload the [Kaggle Script](../../apps/mom-native-cells/kaggle_t4_1b_capacity.py),
enable Internet, select the T4 accelerator, and enable the existing
`GITHUB_TOKEN` secret. The launcher selects one physical T4 by UUID and exposes
it as `cuda:0`. It runs the preflight and capacity in the same session and
publishes the receipts to a verified GitHub result branch.

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

## Capacity result

The Kaggle run returned **FULL PASS** at source commit `75b171e4d`. A–F all
passed. All 12 Read All and 47 generation configurations passed, with
14 and 55 successful measured trials respectively. No trial recorded a
failure. [Verified capacity artifacts](https://github.com/ArcheLabs/mini-cells/tree/d480bf23926cc72adc2b9c2ff3919406375d0438/artifacts/experiments/minicells-t4-1b-capacity-001/33c9856baf45).

| Registered Cells | Prompt and generated tokens | Selection mode | Median (s) | Peak device memory (MiB) |
| ---: | --- | --- | ---: | ---: |
| 0 | 1024 + 128 | No memory | 23.90 | 4,131.2 |
| 1 | 1024 + 128 | Manual | 23.30 | 4,137.2 |
| 1 | 1024 + 128 | Auto | 27.30 | 4,137.2 |
| 8 | 1024 + 128 | Manual | 33.48 | 4,265.2 |
| 8 | 1024 + 128 | Auto | 62.94 | 4,265.2 |

The longest generation cases at eight Cells had three measured repetitions
after warm-up. Automatic selection scored all eight registered Cells and
then fused two, as did the explicit manual selection. At 1024 input tokens,
Read All took 3.68, 7.25, 14.42, and 28.77 seconds at 1, 2, 4, and 8 Cells.
The 2- and 4-Cell generation configurations used 12 new tokens.

The peak sampled device usage was 4.17 GiB against a 12 GiB limit. The tested
Cell contained 117 MiB of CPU tensor payload. Most high-count capacity Cells
were replicas of one seed state. In the independent recovery process, import
of all eight Cells took 7.82 seconds and recovered the exact ZIP hashes.
Continued writing modified only the target Cell, and the subsequent query
succeeded.

Eight is the highest tested registry size. Simultaneous fusion of eight
Cells and generation quality were outside this protocol. The result records
the source commit and runtime environment. Kaggle's script version ID was
unavailable.
