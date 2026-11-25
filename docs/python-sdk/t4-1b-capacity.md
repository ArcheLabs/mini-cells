# MINICELLS-T4-1B-CAPACITY-001

## Decision

**BLOCKED at Stage 0.** No repository-designated MiniCells 1B checkpoint was
identified that the pinned Native MoM SDK can load. The hardware workload is
unrun and the maximum supported T4 configuration is unknown.

| Source | Evidence |
| --- | --- |
| SDK branch at `117c6e4559275b1a3d156be1e3fb48556759bd25` | Loader pins the MoM-340M model and revision `f4081422ea2cf01b1b5d646cac05cad62a5d5438`; Backbone requires hidden size 1024. |
| ArcheLabs default branch `32048a03fa297cd8ab481d34ed86f115de534169` | Repository and relevant MoM refs/configs yielded no designated Native MiniCells 1B checkpoint. |
| [Upstream MoM-1.3B checkpoint](https://huggingface.co/linear-moe-hub/MoM-Gated-Deltanet-1.3B/tree/5c8be94196cb1827e820077680ac2ee46ac949bc) | Safetensors metadata lists 1,618,420,000 parameters. Its configuration uses hidden size 2048, 24 layers, four routed banks plus Shared Memory, four heads of width 256. It is outside the current SDK registration and is not designated as the MiniCells target. |

The target gate reads [the checked-in audit manifest](../../apps/mom-native-cells/capacity_target.json).
An approved entry must carry a repository designation, immutable checkpoint
and tokenizer identity, and match the registered native loader. The historical
SDK checkpoint is explicitly excluded.

## Runbook

```sh
python apps/mom-native-cells/run_t4_1b_capacity.py --stage audit --output ./capacity-model-audit
```

The audit requires Python 3.11+ and emits `environment.json`, `model_manifest.json`,
`cell_schema.json`, `failures.json`, `summary.json`, `decision.json` and `report.md`.
The workload default runs `load`, `storage`, `scoring`, `generation` and
`recovery` as separate workers. A single stage can be repeated with
`--stage <stage> --output <existing-run>`; the run retains the previous stage
measurements. Each runtime worker loads only one model.

The [Kaggle launcher](../../apps/mom-native-cells/kaggle_t4_1b_capacity.py)
uses the repository publisher and the existing `GITHUB_TOKEN`. In this
revision it can publish the audit without starting the GPU workload. The
capacity sweep is intended for one visible T4 once the model gate can pass.

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
