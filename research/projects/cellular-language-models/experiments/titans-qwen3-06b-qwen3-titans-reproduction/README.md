# Qwen3-0.6B Titans reproduction

This experiment is separate from QTR-1 and uses the official post-trained `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`. The adapter reuses the upstream `SegmentedAttention`, `NeuralMemory`, `MemoryMLP`, and `FP32AdamW` at `c60b4efaa3a3c6df4deebd57cbccf75947b87cad`.

| Item | Implementation |
| --- | --- |
| Backbone | 28 layers, hidden 1024, 16 Q heads, 8 KV heads, head dimension 128; all native weights frozen |
| Position convention | Official configuration has 40,960 positions; experiment caps inputs at 32,768, with unchanged native RoPE |
| Layers | Zero-based layers 0–3 replace attention only; layers 4–27 remain native full attention |
| Qwen adaptation | Retain native projections and Q/K norms; rotate Q/K once with Qwen position embeddings before both paths |
| Local attention and recurrence | 512-token causal segments; 512-token write chunks; 1024-token TTT batch boundary; upstream token-shifted read timeline prevents current-chunk reads |
| GQA | Two adjacent Q heads per memory head; 8 independently parameterized memory heads per layer |
| Value residual | Accept preceding layer's original V at layers 1–3, with upstream zero weights and bias −10 for the mix; memory stores this layer's original V |
| Memory | Width 128, depth 3, expansion 1, GELU, FP32; first two matrices I and last matrix 4I via upstream Identity initialization; no K→V distribution fit |
| Inner update | Adaptive LR max 1, first-order momentum, fast grad norm cap 1, higher-order inner gradients enabled; history mix starts at 0.05 |
| Outer update | Upstream FP32AdamW, β=(0.9,0.95), clipping 10, warmup and cosine using each upstream JSON profile |
| Precision | BF16 frozen backbone where supported; FP16 otherwise; FP32 memory, trainable parameters and Adam moments |
| Practical adaptation | One sequence per physical microbatch; accumulation preserves global batch; checkpoint frozen upper layers and use tiled logits |
| Data | Upstream dataset iterators and masking; pinned official Qwen tokenizer replaces the Llama tokenizer; stream revisions recorded by the runner |

Source-of-truth for the training budget: `configs/seg512_{pretrain,gold_sft,quality_sft}/training_config.json` in the pinned upstream checkout. The runner uses the upstream floor rule for one epoch: 953 pretraining steps (999,292,928 nominal tokens), 200 Gold SFT steps, then 59 QuALITY SFT steps. Stage order and dataset are recorded. The independent upstream `scripts/finetune.sh quality` also offers a **different** three-epoch recipe; that is not the selected JSON profile.

## Validation commands

Use Python 3.11+, PyTorch 2.9.x, Transformers 4.57.3/4.57.6 and the pinned upstream research dependencies. From this directory:

```bash
PYTHONPATH=/path/to/pinned/titans-pytorch python -m unittest -v test_model
python run.py /path/to/pinned/titans-pytorch /path/to/artifacts/repro-smoke --smoke
```

Smoke processes two packed 16K sequences and replays the second update from the first checkpoint, comparing the resulting training state. It saves a separate initial and post-step checkpoint. Measure GPU peak and time here; this engineering check does not estimate generalization. The actual formal step contains 64 sequences, and peak/time at that scale should be measured before committing resources. To budget just pretraining, multiply the measured time per 16K sequence by 60,992, plus data and outer-step overhead. A single Kaggle session should not be assumed to cover this budget. No shorter formal sequence is configured.

## Formal command (after acceptance)

```bash
python run.py /path/to/pinned/titans-pytorch /durable/path/artifacts/experiments/qwen3-06b-titans-reproduction \
  --smoke-receipt /path/to/artifacts/repro-smoke/decision.json \
  --dataset-lock /path/to/data-lock.json \
  --quality-train /path/to/QuALITY.v1.0.1.htmlstripped.train \
  --validation /path/to/frozen-validation.jsonl
```

The optional `--dataset-lock` accepts the `revisions` object from smoke’s `dataset-manifest.json` saved as a JSON file. Without it the runner resolves the current dataset commits and compares them with the smoke receipt.

The validation file contains `id`, `input_ids`, `attention_mask`, and `labels` per document, with labels masked by `-100` where excluded. It must be prepared with the pinned Qwen tokenizer. The three arms score identical rows; this measures validation NLL and next-token accuracy. The pinned artifact does not provide its LongHealth or RULER evaluation code or its QuALITY dev scorer/data. This validation runner does not stand in for those protocols. Do not use the formal command until real-weight tests, checkpoint continuation parity, and the required benchmark inputs/protocols have been verified. A run on this path cannot mark the scientific result as PASS.

Add `--resume` to restore from the last saved optimizer boundary. It uses the stored sampler cursor to recreate and replay the locked streams, then restores Python and Torch RNG, parameters, optimizer and scheduler. Older checkpoints are preserved. Persist the entire output directory on durable storage; the runner does not delete checkpoint files.

## Current verification

- Python compilation: passed.
- CPU protocol checks (`test_run.py`): 2 passed. The schedule uses the three pinned budgets; a synthetic FP32AdamW parameter/optimizer/scheduler round trip produced bit-identical continuation.
- Synthetic adapter checks (`test_model.py`): blocked at import by missing `assoc_scan`. This environment's dependency download failed; none of the adapter assertions ran.
- Real-weight GPU smoke, end-to-end resume comparison, benchmark evaluation and formal training: not run.

## Kaggle preflight

Paste `kaggle_smoke.py` into a Kaggle **Python Script**. Select T4, enable Internet and grant the existing `GITHUB_TOKEN` secret access. No script version or dataset commit needs to be entered. Launch once: the script runs the deterministic tests, real data smoke, and disk restore check, then publishes reports to a fresh GitHub results branch. Checkpoints remain in Kaggle Output; reports and logs go to GitHub.

The segmented path runs SDPA in groups of two folded windows. It explicitly
selects Flash or Efficient SDPA when supported by the inputs and GPU; otherwise
it uses Math SDPA with per-group recomputation during backward. It retains the
pinned upstream KV expansion, so native GQA copy elimination is not claimed.
The launcher logs the selected backend and per-layer allocation / running peak.

Memory CUDA training now checkpoints each full NeuralMemory call separately,
replaying it during backward. The upstream TTT boundary, update timeline and
higher-order gradients are unchanged. Reentrant replay avoids saved-tensor
hooks inside the pinned implementation's torch.func.grad. Layer 0 uses a
scalar autograd anchor because its inputs come from frozen layers. The CPU
preflight compares the pinned Memory read, blend and parameter gradients with
and without replay across a TTT boundary.

The upstream rolling accumulation re-copies the preceding timeline at each
TTT boundary. For this no-gated-transition configuration a separate optimization
could assemble all non-final timeline blocks without their final entry, then
append the complete final block in one concatenation. This change is not
combined with recomputation: replay addresses the graphs retained across the
four layers, whereas timeline collection only reduces temporary copy traffic.

For Qwen layers 4–27 the launcher now runs a standalone 16K attention
forward/backward measurement before loading the model. The kernel probes
Flash, Efficient, and cuDNN SDPA and records their eligibility, the selected
backend, and mask/layout details. Packed data uses implicit causal masking;
right-padded data uses a compact padding mask. If no eligible fused backend
can be used, 128-query blocks score the entire key sequence with an absolute
causal mask and recompute their score intermediates in backward. GQA retains
explicit KV expansion. Model projections, normalization, RoPE, and the full
receptive field remain unchanged.
