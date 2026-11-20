# MOM-INT-001 — MoM and multi-Titans integration

Owner: Continual Learning. [Protocol](PROTOCOL.md).

## Architecture audit

- Published checkpoint: `linear-moe-hub/MoM-Gated-Deltanet-340M` at
  `f4081422ea2cf01b1b5d646cac05cad62a5d5438`. Its config declares 24 layers,
  hidden size 1024, four routed memories (top two per token), one shared
  memory, four heads of width 256 and short convolutions of width four.
- Reviewed upstream source: OpenSparseLLMs/MoM at
  `033057c2f80941cb0b70ef62a99dd7ddfc94bf52`. In
  `mom_naive/layers/mom_gated_deltanet.py`, the gate consumes hidden states,
  selects top-k experts and combines their outputs. The layer stores
  expert recurrent state and Q/K/V convolution caches through the FLA
  Cache along with an offset. Router logits are outputs; no independent
  persistent router tensor was identified in this path. Audit the actual
  FLA cache version and counters before designing serialization.
- The model accepts `inputs_embeds` and `past_key_values`, but no dedicated
  external Titans input was found in the reviewed forward. A candidate
  integration is a fixed mapping from actual Titans reads into an embedding
  input, with a cloned MoM cache for each counterfactual. This has not been
  implemented or verified.
- Keep the MA-001.4 shared module and independent Titans states. The MoM
  expert gate does not replace MA-001 external Cell routing.

## Status

[Local asset preflight](artifacts/local-assets.json): **BLOCKED**. The pinned
MoM weights are not present locally; no GPU execution or FLA compatibility
test ran. Source inspection and the local asset probe do not establish
runtime compatibility or integration feasibility.

Next run the real 340M cache continuation on a compatible CUDA environment
and compare cache components and output logits before freezing the codec.
Then wire the MA-001.4 store/read APIs, staged transaction, and three ablation
paths, and verify fresh-process continuation and directory migration.
No replacement model, training, or semantic quality claim.
