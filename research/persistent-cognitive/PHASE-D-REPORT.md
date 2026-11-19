# Phase D verification

Cell and Session use separate schemas and atomic directory snapshots.
Each manifest covers the SHA256 of the safetensors payload and has its
own integrity hash. Cell additionally hashes tensor state and counters.
Restore checks the allowlist, layout, dtype, finite values, model
fingerprints and revision. Legacy artifacts require a separate reader.

Session restore accepts a trusted cache layout from its caller and
fills a fresh cache directly. Expected Cell revision guards publication
within a CellSlot. Session advancement stages changes in an independent
cache and verifies its Cell binding and the caller's model fingerprints.
Fork creates new tensor storage and records the parent revision.

Four CPU tests passed with PyTorch and safetensors 0.6.2. The dedicated
Phase D fixture covers revision conflicts, snapshot independence,
corruption and cross-schema rejection. Session restore was verified
with a synthetic cache layout. Pretrained Qwen restart and Memory-on
fusion/addressing/cross-window retrieval remain open. Phase A/B remains
PARTIAL. No training or CUDA optimization was performed.

The original pretrained T4 smoke passed at result commit 79ed540, with
zero logit error after same-process restore. The separate restart
notebook now waits for Source to exit before Target loads the pinned
model and restores Cell and Session independently. It compares the
next-token logits and committed state digest. The reference tensor
layout comes from an unrelated shape probe. T4 restart results remain
pending. Long-range recall is a separate acceptance item.
