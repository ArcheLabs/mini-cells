# R3m-E remaining Gate 2 engineering protocol

This is disposable engineering, with independent `R3ME-GATE2-ENGINEERING`
source/config/protocol identity. It does not change the frozen scientific config,
original R3m BLOCKED state, Main candidates, Confirmation or historical files.
No complete exploratory matrix is authorized. Source bytes fix the PROTOCOL
constant before any real Gate 2 run; its canonical SHA-256 is included in identity.

Use the verified Seed 151001301 R3h B-H Step 300 parent, original model,
optimizer and strict CUDA RNG restoration on two visible GPUs. Reuse existing
training assets and fixed mixed schedule. No model quality evaluation is performed
on Development; no Development batch is supplied to Proposal or Shadow acceptance.
The first eleven disposable updates follow the production continuation schedule
without compressing learning rate decay. Proposal is deliberately moved to update
1 in this separate engineering identity, so testing does not cost 100 warmup updates.

Session A reuses the actual R3m transaction engine, original Q-first Proposal,
22-scale Trust Region and ten-step paired Shadow. It pauses at relative step 4
with both branches and RNG states saved (three Shadow steps completed). It then
continues locally to relative step 11 to establish the reference endpoint. No
particular admission outcome or path improvement is required: this is engineering.

A separate two-update test injects a different Connection and a forced negative
acceptance score, then compares the actual rejected branch with a two-update
unchanged-Connection control. This deliberate fault injection is labelled as such,
is never treated as a valid Trust Region candidate, and changes no production
threshold. It tests model/optimizer rollback with exact streams and cursors.

Session A performs 26 Cell optimizer updates plus the original 8 Proposal
optimizer steps. Only the step-4 paired state and step-11 reference are exported,
with immutable receipts, trace, manifest, file sizes and SHA-256. Existing private
HF segment transport uploads them in one commit with immediate pinned readback.
No GitHub write preflight or public Checkpoint upload is needed for this check.

Session B is the same Notebook in a new container lifetime. It automatically finds
the source/config-specific HF pointer, verifies all bytes and identity, reconstructs
only a fresh engineering directory, and finishes seven main plus seven Shadow
updates. It must match the reference model and optimizer (GPU rtol 1e-5, atol 1e-7),
with exact RNG, cursor, optimizer identity, seen-batch history, schedule, counters,
Connection and acceptance decision. CPU synthetic tests use exact tensors.
Container hostname, Linux boot ID and PID 1 start ticks record lifetime provenance;
a new Python process in the same container is not accepted as a new Session.

Session A cannot claim cross-session PASS. A rerun in the same container returns
`GATE2_A_SAVED_NEW_SESSION_REQUIRED` without repeating training. Session B publishes
a small review in one HF commit. Verified subsequent runs read that review without
repeating optimizer work. Failed verification retains local evidence and never
updates the verified pointer or authorizes full training. Completed Gate 2 alone
does not authorize Gate 3. Keep the Notebook source pin unchanged across A and B.

Failed HF publication leaves a hash-checked local upload journal. Rerunning in
the same Session retries only publication, without repeating optimizer work.
