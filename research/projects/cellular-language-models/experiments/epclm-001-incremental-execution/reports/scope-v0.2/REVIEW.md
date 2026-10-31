# EPCLM-001 v0.2 scope review — 2026-10-30

Decision: **scope amendment GO; formal execution NO-GO**.
This task changed the protocol, strategy and default scope only. It did not
start a formal experiment, training or a 30M run after the amendment request.
Earlier local development diagnostics are not promoted into round-1 results.
No historical E0 record, checkpoint or scientific conclusion was changed.

## Files and changes

- `PROTOCOL.v0.2.md`: active native-only scope, four-step execution plan,
  correctness/measurement gates, cost limits and conditional scaling review.
- `protocol.json`: eligible T1/X3/X4 identities, 128-position limit,
  training/formal authorization false, deferred 30M, proposed small workload.
- `PROTOCOL_DRAFT.md`: unchanged user-supplied v0.1 historical draft.
- `implementation/epclm001/assets.py`: native-only loader allowlist;
  30M is a deferred metadata entry and is not opened, hashed or loaded.
- `implementation/epclm001/cli.py`: remove 30M/C0 model selection and the
  previous draft's automatic original-model diagnostic; only native fixtures
  are eligible in the optional engineering launcher.
- `epclm001_cpu.ipynb`: remove 30M checkout path; retain optional CPU-only
  engineering and existing automatic report return, not formal execution.
- `README.md`, `FREEZE.md`, claim limits and registry: align scope/evidence.
- Tests: scope/config/CLI agreement, no deferred-model loader calls, rejection
  before opening a fake 30M file, notebook exclusion and report-only payload.

The inference/event modules were prepared under the preceding implementation
request. No existing native model or training algorithm was changed by this
scope amendment. They remain engineering tools, not evidence of actual
trained-native-model performance.

## Baselines and assets

| Model | Definition | Parameters | Trained checkpoint in inspected locations |
| --- | --- | ---: | --- |
| T1 | ModernTransformerLM, width 192, 4 blocks, GQA 6/2, SwiGLU 480, RMSNorm/RoPE | 1,894,080 | Not found |
| X3 | PhaseBLM, width 188, 3 stages × 4 iterations, SwiGLU 638, LayerNorm/RoPE | 1,893,536 | Not found |
| X4 | PhaseBLM, width 188, 3 stages × 4 iterations, SwiGLU 639, RMSNorm/RoPE | 1,893,912 | Not found |

Counts came from constructing the source definitions on CPU and counting
parameters; no forward pass or training was needed. This is configuration
inspection, not model quality/performance evidence. Vocabulary 2048 and context
128 are source definitions, not proof of recovered tokenizer identity.

Checked native experiment root and configured cloud input/output search roots;
`baseline_assets.json` records paths, availability and source hashes. Cloud
paths are not mounted here. Absence in these locations does not imply absence
from historical Kaggle versions or other remote stores.

Expected original output layouts, derived from existing runners, include:

- `implementation/results/dev/baseline-seed-91001/T1-modern-transformer/resume.pt`
- `implementation/results/dev/phase-c-baseline-seed-91001/X3-n1-swiglu-rope/resume.pt`
- `implementation/results/dev/phase-c-baseline-seed-91001/X4-n1-swiglu-rope-rmsnorm/resume.pt`
- Replication output roots chosen by the historical launcher, per seed/model.

These are search hints, not available checkpoint claims. Original runners
write `model_state_dict` plus model/profile/seed/step and corpus identity. No
actual seed/weight/precision binding can be certified without the bytes.
Source defaults name development 91001 and replication 91002/91003. Historical
confirmation seeds 91101/91102/91103 are not consumed by development.

Source training conditions: baseline 10M consumed tokens; sequence 125,
batch 8, 1,000 tokens/update; supported context 128. The development launchers
bind TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
The bare runtime has a different revision literal, corrected by its launchers.
Actual corpus manifests must therefore be recovered and compared. Shared
vocabulary size and configuration cannot establish tokenizer/data equivalence.

Reusable tools include original builders/full forward, tokenizer loader,
`evaluate`, `fixed_validation_starts`, environment recorder and analytic FLOP
estimators. Their historical throughput is not incremental decode evidence.
E0-A stays historical and deferred, with no cross-scale result pooling.

## Final order and current gates

1. Native read-only preflight: locate exact weights and tokenizer/corpus files,
   strict load and bind original metadata/hashes. **Can proceed next**.
2. E1-A + T1 correctness: freeze precision/device tolerances; validate cache,
   positions and prefix logits. **Blocked for trained native models by assets**.
3. Small E1-B: first establish four-executor semantics; freeze a small,
   native-relevant workload and separate holdout. Synthetic engineering is
   independently available; formal hardware claims are **not ready**.
4. E1-C: only after both models' correctness and common quality/measurement
   conditions pass. **NO-GO now**. Formal execution requires a separately
   reviewed frozen measurement plan; this amendment does not start it.

H1, H2 and H3 remain INCOMPLETE. Missing weights are not evidence against the
mechanisms. There is no E1 quality, performance or scale-expansion claim here.

## GPU resources and cost

This amendment uses no GPU compute. A later bounded run should use one GPU,
sequential model loads and one isolated latency worker; batch 1 and context
≤128 first. Raw ~1.9M weights require about 7.6 MB FP32 / 3.8 MB FP16, not
including framework/workspace/cache/activation overhead. This is not a peak
VRAM estimate. CUDA availability was detected, but no GPU workload was launched.

GPU-hours cannot be estimated reliably without recovered native weights and
a measured on-device point. Major costs are CLM's 12 iterations, full-prefix
reference calls, warmup/repeats, allocation/cache management, queue/packing/
gather/scatter, launch/synchronization overhead and optional profiling. Current
FLOP estimates are partial; DRAM/energy counters remain unavailable until
measured. Do not invent a speedup or a GPU-hour budget from tiny CPU fixtures.

## 30M extension trigger

Require complete 1.9M results: parity; measured incremental resource benefit;
event-runtime benefit on meaningful activity distributions; reported modern
T1/CLM gap; a scaling question and added cost. Submit all results, including
negative ones, for user review. **Only new explicit user approval may start
30M**, even if every prerequisite is met.

## Next directly executable work

Read-only recovery of the historical native run outputs and tokenizer/corpus
manifests. Generate the asset lock internally once bytes are available. If
unavailable, retain the blocker and do not retrain. Independently prepare the
small synthetic event workload, keeping it separate from real-model evidence.
