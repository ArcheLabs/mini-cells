# R3b design/reuse audit

Base main:2e47a28a27a0932912673a1140e533e12abcc00a (fetched over SSH, not assumed).
Original R3 implementation/config/SPEC:52b1950b23debbcd3f808c10eb6e0d3ea2d317c6.
Historical result files already merged into main are preserved. CI compares code
with52b and the **whole historical R3 directory** with the starting main; comparing
all newly merged artifacts with the old code-only commit would be a false failure.

| Capability | Reused | New work |
|---|---|---|
| Real hard paths/Q/LCG/zero Sham | clm001_r3 Model/Trainer.update | explicit research-to-engine aliases; no trainer copy or loss changes |
| Oracle/Shadow/proposals/Admission/promotion | original events.process/pending/Nursery/checkpoint | after-commit handoff callback; pre/post observation labels |
| Complete checkpoint state | original checkpoint.save/restore | new R3b identity and historical selectors |
| Byte manifests/size checks/event inventory/state fingerprints | minicells_runtime.r3.package and store | optional adapter injection; default old81 behavior preserved |
| Private HF transport and exact revisions | HFCheckpointStore/HubBackend | event-cursor monotonic check and transparent transport timers/byte ledger |
| GPU queues | original Q→QS,QC arrangement | independent spawned workers, private live trees, immutable handoff, single publisher ACK |
| Probes/fixed suffix/crossover/bootstrap | original capture/bank/fixed_suffix/Snapshot.relative/crossover/contrast | independent frozen bank, full Cell rank, STOP/CTX calibration and isolated snapshots |
| Return path | original R3 prepare/helper/push from main | bounded explicit JSON whitelist and separate HF/GitHub status |

Added observations preserve global Python/NumPy/Torch/CUDA generators, all Trainer
streams and every module training flag. Existing lifecycle operations are NOT
wrapped in that isolation: their original state transitions must persist.
Aliasing avoids silently disabling the original hard-coded QC-R3 LCG branch.

A coordinator never reads another process's mutable latest to form a bundle. It
copies/hash-checks a worker-frozen snapshot and merges latest last. Workers wait
for an HF acknowledgment before another update/event. On remote failure the last
HF index remains valid, pending workers are gracefully released, and no process
is terminated. The GPU host hides CUDA; fresh workers each expose one assigned T4,
so checkpoint restore can't cross-contaminate another worker's CUDA RNG.

Current Gate C remains provenance-incomplete in reports/engineering-acceptance.json.
User success is recorded, but the provided push receipt and DURABLE_COMPLETE do
not prove the independent C state comparison. No new engineering rerun is started.
Current environment has no usable Kaggle CUDA session/Secrets; minimal Pilot is
implemented but actual GPU times/production upload capacity are NOT measured.

The Pilot is a distinct **CLM-001-R3b-GPU-PILOT** run, three original private Train
updates per arm plus one uninterrupted/restored update comparison, with candidate
states for realistic checkpoint size. It contains no GPU fault suite or TorchDispatch.
Beta selection remains the original Train-gradient rule; Calibration direction audit
is separate. Snapshot timing is Calibration-only and its Development extrapolation
is labeled. Gate D cannot be PASS until production/event/bandwidth budget review;
Gate E requires separate explicit500-step authorization and matching identities.

No real assets are claimed frozen yet. BPE reconstruction must reproduce the
required historical tokenizer hash; otherwise the launcher stops and requests a
trusted original HF asset reference rather than silently replacing it. Old R3
checkpoints have not been migrated or reused as R3b training initialization.
