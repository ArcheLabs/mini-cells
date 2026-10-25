# D1 review corrections 001

Status: `CORRECTIONS_IMPLEMENTED_INDEPENDENT_REVIEW_REQUIRED`. GPU remains unauthorized; the Kaggle Notebook defaults to `RUN_BCD=False`.

## P1: durable report recovery

D1 checkpoint schema v2 stores its arm identity, schedule and report snapshot together with full trainer state. Resume reconstructs the D1 curve exclusively from checkpoint `trainer.curve` entries after the Step300 anchor and requires contiguous steps. Parent history is never included. A stale report must be an exact prefix; conflicting history is rejected. Evaluations are reconciled from the checkpoint snapshot and result file, matched to the bytes of their own checkpoints, and sorted without duplicate boundaries. A missing current-boundary evaluation is recomputed without repeating an optimizer update. Missing earlier evidence blocks recovery.

Tests restart from the manifest read back from disk after each injected interruption: (1) checkpoint saved before evaluation/result, with stale result; (2) same point with missing result; (3) evaluation completed before result save; (4) result saved before manifest save. Recovered curves contain exactly the synthetic D1 steps301/302, all evaluation barriers are present, and model/AdamW/RNG/cursor match uninterrupted execution. Opening a completed run again performs zero further updates.

## P1: actual execution path

The production B/C/D loop is extracted into `_execute_run`; production identity/asset/CUDA admission remains in `run_bcd`. CPU tests use the same loop and actual D1 class with a 3,118-parameter synthetic configuration and eight synthetic records. They execute actual forward/backward, restored nonempty AdamW state, keyed routes, independent QA forwards, fresh predecessors, zero-based positions, real free generation and full checkpoint roundtrips. A direct `run_bcd` admission test verifies that a code-content mismatch fails before loading a parent model.

These tests perform synthetic CPU optimizer updates. They do not load or train the original R3h checkpoint or access Tiny/Main/Confirmation model evaluation. Default CPU Gate remains a separate process that imports no Torch and makes no optimizer updates. CUDA correctness and real B/C/D execution are not claimed.

## P2: source identity

The CPU Gate and training launcher reject tracked working-tree changes and untracked Python source. The D1 code-content hash must match the preflight manifest, belongs to the run directory identity and is required in run/checkpoint identities. Resume therefore rejects a changed content hash even under an unchanged source SHA. Ancestor code is bound by the Git source SHA and the clean-checkout check. The Notebook also checks tracked checkout cleanliness before launch.

## P2: historical A sampling declaration

The common-draw field now explicitly lists B/C/D. A is `HISTORICAL_REUSED_ENDPOINT_NOT_CERTIFIED`. Configuration identity changes from `fc4b77e91714a95cca1bdbf6793bb0171ff9df1b4d3e69dacbdb388b6598ba46` to `656da576ad7e6908825c9c496a5434155957f591aadcf4fee00ddbd14f67648b`. Old engineering evidence retains the old identity. Facts, tokenizer, anchor, freeze rules, objectives, updates and evaluation barriers remain as specified.

## Kaggle entry and verification limits

The borrowed anchor loader also expects a `tokens_per_step` field; its adapter now binds this explicitly to the frozen 512 source targets, and a simulated-admission CPU test checks the loader contract and receipt/state checks without CUDA or remote I/O. The training bootstrap requires `r3m/run.py` plus the reused 006/005 and base CLM import paths; these are now included in sparse checkout. Import validation uses the same Notebook sparse patterns locally. No test suite is added to the Kaggle launch. The user-supplied previous log verifies only the `b7685aeb` CPU Gate; revised-source and GPU cloud validation are not inferred from it.

The engineering evidence accompanying the revised source records CPU test outcomes, the fresh data/budget proof and code/config identity. These corrections are implementation evidence for independent re-review, not scientific approval or a GPU release.

## Recorded verification

Revised execution source: `7b3643461cead09285d2bbd1fdb417eb36597191`. All 19 CPU tests pass, including the actual-model synthetic tests and four durable-write interruption cases. The local checkout using the exact Notebook sparse patterns imports the full reused runtime and completes the default CPU Gate. The fresh sample plan and budget totals exactly match the prior plan. Machine-readable evidence is in [`engineering/45c0578ffce9fc5addfee82c2f1d06d4/REVIEW_REPAIR_PROOF.json`](engineering/45c0578ffce9fc5addfee82c2f1d06d4/REVIEW_REPAIR_PROOF.json), with test logs and artifact hashes in the same directory. The Notebook now pins this execution source and retains `RUN_BCD=False`. No GPU run or scientific optimizer update was performed.
