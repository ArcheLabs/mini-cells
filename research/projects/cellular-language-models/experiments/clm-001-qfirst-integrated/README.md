# CLM-001-QFIRST — integrated cost-to-go training

Status: engineering implementation; Gate 0/1 synthetic CPU validation. **No GPU execution or confirmation authorization.** [SPEC.md](SPEC.md) is the unmodified supplied protocol.

## Research question
Can state-conditioned cost-to-go supervision, true hard execution, self-connections and real STOP jointly produce useful paths, with a bounded isolated Shadow lifecycle? This is exploratory; training completion, Q fit, Cell IDs or promotion alone cannot establish the scientific claim.

## Run now
Import [notebook.ipynb](notebook.ipynb) into the original `bingjiangli/minicells` Kaggle Notebook and Run All. The default is **CPU preflight only**, using synthetic data. Existing `GITHUB_TOKEN` returns small metadata; full checkpoints remain in saved Kaggle Output. No new Kaggle Notebook or Action-to-Kaggle API is needed. Failed tests stop the next phase.

Local CPU equivalent from repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/run.py --mode cpu-preflight --output /tmp/clm001-cpu
```

No manual source hash, asset hash or seed form is needed. Setup resolves its research branch once and records the full commit. A reviewed execution is pinned to that recorded source.

## Two arms only
- `QF-FULL`: actual live CE updates continuous computation; inverse conditional-inclusion weighted Huber on detached pre-action states updates Connection. Auxiliary matched critic is trained on counterfactual returns but never chooses the action.
- `J0-FULL`: the same inventory, initialization, budget and candidate governance; existing PD J0 path-level actor/critic. Candidate scores never directly fit its formal router.
- Each arm: **1,075,492 parameters**, six initially accepted Cells, one resident reserved Cell, input-only causal memory, tied vocabulary head, sixteenth-step forced termination distinct from earlier sampled STOP. STOP is illegal before one real action. All Cell/CTX calls cost `1/16`; STOP costs zero.

## Where to inspect
- `implementation/clm001/`: executor, Connection, snapshot Oracle, training, data isolation, nursery/admission, checkpoints/portable, evaluation/accounting/report, existing-publisher adapter, CLI/Pilot.
- `config/proposed.json`: explicit unfrozen configuration, not training permission.
- `reports/`: CPU evidence and immutable input hashes.
- [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md), [ORACLE_TRUST_GATE.md](ORACLE_TRUST_GATE.md), [SHADOW_ISOLATION_AUDIT.md](SHADOW_ISOLATION_AUDIT.md), [PERFORMANCE_BUDGET.md](PERFORMANCE_BUDGET.md).
- Tests: `tests/research/clm001_qfirst`; CI: `.github/workflows/clm001-qfirst-cpu.yml`.

## Later GPU sequence — separate approvals required
1. Review CPU evidence and freeze the Pilot source/config; explicitly authorize GPU Pilot. The approval configuration is generated after that review, not by toggling Notebook booleans.
2. Run the approved `gpu-pilot`: frozen fresh-model Q-calibration labels, repeatability, isolated real-Train timing updates and private Shadow/Admission/checkpoint timings. No old FAST checkpoint resumes a new Q-first optimizer.
3. Review measured cost/variance. Freeze real source/asset/tokenizer identities, lambda, epsilon/temperature, thresholds/CI, seed and data budget. Explicitly authorize 50-step smoke. Do not reduce the proposed scientific budget silently if timing is too slow.
4. Save its Output; attach it in a fresh Kaggle session and verify restore/trajectory reproducibility. Checksums alone do not set `portable_pass`.
5. Only after that review and explicit 500-step approval, run both arms concurrently, one per T4. All candidate work stays on its own arm's device. No process killing or application wall-clock cap was added.

`publish-existing` recovers metadata return from retained Output without training again. Publishing uses current-main ancestry and stages result JSON only; no workflow permission, model weights, tokens or private secrets are included. Save Version remains necessary for portable checkpoint persistence.

## Limits
The labels are Monte Carlo estimates, not an exact Oracle on real language. The QF continuous update intentionally omits policy-state gradients, per protocol; J0 retains its original path-level estimator. A fixed target-independent Shadow path prefix is a candidate policy, not learned path search. Logical call cost is not measured latency. No GPU speed or language advantage has yet been measured.
