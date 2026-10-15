# CLM-001-QFIRST — integrated cost-to-go training

Status: automatic exploratory development launcher. Coding-agent verification uses synthetic CPU only; Kaggle Run All performs preflight and then development. [SPEC.md](SPEC.md) is the unmodified supplied protocol.

## Research question
Can state-conditioned cost-to-go supervision, true hard execution, self-connections and real STOP jointly produce useful paths, with a bounded isolated Shadow lifecycle? This is exploratory; training completion, Q fit, Cell IDs or promotion alone cannot establish the scientific claim.

## Run now
Import [notebook.ipynb](notebook.ipynb) into the existing `bingjiangli/minicells` Notebook, select **T4 x2**, keep `GITHUB_TOKEN` / `HF_TOKEN` Secrets, and **Run All**.

Automatic CPU correctness checks → GPU availability / existing GitHub return preflight → isolated assets and identity → bounded real-model Oracle / checkpoint continuation checks → QF-FULL and J0-FULL concurrently, one per GPU → analysis and GitHub metadata return. Failed checks stop the next step. No manual hashes, approval configuration, mode switches or second session.

The declared 500-step live budget is preserved. A measured advance projection above two hours refuses to start full training; there is no process kill, runtime cap or silent budget reduction. This is not a total-duration guarantee.

CPU-only engineering verification / CI remains available:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/run.py --mode cpu-preflight --output /tmp/clm001-cpu
```

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

## Active automatic workflow
The current user instruction supersedes manual launch approvals in the original [SPEC.md](SPEC.md); its text remains intact. See [EXECUTION_WORKFLOW.md](EXECUTION_WORKFLOW.md).

Runtime lambda uses the declared candidate **0.01**, recorded as development configuration. Scientific thresholds remain unfrozen and results cannot be declared preregistered or confirmed. Unset Admission margins permit measurement/probation but **no automatic Shadow promotion**; the reason is recorded instead of inventing an acceptance threshold.

Save/restore/continuation is tested automatically in the bounded preflight. It is a local restart check, **not a claim of real cross-session Kaggle persistence**. Save Version still retains full checkpoints. Reopening the same source/output resumes checkpoints and skips completed training. `publish-existing` remains available through the CLI to return retained results.

## Limits
The labels are Monte Carlo estimates, not an exact Oracle on real language. The QF continuous update intentionally omits policy-state gradients, per protocol; J0 retains its original path-level estimator. A fixed target-independent Shadow path prefix is a candidate policy, not learned path search. Logical call cost is not measured latency. No GPU speed or language advantage has yet been measured.
