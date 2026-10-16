# Kaggle continuation engineering

**Scope:** infrastructure only, based on frozen R3 `52b1950b23debbcd3f808c10eb6e0d3ea2d317c6`.
No R3 retraining, no R3b training authorization. Original R3 code/config/Notebook/results are unchanged.

## Status

- Implementation: continuation stores, segmented adapter, exact-state validation, historical evaluation, explicit forks and metadata-only publication.
- CPU verification: see `reports/engineering.json` and Actions.
- **KAGGLE_CROSS_SESSION_RESUME=NOT_VERIFIED** until actual A/B/C versions are run and reviewed. Local directory copies never upgrade this status.

## Reuse and identity

`shared/python/minicells_runtime` orchestrates the original R3 **Model, Trainer.update, Checkpoint save/restore, Event process, Nursery, asset loader, frozen probe bank, evaluator and publication helpers**. It does not implement a new model/loss/router/admission algorithm. `run.py` only sets existing package paths.

The synthetic acceptance run uses an explicit engineering envelope `MINICELLS-KAGGLE-CONTINUATION-SMOKE-V1`. Inner algorithm fixtures preserve real R3 names/identity fields (Q-R3/QC-R3/QS-R3); this is not R3 scientific data. `engineering_run_id` binds each new checkpoint to its distinct run. Historical R3 import never adds this field to old identities.

There is one shared `assets/`, all immutable historical `.pt` + hash sidecars, all initial probes, frozen runtime/candidate config, protocol, preflight evidence, private diagnostics and per-arm event state. No per-arm portable asset duplicates are created. `continuation.json` is committed **last**, points to a content-addressed manifest, and indexes exact paths/event revisions. Small mutable selector/status records are versioned separately, so a failed next manifest does not invalidate the previous commit. Weight/data files remain single-copy.

Default capacity ≤16 GiB is a project safety margin, not a claim about a platform hard quota. Total actual files, old manifests and metadata versions are counted, with conservative allowance for the last pointer. Exceeding the limit fails with a largest-files breakdown. Nothing is silently deleted.

## Minimal real Kaggle acceptance (CPU, no GPU allocation)

Use `notebook.ipynb` in a **separate engineering Notebook**, leaving an active R3 session untouched. It checks out a full fixed SHA. Kaggle runs only the small integration fixture, **not the unit-test suite**, and does not download TinyStories or run a gradient Pilot.

1. **Version A**: defaults (`ACTION='train'`, no input, stop=6). Save & Run All. Output must contain `minicells-engineering-run/continuation.json`, status `PAUSED`, three arms at 6.
2. **Version B**: Add Input → Notebook Output Files → **A's exact saved version**. Set `RESUME_INPUT` to its mounted `.../minicells-engineering-run` directory; `REQUIRE_RESUME=True`, stop=10. Run All. Logs must say `RESTORED_FROM_STEP=6`, `STARTED_FROM_RANDOM_INIT=false`; no CPU suite, Pilot or beta recalibration occurs.
3. **Version C**: Add B's exact Output. Set `ACTION='evaluate'`, `RESUME_INPUT` to B's directory, `EVALUATE_STEP=6`. Run All. This only evaluates the historical checkpoint and runs a tiny CPU uninterrupted 0→10 control for state comparison. It never takes an optimizer update on the evaluated model or modifies Input. Control training is isolated synthetic engineering work, not an R3/R3b run.
4. Supply A/B/C **version links (`scriptVersionId`)**, mount paths and the saved evaluation/control/continuation metadata. We then verify A→B and B→C provenance before marking the real gate PASS. The Notebook never fabricates a version number or upgrades itself from local evidence.

If Kaggle disallows a Notebook depending on its own historical Output, use two alternating engineering Notebooks with the same template. No local download or cloud credentials are required. CPU fixture steps are 0→6→10, **not** the proposed real 0→200→350→500 experiment.

`/kaggle/working` is local until a version successfully saves and becomes mountable by a later session. No claim of live remote durability or recovery after an unsaved cloud failure is made.

## CLI

```bash
ENTRY=research/projects/cellular-language-models/engineering/kaggle-continuation/run.py
python "$ENTRY" --mode auto --output /tmp/segment-a --stop-after-step 6
python "$ENTRY" --mode verify-checkpoint --checkpoint-input /tmp/segment-a
python "$ENTRY" --mode auto --checkpoint-input /tmp/segment-a --require-resume true --output /tmp/segment-b --stop-after-step 10
python "$ENTRY" --mode evaluate-checkpoint --checkpoint-input /tmp/segment-b --arm QC-R3 --step 6 --output /tmp/eval
python "$ENTRY" --mode compare-control --checkpoint-input /tmp/segment-b --output /tmp/control-comparison
```

`stop-after-step` is a segment bound outside frozen config/hash. The total target remains 10 in the smoke. Each arm has its own reliable cursor. A satisfied segment is skipped; repeated same-parent/same-output resume returns existing verified state. Missing Input under require-resume=true fails closed. Providing Input under require-resume=false also fails.

Historical selection defaults to the **latest indexed committed event revision of the requested Step**, or use `--revision` for an exact revision. Every path/byte/identity is checked before torch deserialization. Only trusted project-private Outputs are accepted; SHA256 is integrity checking, **not authentication of arbitrary malicious pickle**.

Historical R3 without a continuation envelope:

```bash
python "$ENTRY" --mode import-r3-output --checkpoint-input /mounted/old-r3-output --assets /mounted/old-assets --config /frozen/candidate-config.json --source-sha ORIGINAL_FULL_SHA --output /working/adopted-readonly
python "$ENTRY" --mode evaluate-checkpoint --checkpoint-input /working/adopted-readonly --arm QC-R3 --step 100 --output /working/retrospective-eval
```

Import requires original runtime/gate evidence, source/protocol/config/data/tokenizer identities and initial probes. It does not rewrite old checkpoint bytes or run new preflight. A different source cannot resume it; a reviewed migration or explicit fork is needed. Posthoc evaluation is always labeled retrospective/exploratory and cannot recreate missing native R3 snapshots.

Fork requires **`--new-config`, `--new-run-id`, `--optimizer-policy inherit|reset`**. It records exact parent checkpoint SHA/step/arm/run, new source/config/protocol, inherited probe provenance and optimizer policy, then saves a new checkpoint without training. Model/data/probe changes require a separately reviewed migration. Event IDs are rebound explicitly to the new child identity; parent state remains untouched. Reset policy retains historical journals/data cursors but resets optimizer and independent policy RNG to the new configuration, as recorded in provenance.

## Scientific execution remains gated

The provided Notebook/`auto` starts **only synthetic engineering fixtures**. The reusable adapter supports frozen real R3-compatible runtime state and retains original R3 authorization + gradient/alignment checks and two T4 queues (Q→QS / QC). A real new-run adapter additionally requires reviewed cross-session PASS evidence bound to the exact source/runtime. There is no command here to bypass R3 authorization or start R3b. Integration of a future R3b-specific scientific configuration is outside this engineering change.

## Diagnostics and performance

New engineering runs freeze a nonempty plan including Step0 and final Step, with `periodic_diagnostics_enabled=true` before asset creation. Light scheduled diagnostics collect greedy/sampled/uniform quality/cost/path/STOP/CTX/self-loop/exposure plus frozen-state Q ranks/STOP gaps and recorded Q/LCG gradient norms/cosine. Expensive R3 functional/crossover/counterfactual diagnostics run only at `counterfactual_snapshots`. Fixed initial probes are never regenerated after restore.

Snapshots bind source/config/probe/checkpoint hashes, sample count, timestamp and measured duration. They are atomic, sidecar-verified and idempotent. A snapshot is generated after all events at its Step; diagnostic RNG is isolated from training RNG. `resource-estimate.json` reports observed and remaining diagnostic costs; unmeasured classes are explicit rather than silently assigned zero. Output capacity includes private diagnostics. No runtime kill is introduced.

## Publication

`--mode publish-only --checkpoint-input <bundle>` reuses the existing authenticated GitHub helpers and publishes **only an explicit aggregate schema** (status, hashes and per-arm progress). Never checkpoints, optimizer/RNG, assets, per-sample scores, Oracle labels or credentials. A paused bundle cannot claim complete or H1/H2 support. Missing Secrets/push failures do not affect a committed continuation; retry publication without training. Kaggle publication is optional for the synthetic fixture; saved Output remains primary.

## CPU validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/engineering tests/research/clm001_r3
```

The CI workflow also checks the original R3 directory against baseline. No production GPU run is part of CI or implementation testing.
