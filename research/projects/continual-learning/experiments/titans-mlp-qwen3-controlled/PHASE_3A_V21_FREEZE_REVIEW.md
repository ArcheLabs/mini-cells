# Phase 3A v2.1 freeze review

**NOT_FROZEN. Formal Test remains SEALED.**

The [baseline results](https://github.com/ArcheLabs/mini-cells/tree/3e7335ea073aeb3c02ebef15d842f98b23238af5/artifacts/experiments/titans-mlp-qwen3-controlled/phase3a-v21-validation-w3zwp8u_) came from source `5afbdef` on a Tesla T4 with 1024 training and 256 validation facts. All five listed artifacts were independently read back and matched their declared sizes and SHA256 hashes. Shared Linear remains the primary candidate and both Separate controls are retained. The old pooled FPR was computed on calibration data and cannot establish generalization.

## Independent rejection audit

Complementary four-relation profiles preserve balance in a deterministic
32/16/16 entity split at full scale. Selection chooses checkpoints,
Calibration fits thresholds, and Audit scores them without retuning.
The reduced CPU run follows the same design with 4/2/2 entities.
Each rejection partition uses its own equally sized A1 write gallery.
There is no Family C rendering. Unknown-entity and unknown-relation
fixtures are unique within each family.

Global max cosine is the deployment reference. Family thresholds require
a trusted observable template ID, such as one emitted by a controlled
query frontend. They are diagnostic for unlabelled queries and cannot be
transferred to C by using a hidden family label. The top-two score margin
is a family-independent alternative. Curves sweep budgets on Calibration
and report Audit FPR, coverage and risk. No audit point selects a new cutoff.
Wilson intervals are query-level descriptive uncertainty. Repeated queries
for a fact across templates are correlated, so pooled intervals cannot
replace seed-level inference.

## Acceptance matrix and owner decisions

| Dimension | Existing rule | Baseline evidence | Proposed decision |
| --- | --- | --- | --- |
| Seen 32/128 | 90% at 32 | Seed 42 only | Keep 32 mandatory, report 128 diagnostically |
| Held Query | 80% at 32; paired gain at least 10 points and CI lower bound above zero | Shared 94.9% at 32, 85.9% at 128 | Retain, use paired seed-level intervals |
| Held Write 32/128 | Report | Seed 42 validation only | Diagnostic, no new threshold |
| Held Both | 80% at 32; paired gain at least 10 points and CI lower bound above zero | Shared 95.7% at 32, 89.8% at 128 | Retain |
| Scaling | 80/70/50% at 32/128/512 | 512 unavailable in validation | 512 reserved for approved formal Test |
| Value | 90/80/60% at 32/128/512 | Strong seed 42 validation retrieval | Retain and require all configured levels |
| Unknown FPR | Aggregate at most 5% | Old result in-sample | Retain; report family and type with intervals |
| Coverage and selective recall | No approved floor | B coverage low at the old pooled cutoff | Propose coverage 80% and selective Recall@1 90% on independent queries, pending owner approval |
| Independent audit | No retuning | New CPU evidence is a plumbing test | Mandatory; use matching gallery sizes when transferring a cutoff |
| Reproducibility | Five independent seeds | One prior full-scale seed | Mandatory |

An 80% coverage and 90% selective accuracy target implies at least 72%
correct-and-accepted known queries. The floor expresses an operational
minimum rather than a fitted threshold. A separate family FPR cap should
remain diagnostic until the owner approves both a cap and adequate sample
sizes. For example, even 0/64 observations leaves a Wilson upper limit
above 5%; correlated template views further limit precision.

## Execution and remaining work

`phase3a_five_seed.py` implements independent sealed validation workers
for 42–46 and seed-level means, SD and t intervals, with paired recall
gains versus Raw Cosine. It requires the approved configuration digest.
The digest is a reproducibility check; owner approval is a procedural
requirement. The five-seed job has not been launched. Each worker reports
why 512 candidates are unavailable in validation. The final Family C
evaluation remains separate and needs approved post-freeze wiring,
including candidate-count-matched rejection calibration and publication
of the full five-seed bundle.

Before formal execution, the owner must approve the primary deployment
calibrator, coverage and selective-accuracy floors, how family-level
failures affect acceptance, and the final configuration digest.

## Independent CPU smoke results

Seed 42, 32 train facts, 32 validation facts, 8-way rejection gallery.
These measurements do not establish full-scale acceptance.

| Adapter | Cutoff | FPR | Coverage | Selective R@1 |
| --- | --- | --- | --- | --- |
| Shared Linear | global | 2.08% | 0.00% | undefined |
| Shared Linear | family | 0.00% | 0.00% | undefined |
| Shared Linear | margin | 4.17% | 4.17% | 100.00% |
| Separate Linear | global | 4.17% | 4.17% | 100.00% |
| Separate Linear | family | 0.00% | 4.17% | 100.00% |
| Separate Linear | margin | 8.33% | 4.17% | 100.00% |
| Separate MLP | global | 8.33% | 4.17% | 100.00% |
| Separate MLP | family | 10.42% | 8.33% | 50.00% |
| Separate MLP | margin | 6.25% | 8.33% | 100.00% |

[Raw audit and curves](smoke/cpu/independent-rejection-audit.json)
