# P4-M3 Stage 3 Peer Titans Screening

80 paired CPU FP32 episodes across four configurations. The primary comparison is P2 versus P0 joint exact match.

| Workload | Union | Peer | Paired delta (95% CI) | Wrong-domain accept A / B |
|---|---:|---:|---:|---:|
| S8-A | 100.00% | 100.00% | +0.00% (+0.00%, +0.00%) | 1/160 / 2/160 |
| S8-B | 100.00% | 100.00% | +0.00% (+0.00%, +0.00%) | 1/160 / 2/160 |
| S16-A | 100.00% | 100.00% | +0.00% (+0.00%, +0.00%) | 2/320 / 3/320 |
| S16-B | 100.00% | 100.00% | +0.00% (+0.00%, +0.00%) | 2/320 / 3/320 |

## Hypotheses

H1 concerns interference and earlier associations. H2 asks whether joint accuracy benefits from separate Cells after approximate byte matching. H3 tests each branch with removal and wrong-domain queries. H4 requires exact batch/serial equivalence. H5 records scaling costs without assuming equal compute.

## Protocol and checks

S8 and S16 use nested associations from the same master mappings. Each configuration has 20 episodes over seeds allocated 7/7/6. Chunk 4 and 128 presentations per association are frozen. The two acquisition paths together use 122,880 calls and 491,520 presentations. Both modes use the same episode inputs. Mode A matches each Cell width at 64. Mode B compares a width-129 union with two width-64 Cells, 248 bytes apart in dynamic state. The memory receives keys and values during acquisition and queries alone during Recall. Candidate values and labels are used by the scorer.

P1 isolated reads, P2 batched reads and P3 serial reads match exactly. Branch ablations preserve the retained slot. Bidirectional acquisition checks and state hashes verify isolation. A different process restored all 240 checkpoints and reproduced fused and union output hashes.

## Retention, rejection and uncertainty

| Workload | Union early / late Recall | Peer early / late Recall | Unknown accept Union / Peer |
|---|---:|---:|---:|
| S8-A | 100.00% / 100.00% | 100.00% / 100.00% | 2/320 / 4/320 |
| S8-B | 100.00% / 100.00% | 100.00% / 100.00% | 1/320 / 4/320 |
| S16-A | 100.00% / 100.00% | 100.00% / 100.00% | 9/640 / 5/640 |
| S16-B | 100.00% / 100.00% | 100.00% / 100.00% | 2/640 / 5/640 |

Early/late groups use first-pass acquisition order and are scored after full replay. This is a positional retention diagnostic. Wrong-domain and independent unknown queries are distinct. The frozen acceptance threshold is maximum candidate cosine ≥ 0.8. Exact counts and per-domain intervals are in analysis.json. Bootstrap resamples complete paired episodes 10,000 times per configuration. Intervals describe within-screening episode variation at the fixed seeds.

## Resource accounting

| Workload | Dynamic bytes Union / Peer | Acquisition seconds Union / Peer | Read ms Union / batch / serial |
|---|---:|---:|---:|
| S8-A | 34,048 / 68,096 | 0.871 / 0.856 | 0.645 / 0.344 / 1.250 |
| S8-B | 67,848 / 68,096 | 0.823 / 0.820 | 0.775 / 0.342 / 1.358 |
| S16-A | 34,048 / 68,096 | 1.566 / 1.563 | 0.625 / 0.343 / 1.199 |
| S16-B | 67,848 / 68,096 | 1.620 / 1.584 | 0.761 / 0.360 / 1.331 |

Timings are CPU wall measurements of the existing helpers. The serial helper includes read-state hash checks. Batch timings include state packing. These measurements do not establish hardware overlap or equal compute. Peak RSS is the writer process high-water mark, including PyTorch. Full static, checkpoint, temporary-state and timing data are retained.

## Screening decision

Classify each observed delta using its paired interval. Practical effect size and false-acceptance tolerance were not pre-registered. Confirmation needs separate review and frozen thresholds.

Run source 6ef2e23794ef1b140ac244c22f0a07d24583285e.

- S8-A no clear advantage.
- S8-B no clear advantage.
- S16-A no clear advantage.
- S16-B no clear advantage.

Hold confirmation. The paired screening intervals do not establish a positive composition advantage in the tested workload.

## Acceptance and limitations

| Gate | Result | Evidence |
|---|---|---|
| A Integrity | PASS | Frozen source/config hashes, nested deterministic paired data, exact per-association exposure, 80/80 complete records |
| B Composition | PASS | Exact P2/P3 outputs, ordered slots and both ablations, isolated writes, 240 fresh-process restores |
| C Measurement | PASS | Full per-episode metrics, per-seed summaries, 10,000-replicate paired intervals and measured CPU resource accounting |
| D Delivery | PASS | Stage 2 restore regression, three Stage 3 tests and four separate calibration episodes passed. Raw evidence, source and integrity manifests committed on the Stage 3 branch |

The 100% joint scores saturate the tested task. The empirical paired interval [0, 0] comes from identical scores in this sample and cannot resolve differences at larger loads or smaller budgets. Early and late groups both reach the ceiling after repeated presentations. This observation does not measure forgetting in a no-replay sequential setting. The two capacity modes reuse the same master episode seeds, so repeated peer results across modes are paired observations.

CPU execution used one FP32 PyTorch thread on an Intel Core i7-4790K. Peak process RSS was 239,214,592 bytes. Mean paired checkpoint payloads were about 118.6 kB for peers versus 59.2–59.3 kB for Mode A Union and 110.1–110.2 kB for Mode B Union. The historical Stage 2 files remain the frozen baseline.
