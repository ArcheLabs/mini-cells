# MINICELLS-MVP-QUALITY-001

## Repository audit and implementation plan

The T4 capacity experiment verified the exact 1.3B checkpoint, single-GPU
execution, registration of eight Cells, selection and two-Cell fusion.
Its same-seed replicas and latency trials provide no quality evidence.

| Arm | SDK execution |
| --- | --- |
| A | Separate empty registry sharing the frozen backbone and a fresh native cache for each question. |
| B | The one or two prelabelled relevant Cell IDs, or an empty selection for unrelated questions. |
| C | Two preassigned irrelevant Cell IDs from a deterministic rotation. |
| D | The SDK's auto route, with independent Shared Read Output Norm scoring, full ranking and the standard equal fusion. |

This experiment uses the internal query entry behind the public SDK so the
existing probe path can retain generated token IDs and the actual loaded
Cell IDs. The only SDK refactor extracts its existing selection and
working-cache logic for reuse. For teacher-forced scoring the experiment
replays the same prompt on fresh caches, reuses the native LM head, and
feeds one prior gold token at a time. The logits after the final prompt
token score the first target. The teacher path excludes EOS, uses no
padding, and rejects empty, unstable or truncated target encodings.
Generation follows the SDK greedy implementation and uses natural EOS
with an eight-token ceiling. Selection is completed using only the prompt.

## Frozen protocol

- Use the audited MoM 1.3B and tokenizer at
  `5c8be94196cb1827e820077680ac2ee46ac949bc`, FP16 on one Tesla T4.
- Create and independently write eight Cell Bundles. Preserve corpus,
  recurrent tensor and Bundle hashes. Import the exported Bundles through
  the same SDK and confirm hashes and persisted state after evaluation.
- The seeded generator defines eight distinct domains and disjoint prompts
  and source records. Smoke has 9 questions. Development has 40 single,
  8 cross and 8 unrelated. Held-out has 80 single, 24 cross and 24
  unrelated. The facts underlying held-out questions are present in the
  Cell writes. The questions are withheld from configuration changes.
- The selector uses Top-2 with the SDK default `None` threshold. This
  configuration will still choose up to two candidates on unrelated
  questions. Log those selections and fallbacks, plus the full rankings.
- Cross-domain questions use a two-hop record split across Cells.
  Score both one-Cell ablations and compare each pair against the best
  single on the same question. Report unrelated interference separately.
- Use 2,000 paired question bootstrap draws. The implementation freezes a
  five percentage point practical margin for automatic exact match versus
  the base model on the pooled single- and cross-Cell questions. A lower
  bound above the margin yields PASS. An upper bound at or below the margin
  yields FAIL. An overlapping interval yields INCONCLUSIVE. Incomplete or
  inconsistent full evaluation yields INVALID.
- Smoke includes repeated isolated runs with exact generation outputs and
  0.02 mean NLL tolerance. Held-out starts only after clean smoke and
  development execution. This protocol does not tune the scorer.
- Numeric accuracy, format validity, token F1, macro and micro answer
  likelihood, routing and paired differences can be recomputed from the
  published JSONL and dataset. Task scores are specific to this synthetic
  fixture with explicit domain cues.

## Run

Upload [kaggle_quality.py](kaggle_quality.py) as a Kaggle Python Script.
Select GPU T4, enable Internet and the existing `GITHUB_TOKEN` Secret.
The entry selects one GPU by UUID and runs smoke, development and held-out
in one job, then publishes the outcome using the existing publisher.
Bundles remain in `work/bundles` in Kaggle Output, with their manifests and
SHA-256 in GitHub. A publication retry uses the existing Kaggle output and
`MINICELLS_QUALITY_PUBLISH_ID`.

A compatible one-T4 environment can call `python run.py --output PATH`.
The CLI supports `--split`, `--arms`, `--evaluation`, `--seed`,
`--max-new-tokens`, and audited `--bundles` from a previous work directory.
The dataset and model revisions are fixed, validated CLI options.
`--resume` requires the same code, model, dataset and settings. Successful
rows are retained. Failed rows can retry once with both attempts preserved.
If the protocol needs correction, use a new version and output directory.

Local software tests use CPU fixtures. A product quality verdict requires
the actual T4 held-out artifact.
