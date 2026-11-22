# MOM-MULTICELL-FUSION-001

This development experiment compares the frozen MoM 340M read paths on the same
Cell snapshots, tokenizer and prompts. See [the source audit](architecture.md)
and [the acceptance record](acceptance.md).

Run the repository's `apps/mom-native-cells/kaggle_multicell_fusion.py` as a
Kaggle Python Script with one CUDA GPU, Internet access and the existing
`GITHUB_TOKEN`. The launcher verifies the pinned wheel, runs CPU tests, checks
the real model locally and across process boundaries, trains only the Fusion
Gate, evaluates all arms, profiles in a separate process and publishes the
report to a verified GitHub branch. Gating failures stop execution and return
partial diagnostics. Checkpoints remain in Kaggle Output.

## Arms

| Arm | Reads and fusion |
| --- | --- |
| E0 | Native empty-cache backbone |
| E1 | Native one-Cell reading and leave-one-Cell-out controls |
| E2 | Historical post-transform equal fusion and pre-transform hierarchical equal fusion |
| E3 | Native Top-2 reads with a query-conditioned Cell gate |
| E4 | Native Top-2 reads with factorized weights or a joint routed-candidate gate |
| E5 | Four routed units per Cell, with and without the frozen router prior |
| E6 | Null ablation of the learned variant selected on validation NLL |

For two-stage comparisons, the shared bank is weighted via the same Cell
gate rule throughout. Unit ablations remove one routed bank identity across
all Cells and renormalize eligible routed candidates. Cell ablations re-run
inference with a Cell omitted.

## Pilot protocol

The original 16-group v2 fixture is replayed in all four mapping conditions
and is kept out of Gate training and selection. A new deterministic fixture
uses disjoint identifier ranges and seeds for train, validation and test, with
2, 1 and 3 episode groups respectively. Each group includes single hop,
complementary evidence, paraphrase, abstention, and distractor queries with
1/2/4/8 Cells. Each learned arm uses 16 paired SPSA updates. The gate uses the
same initialization and data schedule across modes. This pilot is intentionally
small.

Candidate accuracy uses summed log probability with equal-length candidates.
T5 uses mean log probability for the unequal-length abstention control. Greedy
exact match is computed with a shared candidate-length generation budget per
query. Test reports include per-query scores and outputs, NLL, Cell and routed
bank ablations, interference, complementary gain, per-layer gate entropy,
candidate weights and distractor count. Replay reports additionally count
queries that follow all four swaps. Benchmarks report 1/2/4/8/16 working
states with native recurrent segments and independently imported Cell
states. Instrumented phase times and ordinary latency are measured separately.

## Runtime interface

`POST /api/read` accepts a question and generation budget and reads the
registry's Cells. The default is historical equal fusion. Configure the
server at launch with `--fusion-mode` and, for a learned mode,
`--gate-checkpoint`. `--router-prior` applies to All-4.
The legacy query endpoint and the explicit write endpoint remain available.
