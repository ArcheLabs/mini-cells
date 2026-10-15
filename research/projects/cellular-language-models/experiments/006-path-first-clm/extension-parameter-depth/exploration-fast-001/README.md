# FAST-001 — small joint CLM × MoE development environment

ID: `006-PD1-FAST-001`. Status: **ENGINEERING_PROPOSED**. This change authorizes CPU engineering only.
The full protocol is [SPEC_FAST001.md](SPEC_FAST001.md). Prior PD-001 models, defaults, results and checkpoints are unchanged.

## Exact matrix and candidate geometry

| Arm | Actual parameters | Relative to MoE | Structure |
|---|---:|---:|---|
| M0-TINY | 2,013,440 | 100% | 128 width, 6 attention/MoE layers, 4 experts, Top-2, expert hidden 176 |
| CLM-S8 | 1,011,000 | 50.2126% | 6 shared Cells, hidden 248, 8 real actions |
| CLM-S16 | 1,011,000 | 50.2126% | identical parameters and initialization, 16 real actions |

Shared: vocab 4096, sequence 64, effective batch 8, microbatch 1, FP32, alpha .125, step capacity 16.
The default candidate budget is 500 steps = **256,000 target tokens per arm**. Only joint J0 is enabled.
STOP has no trainable action rows. A constant masked compatibility slot lets the unchanged PD executor run.
S8's reserved later step rows are the same capacity as S16, rather than padding one arm to fake equality.
Cell repetition and CTX_READ each consume one action; a single AdamW updates the entire visited graph.

## CPU engineering now

Run from repository root; replace `$FAST` below with this experiment directory (no shell variables are required):

```bash
python $FAST/run_fast001.py --mode params --all-arms
python $FAST/run_fast001.py --mode search --output /tmp/fast001-balanced
python $FAST/run_fast001.py --mode preflight --output /tmp/fast001-cpu
python -m pytest -q tests/research/path_first_fast001
```

`search` writes a candidate into a fresh output directory; it never replaces protocol/config automatically.
A failed gate is an error, not permission to train. CI uses CPU wheels and synthetic stories. The real
TinyStories shard, independent confirmation stories and old token/checkpoint files are not read by CPU Gate.

## Explicit user stages on Kaggle

Import `notebook.ipynb` into the existing Notebook. **Default Run All is CPU preflight**. No GPU training begins.

1. After review, set `MODE='benchmark'`, `ALLOW_GPU_PILOT=True`. One idle T4 measures all three arms serially,
   microbatch 1 and 2, 3 warmup + 10 measured steps and one separate offline profiler step per case.
   This stage is not a language experiment and never auto-transitions to training.
2. Inspect `pilot/pilot.json`: projected three-arm time is `max(S16, S8+M0)` using p90 steps, plus 50% and
   300 seconds for evaluation/setup/saving. It is an estimate, **not a 30-minute guarantee**. If over target,
   review bottlenecks; the code does not silently reduce steps or change batch/model/loss.
3. Only after user approval set `MODE='train-matrix'`, `APPROVE_DEVELOPMENT=True`.
   For an intentionally accepted over-target projection, also set `ACCEPT_SLOWER_THAN_TARGET=True`.
   The Notebook freezes source/config/pilot, prepares new data, starts S16 and S8 on separate T4s, then M0
   on the next free GPU. All workers join before serial evaluation/profile/interventions/report.
4. Retain Output with Save Version. The run creates a portable hash-verified checkpoint bundle. To restore
   in a new cloud session, attach it as input, set `RESTORE_FROM` and `RESUME=True`. Source/config/asset
   identity must match. A completed arm is reused and never optimized again.

No GitHub or Kaggle API permission is needed for this new engineering entry. The final reports and portable
bundle are in retained Kaggle Output. Only small reviewed metadata/reports may be committed to Git; no corpus,
`.npy` token arrays, `.pt` weights, raw profiler traces or secret environment values belong there.

### Equivalent CLI

```bash
python $FAST/run_fast001.py --mode benchmark --device cuda:0 --allow-gpu-pilot --output /kaggle/working/fast-pilot
python $FAST/run_fast001.py --mode freeze --pilot /kaggle/working/fast-pilot/pilot.json --approve-development --output /kaggle/working/fast-frozen.json
python $FAST/run_fast001.py --mode train-matrix --config /kaggle/working/fast-frozen.json --arms CLM-S16 CLM-S8 M0-TINY --devices cuda:0 cuda:1 --output /kaggle/working/fast-development
python $FAST/run_fast001.py --mode evaluate --all-arms --config /kaggle/working/fast-frozen.json --assets /kaggle/working/fast-development/assets --work /kaggle/working/fast-development --device cuda:0
python $FAST/run_fast001.py --mode intervene --arms CLM-S8 CLM-S16 --config /kaggle/working/fast-frozen.json --assets /kaggle/working/fast-development/assets --work /kaggle/working/fast-development --device cuda:0
python $FAST/run_fast001.py --mode report --config /kaggle/working/fast-frozen.json --work /kaggle/working/fast-development
```

Use `--resume` for existing runs. Optional `--max-session-seconds N` requests a completed-step checkpoint pause,
not a hard stop; it defaults to None. A one-GPU CLI schedule uses `--devices cuda:0` and runs serially. The default
Notebook expects two T4s for training; insufficient memory/hardware produces an actionable error instead of
changing the scientific budget. A partial/failed matrix stays `INCOMPLETE` and never produces a three-arm win.

## Data isolation and remaining freeze items

Candidate raw train rows `[0,8192)`, development `[50000,50512)` of the pinned TinyStories **train** shard.
Selection uses raw row IDs, deterministic normalized-word 5gram 64-bit SimHash (Hamming <=3) plus exact normalized
content dedup across both selections. Duplicates are removed, never replenished from undeclared rows. All retained
raw IDs, actual story counts, content hashes, tokenizer, windows and source checksum are written into a new manifest.
Near-duplicate checking is this specific deterministic criterion, not a claim to detect all semantic paraphrases.
Tokenizer fits training text only. Shifted target windows never cross stories. Frozen evaluation selects distinct
stories first (up to 256 windows); midpoints use the first 128 of the same selection. It does not read validation
split confirmation used by 005 or train rows >=60000 reserved by 006/PD.

Real source/tokenizer/window hashes and final retained story counts remain **unmeasured until asset preparation**.
Real T4 throughput/peak memory and the 30-minute objective remain unmeasured until the user-authorized pilot.
The default microbatch 1 is a candidate; microbatch 2 is tested but not automatically adopted. Changing it requires
another matching config-bound pilot/freeze. Cross-session Kaggle retention remains to be verified by a real saved
Output restore; CPU portability tests alone do not prove the cloud retained files.

## Interpretation

Main results are token-aggregated greedy development NLL/PPL, sampled-minus-greedy gap, S16-M0 and S16-S8 NLL
differences with story-paired bootstrap (1000 replicates). Single-seed uncertainty is not across-seed uncertainty.
The baseline learning check freezes an exploratory minimum .1 train-CE drop and .1 dev-NLL drop. A weak baseline
is `UNDERPOWERED_BASELINE`. The descriptive proximity tolerance .02 NLL is **not** a noninferiority margin.
Interventions are symmetric for S8/S16: edge removal/shuffle, same-multiset permutation, genuine self-loop replacement,
context-zero replay and valid 4/8/12/16 early readouts. Self-loop replacement examines at most 8 eligible tokens/window
and reports eligible/evaluated counts; its full-window reexecution cost is not a causal per-token compute estimate.
All scientific decisions remain false. No fourth training arm, STOP, J1, AMP or longer budget is enabled.
