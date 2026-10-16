# CLM-001-R3b — route stability and Cell specialization

**Engineering implementation; formal500-step research remains HOLD.**
Read [SPEC](SPEC.md), [analysis definitions](ANALYSIS_PLAN.md),
[reuse audit](DESIGN_AUDIT.md), [engineering acceptance](reports/engineering-acceptance.json)
and [proposed config](config/proposed.json). Historical R3 is unchanged.

## One Kaggle entry

Import `notebook.ipynb` into the original bingjiangli/minicells notebook. Keep the
existing HF_TOKEN/GITHUB_TOKEN Secrets and private repo
`libingjiang/minicells-research-checkpoints`. No Add Input, local checkpoint download,
manual hash/seed fields, repeated pytest suite or synthetic fixture masquerading as
real training. Source SHA is pinned in the Notebook.

Default ACTION=auto performs static CPU/source/identity checks, real storage and
GitHub permission preflight, reuses/prepares the original assets, then a **minimal
private two-T4 Pilot**. Historical cross-session acceptance is informational:
the user has manually checked it and requested removal of Gate C because the old
versions were not retained. No A/B/C replay or version URL is required. This is
recorded as USER_VERIFIED, not an automated test PASS. Formal500-step authorization
and the actual Pilot/resource review remain required.

Pilot hardware:GPU T4×2. Its same host publisher measures real checkpoint sizes,
three private Train updates per arm, snapshot cost and a restored trajectory. It
returns PENDING_REVIEW and preserves both HF and GitHub evidence. No GPU termination
limit. Review real timings and fix the upload/snapshot budget before registration.

## Formal execution (only after Gate A–E and explicit approval)

- `train`: original three arms, seed151001301, fixed500-step budget; segment200.
- `resume`: exact HF/GitHub checkpoint index or same source/run lookup; next350/500.
  Never regenerate the assets, beta, run identity or frozen probes on resume.
- `evaluate`: choose arm, step and optional Event Revision. At400 revision0 is
  pre-event; omitted revision means latest committed event. Independent Development
  evaluation, no optimizer update, original bundle unchanged.
- `fork`: explicit new experiment+run IDs and inherit/reset optimizer policy.
  Parent model/probes and Event cursor provenance preserved; no training occurs.
  A different algorithm/shape needs its own reviewed runner, not a disguised Resume.
- `publish-only`: recover committed HF evidence, retry GitHub; never retrain.

The launcher exposes corresponding CLI flags through `python r3b/run.py --help`.
Do not change training.steps for segment control. Training/evaluation can resume
from different arm progress. The host validates immutable handoffs and ACKs only
verified HF commits. HF outage returns STORAGE_BLOCKED; GitHub failures have a
separate retry status. All research results remain single-seed exploratory;
CAUSAL_FEEDBACK_NOT_TESTED and MULTISEED_NOT_TESTED cannot auto-upgrade.

## Saved evidence

At every declared boundary, HF stores the original full state, identity, frozen
Calibration/Development probes, all Event revisions, diagnostics and assets.
GitHub receives only the declared aggregate JSONs, checkpoint-index and storage
status. Per-story scores/raw probes/Oracle labels/weights/optimizer/RNG stay private.
Plots are generated from recorded aggregates and included in the HF bundle; no
new model evaluations are run for plotting. Positive or negative outcomes are saved.

## Current release limitations

- No real GPU Pilot has run here. Do not infer production bandwidth from the
  tiny CPU engineering run. Calibration snapshot extrapolation is explicitly not a
  measured full Development/Event budget. Formal release requires budget review.
- Actual beta, real asset manifest, accepted Pilot and formal authorization remain
  unfrozen. `experiment.lock.json` is explicitly PROPOSED_NOT_REGISTERED.
- The three production models each allocate1,075,492 parameters (including the
  frozen critic and reserve);1,070,819 have requires_grad=True. Reserve parameters
  do not imply initial accepted-path usage. See reports/parameters.json.
