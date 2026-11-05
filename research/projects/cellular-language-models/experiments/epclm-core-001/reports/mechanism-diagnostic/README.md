# C1 mechanism diagnostic implementation record

Status: IMPLEMENTATION_READY; real private asset recovery and GPU D0–D3 NOT_RUN.

This is post-training, exploratory development analysis of the frozen C1 trained
at `398a191dbb32e383c86c89b9c981777c7cd6c77f`, not preregistered confirmation.
The supplied [specification](../../C1-MECHANISM-DIAGNOSTIC.md) is preserved verbatim;
its PROPOSED label describes that source document. Executable intervention choices,
asset identities, tolerances and budgets live in [diagnostic-protocol.json](../../diagnostic-protocol.json).
The historical training protocol, model implementation and result branches are unchanged.

## Execution

Import [c1-mechanism-diagnostic.ipynb](../../c1-mechanism-diagnostic.ipynb) into the
existing private `bingjiangli/minicells` Notebook. Enable GPU, Internet and existing
`KAGGLE_API_TOKEN` / `GITHUB_TOKEN` secrets, then Save & Run All. No Add Input,
Output mounting, manual source SHA or new dataset creation is needed.

The diagnostic CLI exits before the training launcher. It never creates an optimizer,
performs backward, invokes training or falls back to replacement weights. The frozen
loader pins the training source/protocol/data, outer checkpoint hash, checkpoint
payload checksum, model configuration, completed tokens/updates and weight hash.
Only the C1 checkpoint, manifest, tokenizer and development documents are recovered;
no F0 checkpoint or training documents are needed. Optimizer state inside the original
trusted checkpoint is parsed by the existing loader but never installed or used.
A separate `DiagnosticCore` inference subclass has exactly the original parameter
names/shapes and normal forward path; the only new path omits left/right message terms.

Recovery stops at the first verified exact asset bundle, scans at most 25 saved
versions, and has a 600-second interruptible network budget. Asset recovery failures
stop with BLOCKED_MODEL_ASSET_UNAVAILABLE; hash/configuration integrity failures stop
with FAILED_BASELINE_INTEGRITY. The saved-version PENDING_SAVE metadata is not treated
as proof that assets exist. D0 must independently reproduce NLL 5.814626392302277
within absolute 1e-4, without relaxation, before D1–D3.

## Fixed choices and interpretation

- First 16 original development documents, unchanged ordering and BOS/EOS IDs.
- D0 repeats the complete baseline, checks pure read/repeated input, and checks exact
  segmented/snapshot continuation on up to 32 events. It verifies frozen weights.
- D1 RESET_AT_PROBE resets once before the midpoint event of each document; reports
  both full-document and post-probe paired differences. ENTRY_ONLY clears non-entry
  cells before every event. RESET_EVERY_TOKEN is explicitly an extreme intervention.
- D1 controlled histories use up to 8 accepted pairs from four names and balanced
  color alternatives. Both directions are scored, including both candidate targets
  from cloned states. Prefix event counts and target token lengths must match;
  concatenated versus segmented BPE boundaries must agree. Exclusions and per-pair
  margins/NLLs stay in private Output. Fewer than 8 accepted pairs is INCONCLUSIVE.
- D2 NEIGHBORS_REPLACED_BY_SELF preserves all three learned projections;
  CROSS_CELL_REMOVED uses center only. Single-event interventions start from cached
  normal pre-event states. Cell probes use IDs 0,1,4,8 and ring distances.
- D3 full trajectories use K=2,4,8. At normal pre-event probes, the event is processed
  once at K=4, then cloned states evolve an additional 0,4,8 steps with no new event.
  Those conditions therefore have total depths 4,8,12. State change alone is not gain.
- Probe positions are fixed at floor(25%,50%,75%) of each document's prediction count;
  only these small states are retained, not complete state trajectories.
- Natural-document uncertainty uses 1000 paired document bootstrap draws, token-weighted
  aggregate differences, unweighted document mean differences, counts and direction
  fractions. These intervals do not measure between-training-seed variance.
- Exploratory labels use paired interval signs. Semantic-history signal additionally
  requires at least 8 pairs, at least 75% bidirectional pair success, and better normal
  correct-target NLL than reset. These descriptive rules were selected after training
  but before diagnostic results, and do not alter H1–H5 NOT_EVALUATED.
- No automatic 64-document extension or follow-up training. Inference has its own
  2400-second budget, all execution 5400 seconds, publication reserve 600 seconds,
  safety 1200 seconds and outer Notebook hard limit 7200 seconds.

Each completed condition is saved atomically and bundled with the existing immutable
recovery manifest. Exact diagnostic-source/protocol progress can be recovered from
saved Output separately from the old training source. COMPLETE results retry only
publication; partial runs recheck D0, then skip finished conditions. Terminal failures
are retained and republished, without automatically rerunning the experiment.
Model assets remain in a separate private cache and are not copied into diagnostic
result recovery bundles or Git. Publishing uses the existing authenticated publisher,
a result-only commit descending from main, a separate result branch, and remote SHA
verification. Public reports omit per-document, controlled-pair and per-event details.
Private Output retains the full decision and event details plus a `reports/` directory
with the six summary files and figure. Network/recovery and synchronized inference
metric-collection timings are reported separately; timings are not bare-kernel benchmarks.

## Validation and remaining gate

One small synthetic CPU contract test covers strict frozen loading and corruption
rejection, baseline equivalence, snapshot/segmentation, state isolation, communication
semantics, event injection counts, no optimizer/backward, early-stop saved-output
recovery, completed-report recovery and CLI routing. No real training, GPU run,
new dataset download or local performance benchmark was performed.

The private saved checkpoint has not been accessed in this local implementation task.
Actual recovery, D0 tolerance, numerical stability under all interventions, inference
cost and scientific findings remain pending the user's Kaggle execution.
