# R3m-E — Capability Differentiation & Reorganization Pilot

Status: `PROPOSED_EXPLORATORY_ENGINEERING_READY`; formal confirmation is not authorized.

## Questions

1. Does Story plus Arithmetic training produce causally measurable Cell functional differences?
2. Does one fixed-time Connection reorganization improve Arithmetic while retaining Story?
3. Does any benefit persist through the final 200-step consolidation phase?

R3m-E has an independent identity and disclosures. It does not register or release original R3m,
change its 512 Main candidates, or access Confirmation. Its exploratory conclusion is limited to
the declared R3h model lineage and development data; global historical isolation is not claimed.

## Frozen matrix and budget

Seeds `151001301`, `151001302`, `151001303`; arms `E0-STORY-STABLE`, `E1-MIX-STABLE`,
`E2-MIX-REORG`, `E3-MIX-SHAM`. Every job starts from its seed's hash-verified R3h B-H Step 300
state and continues 600 updates to global Step 900. Each update consumes 512 tokens. Logical
budget: 7,200 updates / 3,686,400 tokens. Shadow and proposal work is separately recorded.

All arms retain six Cells, eight sequential hard Cell calls per token, the original 4096-vocabulary
Tokenizer, and Cell AdamW state. The Cell optimizer is continued; no Cell, Connection, optimizer,
or model component is reinitialized. Schedule begins at the restored Step 300 learning rate and
epsilon and linearly reaches 20% of that learning rate and epsilon 0.02 over the 600 updates;
there is no new warmup. The inherited R3h/R3i configuration and schedule identities are checked.

E0 is Story-only. E1/E2/E3 use the same fixed mixed task schedule: 90/10, 70/30, then 50/50
Story/Arithmetic over Phase II intervals 0–100, 100–200, 200–400; 50/50 continues through
Phase III. Story and Arithmetic each receive exactly 512 tokens per selected update. Each Seed
starts each arm from the same checkpoint RNG state, so E1/E2/E3 have the same task schedule,
Arithmetic fact pool, and independent data RNG stream. Proposal sees only already-observed
training batches; development labels/data are excluded from its input.

E2 makes exactly one proposal at relative update 100 (global Step 400), with existing Q-first,
22-scale Trust Region and 10-update paired Shadow acceptance. Original KL/action/path thresholds
and acceptance rule are reused without relaxation. A rejected proposal remains a real rejection.
E3 performs the same Shadow schedule as a ghost/sham and retains the original Connection.
No additional proposals occur. Shadow updates occupy the corresponding 10-step main interval.

## Data

Story training reuses the R3h TinyStories Train asset. Story evaluation reuses the original R3h
Development asset and is labelled `REUSED_HISTORICAL_DEVELOPMENT`. Arithmetic Train/Dev is
generated from unordered two-digit addition facts `10 <= a <= b <= 99`; a frozen SHA-256 split
assigns canonical facts (not templates) to Train or Dev. Template variants remain within their
fact's split. No arithmetic fact is shared across roles. The original R3m Main/Trigger/Pilot
candidate data and all Confirmation content are forbidden.

## Measurements and barriers

Every arm is evaluated at global Steps 300, 400, 410, 500, 700, 900. Report Story NLL/PPL,
Arithmetic answer-token NLL, full-sequence NLL, autoregressive exact match, and carry strata.
Also report task-specific Cell exposure, train-only signed-gradient cosine/conflict, paths, legal
Cell replacement interventions, and E1/E2 OO/ON/NO/NN counterfactual grids at Steps 700 and 900.
Compute for main updates, shadow updates, proposals, probes, and evaluation is separate.

E2–E1 is the primary paired comparison; E0 is not an equal-data Arithmetic baseline. Pilot
thresholds in the frozen JSON are exploratory only. Incomplete seeds/arms remain inconclusive.
No best-seed/best-checkpoint selection is allowed. Results can be described only as exploratory,
never as formal confirmation.

## Gates

- Gate 0 freezes identity, arithmetic generator/split, schedule, matrix, budgets, proposal,
  acceptance, barriers, and exploratory decisions.
- Gate 1 uses synthetic CPU fixtures for arithmetic fact isolation, token masking, schedule,
  task budget, legal routing budget, rollback and report barriers.
- Gate 2 is an opt-in one-update real-checkpoint engineering smoke; it must not advance the
  production cursor or count as a study update.
- Gate 3 full exploratory training stays disabled until engineering gates pass and compute is
  explicitly authorized. The original R3m scientific gate remains blocked.
