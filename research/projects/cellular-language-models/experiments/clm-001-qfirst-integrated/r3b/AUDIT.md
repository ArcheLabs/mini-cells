# Diagnostic execution audit — 2026-10-17

## Baseline and scope

Remote main: `bd051e6854d56e21655e304a6763eee0b369b59c` (PR263 merge).
This branch carries the later validated CPU-publisher/GPU-RNG repair and the
user's explicit removal of missing historical version URL Gate C. That decision
is retained; it is not upgraded to an automated test PASS. No historical R3 files
or results are edited. The accepted old Pilot keeps source67b3dd47ab88 and its
original HF revision; it is not re-labelled as a new-source Pilot.

The public result branch was read at its actual commit, captured in
`baseline-evidence.json`. Exact final HF reference is recorded, but no private HF
credentials or GPU measurements were used from this host.

## Classification of every runner boundary

| Call | Category | Scheduling |
|---|---|---|
| `Trainer.update` / Oracle / LCG / clipping | TRAINING_SEMANTIC | unchanged |
| `events.process`: propose-cell, propose-connection/path, pre-admission, admit-*, promote | TRAINING_SEMANTIC | original order, local save and remote ACK after each transaction |
| original Admission Development assessments, candidate private updates | TRAINING_SEMANTIC | synchronous, never substituted or deferred |
| Step0 Calibration and independent Development Bank capture | required fact capture | once, persisted bytes; restore never regenerates |
| new Journal/checkpoint/registry telemetry | TRAINING_TELEMETRY | no extra model forward/backward |
| four-policy evaluation, routing, STOP/CTX counterfactual, fixed_suffix | POSTHOC_ANALYSIS | immutable historical checkpoint tasks |
| bootstrap / aggregate / plot | POSTHOC_ANALYSIS | recorded observations only |

`diagnostic_schedule` never calls `events.process` and cannot suppress an Event.
Synthetic CPU reference scheduling is explicitly restricted to CPU synthetic runs;
it exists to compare enabled/disabled/deferred measurements with the exact same
identity, including real original scaled proposal/admission/promotion operations.

## Identity and storage

The execution plan is part of a new runtime identity. Old runs are never silently
migrated. Deferred references retain all original checkpoints. Tasks bind training
source/run/runtime/assets/tokenizer/protocol, checkpoint hash, exact event revision,
analysis source/plan, role/windows/seed/policies and original frozen bank hash.
Initial hidden states are labelled FROZEN_INITIAL_STATE, not live recollections.
HF task results use `analyses/.../private/<task>/result.json`, followed by CAS index
commit and pinned readback. Training runs/latest is never written by this store.
CAS reentry verifies committed result hashes before skipping execution. Changed
result bytes at one logical task key are a conflict, not last-writer-wins.

The analysis launcher accepts only the existing private HF repo, exact verified
bundle references and the existing trusted GitHub index loader. No Add Input or
local download handoff. Deserialization occurs only after trusted HF fetch + full
byte verification. Hashes alone are not a trust boundary for arbitrary pickle.
