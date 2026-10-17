# R3b diagnostic optimization — resource review

## What was actually measured

Previous successful Pilot: user log + fetched public GitHub index, source
67b3dd47ab88, approximately514.7s wall. Calibration snapshot times were162.534,
156.384,159.897s. Previous total26060.03s (7.24h) was a conservative extrapolation,
not a full production run. The previous report does not contain per-function
Quality/Functional/Action timing: NOT_MEASURED. We did not repeat that Pilot.

New work in this release is synthetic CPU/FakeHub verification. CPU task outputs
measure per-policy Quality, Functional, Router, Action and total elapsed time;
these are not Tesla T4 production estimates. No real GPU benchmark or training
was launched. New two-hour verdict: **BUDGET_NOT_VERIFIED / NO-GO for formal run**.

## Critical path changed

General four-policy Development evaluation, Router/STOP/CTX counterfactuals and
Functional fixed suffix are no longer called by default from `runner.segment`.
Checkpoint collection and Journal-only telemetry remain. No MC/probe/sample count
was reduced. All milestone and original event saves/ACK remain; required Admission
quality judgments and Shadow private training still run synchronously.

Training lower planning reference: two sequential arms on GPU0 at about3s×1000
updates =3000s (~50min), excluding all bank capture, semantic events/private
updates, serialization, shared HF waits/readback and startup. This is not a bound
or completion guarantee. Upper bound is NOT_VERIFIED, especially Step400 Admission.
No timeout, kill, hard wall-clock budget or data-budget reduction was introduced.

Offline initial plan has102 task units for a complete three-arm matrix:39 Quality
policy-set tasks plus42 Functional role tasks plus21 Action tasks. Task count is
not GPU-hours. Timing budget remains NOT_VERIFIED until separate Calibration and
Development tasks, production event costs and HF production history are measured.
max-tasks stops only between committed analysis tasks; sessions resume from the
private index without repeating verified COMPLETE tasks.

## Next permitted validation

Read the already accepted Pilot exact HF revision. A separately approved micro
benchmark may run ONE QC-R3b calibration Action task at Step3/revision0 and a
separate small Quality task if approved. Neither changes parameters or repeats
full Pilot. A later 1–3-update delta test requires an explicitly named engineering
run and approval. Full production Admission/event costs must also be accounted
for before claiming <=7200s. Total study time (training + offline work) is reported
separately; moving work to another session does not reduce total GPU consumption.

## Known limits / release

Formal500-step authorization stays false. Minimal real GPU delta/performance
Gate4, frozen execution plan approval and Gate5 release are outstanding. CLI auto
now performs preflight then reports HOLD; it cannot silently repeat full Pilot.
Analysis Notebook defaults to listing old pinned checkpoints and starts no GPU
work unless the bounded task action is explicitly selected by the operator.
