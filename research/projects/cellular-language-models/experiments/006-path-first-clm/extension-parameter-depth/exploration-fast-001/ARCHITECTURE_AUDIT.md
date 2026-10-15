# Architecture and compatibility audit

Read-only baseline: `91b1340`, latest PD-001 performance-repair branch at task start. Remote main was checked;
its merge of PR245 precedes the latest direct-development launcher changes. FAST is appended to this newer branch
so the validated light executor and current authorization behavior are available without editing historical code.

| Existing component | FAST use | Difference |
|---|---|---|
| `pd001.models.PDCLM.forward/prefill/decode_step` | inherited verbatim | smaller model configs, fixed 8/16 actions |
| `pd001.router.Router.forward` | subclass reuses scoring/critic math | removes STOP weights and maps START row; constant disabled STOP slot for inherited executor |
| `pd001.losses.joint_loss/sample_paths` | used verbatim | J0 only, cost zero, no STOP |
| `pd001.engine.step` | called directly | FAST thin loop controls evaluation/checkpoint scheduling, independent namespace |
| `pd001.performance` | engine version/config contract audited | default light, diagnostics every100; no resident full accounting |
| `exp006.baseline.MoELM` | used verbatim | width128,6layers,4experts,Top2,hidden176 |
| `pd001.checkpoint.save/restore` | called directly | identity `006-PD1-FAST-001`, extra FAST history/curves |
| `pd001.evaluation.intervention_pair` | reused for edge-zero/order/context-zero | fixed-router shuffle and conditional self-loop subset need FAST adapters |
| `pd001.accounting.Counter/TrainingGemmCounter` | offline only | reports lower-bound FLOPs, touched weights and repeated work distinctly |
| `pd001.analysis.paired_interval` | called directly | 1000 story-cluster bootstrap draws, no superiority/confirmation test |
| `pd001.persistence.export_bundle/import_bundle` | reused transport | FAST validates experiment identity; no network or Git weight upload |
| `pd001.orchestration` | interface audited | new small dynamic scheduler makes S16/S8 first and assigns M0 to first free device |

The original MoE internally captures its native Top-2 router decisions via transient hooks; FAST does not add
per-layer diagnostic hooks to normal training. The light CLM step has no resident Dispatch/Counter or full path clones.
Periodic gradient/state capture uses the existing engine and is timed separately from ordinary steps in benchmark.
Read-only input memory differs structurally from six-layer MoE attention; this confound is stated, not hidden.

S8/S16 alpha remains .125 and step table capacity16 is shared; all names/shapes/initial weights match. No learned
unused STOP row is retained to satisfy parameter targets. Budget is the sole execution difference. Initialization
uses inherited name-seeded `exp006.model.initialize`. A single AdamW sees all parameters every joint optimizer step.
Action RNG is independent of NumPy data order and private sampled eval generators. Router critic detaches state and
step features; exact enumeration includes Cell parameters changing later routing probabilities.

Historical protection: only this new subtree, `tests/research/path_first_fast001`, and `.github/workflows/fast001-cpu.yml`
may differ from baseline. No old metrics, configurations, notebooks, checkpoints or sealed confirmation arrays change.
Data governance reads configuration/seed metadata only. Actual FAST data hashes await user-reviewed preparation.
