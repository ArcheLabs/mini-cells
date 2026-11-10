# B0-P2 implementation and pre-execution review

Status: implementation and CPU gates complete; actual T4 benchmark, common-budget selection, P2 training, production archive download/restore and research analysis remain pending. No P2 training updates or Test access occurred locally. This report does not declare the experiment complete or support any scientific hypothesis.

## Source review

Current remote main is ce9f1905b8d9a8badd2dbed111f68e180e71a619. The verified B0 execution source is8bb11b03da52ecf9ae74bca8846eb4ece065a091. Main does not contain the B0 research adapter; this implementation branches from that verified adapter and checks the original main dependency hashes. All changes are isolated under m5-sv/b0-p2/.

E0 already supplies differentiable full causal512-token forward. E6/E7 forward already creates a new Runtime and retains the full Episode training graph. Its local SWA window32 is intrinsic architecture, not a truncated task input. B0's mean masked loss correctly supervises only the8 answer placeholders; no current/future gold answer is fed in. The existing M1 sampler already reshuffles at every epoch boundary and snapshots both order and RNG. No second Transformer, new training framework, optimizer rule, scan kernel or cross-Episode memory path is needed.

The original modulo partition produces disjoint conditional Pair support. Finite-sample marginal imbalance can coexist with that conditional difference; reducing marginal JSD is not a proof that every distributional confound is eliminated. The new split remains an unseen-pair/shared-symbol development test.

E7 adds no trainable parameters beyond E6. Its exclusive-parameter initialization is explicitly NOT_APPLICABLE. The unchanged classes instantiate10,130,688 E0 and10,021,120 E6/E7 parameters. Formal execution checks the complete fresh initialization hashes against the previously validated T4 records; it never loads older training weights.

The known checkpoint portability defect is inherited from RecurrentBaseline._version shadowing PyTorch's metadata version name. P2's private save adapter exports the same model tensors in a plain dictionary, preserving tensor values/aliases and the original atomic envelope, optimizer, RNG and sampler. Historical classes and checkpoint files are untouched. Exact tiny continuation and real checkpoint restoration are tested.

## Data findings

Reconstructed source banks match their original frozen hashes:3746 Training unique pairs,506 Validation unique pairs, zero overlap,4252 combined pairs and full256-Key/256-Value coverage. Candidate Key degree is7–27 and Value degree7–28. Preserving this pool makes Training degree14–15 everywhere impossible. The user explicitly accepted treating that as an ideal and froze the following marginal criteria before GPU execution.

| Marginal | Actual JSD, bit | Accepted maximum |
|---|---:|---:|
| Unique Key |0.0103433631|0.02|
| Unique Value |0.0105794380|0.02|
| Observed Key |0.0049270441|0.01|
| Observed Value |0.0039346556|0.01|

New Training and Validation have512/64 Episodes,4096/512 supervised observations and3746/506 distinct pairs. Both cover every Key and Value. Pair overlap is zero; Episode input duplication is zero. Validation unique degrees are exactly250 symbols of degree2 and6 of degree1 on each side. Training unique Key degrees are6–25 and Value degrees5–26, dictated by the original graph and Validation coverage.

Source-bucket unique counts are3300/446 in Training and446/60 in Validation. Observation counts are3609/487 and451/61. Source-proportion differences are0.0004833 for unique pairs and0.0002441 for observations. All per-symbol frequencies, multiplicities, total variation and JSD are preserved in data/SPLIT-AUDIT.json.

Supplemental observations reduce marginal imbalance while preserving every assigned pair. Training still has350 duplicate observations (8.5449%); Validation has6 (1.1719%). Maximum Training pair multiplicity is7, versus3 previously; Validation maximum is2. This within-bank allocation is an explicit component of the new data distribution, not an unnoticed claim of4096 distinct associations. Exposure and data allocation change jointly, so improvements cannot identify a single cause.

Frozen dataset content identity: d2d01b6829909ecb10ee7045009085c2269d9dee7f20f072e6c8c141c28279ee. Runtime reconstructs and checks the committed bank hashes. Test remains reserved, ungenerated and unaccessed; no new physical sealed Test artifact is claimed.

## Monitoring and integrity

The new private M2 adapter observes actual per-update parameter change/norm, LR, epoch, tokens and GPU allocation; optimizer/loss/gradient/rollback math remains unchanged. Read-only complete-Episode evaluations include32 Training Episodes and64 Validation Episodes at initial/scheduled/final boundaries. Stable log-softmax provides full/type/conditional NLL, clustered intervals and prediction concentration/ranks. Saved probabilities/log probabilities and labels are independently checked.

Summary norms and final state activations/resources are observed without altering outputs. E7 records actual admissions, rejection, occupancy, eviction, EMA/innovation statistics and selected-position overlap with the factual Key/Value tokens. Missing state-update norms, raw retrieval scores, calibrated saturation and causal Store contribution are NOT_AVAILABLE. These are position-level observations, not claims of semantic retrieval or causal memory usefulness.

Complete bundles include actual datasets, source/config/data/init/environment identities, update logs, prediction/state diagnostics, full probability arrays and normalized Model/Optimizer/RNG/Sampler/Ledger checkpoints. Identical latest/final payloads use tar hard links with full per-path hash checks. The independent verifier downloads full assets after publication, verifies every archived file, restores the downloaded final model and repeats both diagnostic banks. Companion metadata indexes preserve the final restore proof and report; a summary index preserves paired Episode-bootstrap EM/NLL differences and resource costs.

## CPU validation

The focused suite passes14 tests, using small models and synthetic CPU optimizer fixtures. Coverage includes deterministic balanced-bank reconstruction and source hashes, coverage/leakage/marginal gates, extreme-logit stable NLL decomposition, common two-epoch sampling, read-only instrumentation, actual optimizer changes and exact restart for all three original model classes, normalized checkpoint export, independent payload restore/probability reproduction and corruption rejection, shared-budget selection or rejection, archive hard-link integrity, a real child-process timeout and protected historical Output namespaces.

No canonical10M optimization, old Overfit rerun, old Pilot continuation or GPU performance claim is part of those tests. Source code and Notebook syntax compile. Full logs and content hashes are in audit/.

## Hardware execution still required

The local environment is Python3.11/Torch2.6 CPU, without CUDA or authenticated Kaggle execution. The Notebook uses the existing T4 environment and GITHUB_TOKEN. GPU execution is neither simulated nor marked passed.

The previous E6 bundle was locally downloaded completely and its archive SHA verified; its recorded fit wall is1623.2706 seconds and training compute throughput204.3089 input tokens/s. Full verified historical timing retrieval for all three arms is enforced automatically before new benchmarking. No missing E0/E7 timing is guessed from the truncated Notebook log.

The final common1024/768/640-update budget is deliberately unresolved until actual independent temporary T4 benchmarks. The7200-second cap includes all three arms, benchmarks, checkpoint and independent restore; it persists across resume. A forecast above6480 even at640 stops execution. CPU data work and external artifact transfer are separate from charged GPU phases. The full session is additionally bounded below3 hours.

Run notebook.ipynb once in the existing Kaggle Notebook to perform the remaining gated execution. Completion requires all expected updates, independent download/restore and remote SHA verification. Source, precision, sequence length, model size, per-arm budgets and scientific standards will not be relaxed on failure.
