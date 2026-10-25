# D1 independent review closeout 002

Review baseline: `1b840531411f95e8f43bd0bc908d11da7402ffc6`. The independent re-review accepted the original four repairs and requested two remaining audit clarifications. This closeout implements those clarifications and does not constitute GPU authorization.

## Requested GPU versus observed use

The execution manifest now starts with `gpu_requested` derived from the requested device and `gpu_used=false`. A CUDA device request or a device-count check alone does not count as actual use. Actual CUDA model tensors or positive allocator peak records establish observed use. A failure during parent loading is also checked using already initialized allocator statistics. This query neither creates tensors nor initializes or synchronizes CUDA. If observation fails and no prior positive evidence exists, the manifest records null/UNKNOWN. Positive evidence retained from a previous attempt remains valid.

CPU tests cover a rejected GPU request, failure-manifest recording without initializing CUDA, simulated positive allocator evidence, an unavailable observation, and preservation of prior confirmed evidence. These simulated allocator responses are API tests, not GPU execution tests.

## Meaning of the fixed budget

Each arm may retain at most 200 logical updates. This is the fixed scientific trajectory budget already specified by `steps=200`. It does not promise that all failed attempts combined will make at most 200 optimizer calls. An interruption between 25-step checkpoints can repeat work on resume. Such repetitions cannot extend the retained trajectory or change evaluation barriers.

`optimizer_updates_actual` is renamed `logical_optimizer_updates`, and `logical_optimizer_updates_total_max` names the three-arm logical bound. Cumulative optimizer calls and GPU time across failed attempts are null/not measured. No hard runtime termination or cumulative retry cap is introduced. Recovery tests still require exact retained progress, no missing curve entries and zero extra updates when a completed run is reopened.

The frozen config SHA remains `656da576ad7e6908825c9c496a5434155957f591aadcf4fee00ddbd14f67648b`. No facts, labels, model, objective, parameter-freezing rule, optimizer continuation, sampling plan or logical update budget changes. All previous evidence remains under its own original source identity.

## Release status

`RUN_BCD=False` remains the Notebook default. No real checkpoint is trained or evaluated in the CPU fixtures; no GPU scientific run is started. A short review of this closeout and its CPU evidence is the next step before any separate B/C/D authorization. The full R3m-F Pilot remains outside the scope.
