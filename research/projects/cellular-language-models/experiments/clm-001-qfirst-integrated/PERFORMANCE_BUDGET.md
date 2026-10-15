# Performance budget — GPU unmeasured

Candidate live budget: 500 optimizer steps * effective batch 8 * sequence 64 = 256,000 live next-token targets per arm. Separate 50-step smoke is 25,600 targets. Path budget B=16, minimum one non-STOP action; no forced wall-clock termination or timeout kills training.

Per step: at most two captured token states, up to four candidate actions each, two suffix draws. A conservative upper bound is 2*4*2*16=256 counterfactual operator calls per optimizer step, versus at most 8*64*16=8,192 live calls. This is an **operator-count bound, not a 3.125% runtime claim**: single-row suffix work, small kernels, serialization, snapshot copies and GPU synchronization can dominate. Context reads cost more than Cell calls in hardware despite equal provisional logical cost. Lambda and any revised cost scale remain unfrozen.

Live training does not enable TorchDispatch, dense path-capture diagnostics, repeated dev inference or full gradient statistics every step. It retains bounded Oracle provenance, finite-gradient checks, one immutable snapshot and a rollback backup for the coordinated update. Those transfers/copies still need real GPU profiling. Per-step timers are host wall/enqueue measurements; only isolated Pilot synchronized totals are GPU wall measurements.

Offline profile reuses PD's executed mm/addmm/bmm counter, including tied output and Context matmuls; it is a supported-GEMM lower bound, excludes nonlinear/normalization/dispatch costs, and never runs in the live hot loop. Exact resident inventory, actual leaf-module parameter work, repeated weight work, window-union touched weights and memory/latency are distinct fields. A window union is not advertised as per-token active parameters.

Automatic preflight before development: fresh full-size frozen-model Q-calibration repeatability; three isolated real-Train private timing updates per arm; one bounded private Shadow probation, paired Admission timing, checkpoint timing and offline profile. Report rollout/backward/Oracle/snapshot/Shadow/Admission/checkpoint separately, plus synchronized total and peak memory. The 500-step projection excludes cloud setup/final analysis and must be reviewed. No GPU timing, language gain or two-hour completion guarantee is currently available.

After automatic functional checks, two T4s are used one per arm. Offline diagnostics and publication run after both training workers, preventing same-card profiler interference. Real suffix and Shadow token work are reported separately from live token budget.
