# R3 baseline and implementation audit

Base source: 4fcb95490c2ca68affbab6881135f2e85e5bea87 (current R2 research branch).
Remote main inspected: 0cff2c2894a1e6e5cd01ca66b7ff9796857fba80. No newer R2 engine was found.
R2 result ref was fetched read only; aggregate report and Q calibration hashes and source commit
are recorded in BASELINE_EVIDENCE.json. Both 500-step arms completed, but the report does not
establish positive Q policy value or validated specialization. Published metadata contain no checkpoints;
no historical weights are assumed downloadable or reused for R3 initialization.

Read-only reuse: R2 action registry, cost scorer, causal input Memory/Cells through PD001/006,
relative-Q/Rank mathematics, numerical/optimizer transaction helpers, immutable full-suffix Oracle,
accepted-state witness, accounting and authenticated result publisher infrastructure.
Version-sensitive adapters are isolated in clm001_r3: event-observing hard rollout, competition,
three-arm trainer, lifecycle/checkpoint identity, expanded calibration and orchestration/reporting.
No import monkeypatch or edit changes R1/R2 behavior.

Default A is implemented, not ST-forward B: backward hooks see the actual post-operation tensor.
Only detached inputs, post-gradient and candidate deltas enter the extra scorer objective. The formal
forward uses no mixture or candidate output. Candidate-only private training creates observation leaves
when all continuous parameters are frozen, so Connection probation still has a real downstream CE gradient.
The Q and LCG vectors are measured at one pre-update parameter snapshot; groups clip independently.
Zero-credit QS is computational control and preserves the Q parameter trajectory in deterministic tests.

The source protocol's development=false is preserved as the initial state. The user's established one-launch
workflow is implemented by automatic verified phase transitions, not manual authorization fields:
real gradient/alignment/resource gates pass → frozen beta/new runtime hash → development. Confirmation
and additional registered seeds are not automatically opened.

Known limits: local credit is biased; subset inclusion does not make subset-softmax an unbiased full-policy
estimator. Beta scale is unknown before GPU Pilot. Parity groups have limited scientific power. MC values
can be unidentifiable. Frozen-state and live-state trajectories do not establish causal self-reinforcement.
Default matched-shuffle training interventions and multi-seed execution remain explicitly NOT_TESTED.
GEMM counters are isolated Pilot/profile lower bounds, not a fake full-kernel training FLOP total.
Resource projections are measurements plus a frozen safety factor, not runtime guarantees or hard kills.
