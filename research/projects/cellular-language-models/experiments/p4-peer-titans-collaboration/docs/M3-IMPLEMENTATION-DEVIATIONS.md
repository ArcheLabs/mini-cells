# M3 preparation deviations and open decisions

- Specification §17 governs this delivery. The CPU probes inspect frozen state and exercise existing forwards. Composition implementation, fitting, Smoke and Screening remain pending review.
- Existing M2 read helper is single-stream. A temporary batch view passed parity and immutability checks on restored A/B. Implement it as a thin adapter rather than adding shared writable state.
- Mode B is a near-match because hidden width is integral. H=129 is 248 dynamic bytes smaller than two H=64 Cells. Report architecture, compute and storage separately.
- E6 is a hybrid Mamba-1/SWA/State Summary checkpoint. The audited E2 pure Mamba-1 source can be reused. Selection and hashing of trained standalone weights remain open. An initializer-only Smoke qualifies engineering checks and needs explicit labeling.
- S2 requires separate information paths and measured execution evidence for concurrency. S3 is conditioned on SSM hidden state. Fair head fitting, matched compute and absent-branch controls still require validation.
- Cross-domain queries have explicit slot/domain context. Any future same-Key collision without domain context requires set-valued scoring.
- Prior M1.1 results include a three-mapping CPU smoke and an eight-exposure T4 timing smoke. The full screening matrix is pending, so use their capacity and interference observations as development evidence.
