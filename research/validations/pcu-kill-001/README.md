# PCU-KILL-001

This validation is the smallest formal-ready test of independently composable
Cell forks in a pretrained MoE. It cellularizes only the final actual Granite
MoE block, preserves the original parent-expert router, trains A and B from a
common frozen foundation, and merges only registry/file deltas.

The frozen foundation is `ibm-granite/granite-3.1-1b-a400m-base`. The verified
Granite-MoE implementation stores expert projections as
`gate_up_proj[E, 2I, H]` and `down_proj[E, H, I]`, and explicitly splits the
fused projection as `gate, up`. The implementation records this fact in the
model manifest and rejects any other layout rather than guessing.

## Safe workflow

The engineering backend is explicit:

```bash
python scripts/research/run_pcu_kill_001.py \
  --phase engineering --seed 26090501 --backend granite --device cuda
```

The `--backend toy` option is only a low-cost infrastructure test. Its output
always contains `scientific_evidence=false` and must never be used as formal
evidence.

After the real engineering run resolves the model revision, model hashes,
target architecture, K, optimizer, and thresholds, freeze the protocol:

```bash
python scripts/research/freeze_pcu_kill_001.py \
  --branch codex/pcu-composability-kill-001 \
  --model-manifest <engineering MODEL_MANIFEST.json> \
  --engineering-summary <engineering summary.json>
```

Then commit the frozen protocol and run only the non-consuming preflight:

```bash
python scripts/research/run_pcu_kill_001.py \
  --phase formal --preflight-only
```

Formal seeds `26090511`, `26090512`, and `26090513` are reserved in
`research/formal_seed_registry.json`. This implementation task does not run
them. There is no reset operation after a formal seed is touched.

## Artifact and merge contracts

Engineering/formal run directories use
`artifacts/research/pcu-kill-001/<phase>/<seed>/`. The required JSON schemas
are implemented by the `pcu_kill_001` package: dataset audit, model identity,
equivalence, cache equivalence, gradient geometry, Cell registry, branch
manifest, merge manifest, metrics, decision, and provenance.

Branch workers store fork-minus-parent tensors. A merge checks foundation and
protocol hashes and performs a deterministic registry union. It never averages
weights. Overlapping parent Cells remain independent A/B fork records, and
rollback removes only the requested branch.

The machine decision schema is
`minicells.pcu-kill-001.decision.v1`. Engineering output is diagnostic only;
the formal result wording must respect the decision state and must not claim
that PCU or CLM has been proven.
