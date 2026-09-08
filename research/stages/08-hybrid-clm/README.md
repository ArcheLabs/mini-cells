# Stage 08 — HybridCLM

Stage 08 is the current software-facing bridge in the MiniCells research
line. The strategic path is:

```text
MoE → Hybrid CLM → progressive dependency withdrawal → Standalone CLM
```

HybridCLM keeps a pretrained MoE as the mature computational substrate while
attaching independently manageable, reversible Cell mutations. The public
toolkit supports explicit placement, attachment, alpha scaling, validation,
serialization, and rollback for the tested Granite MoE backend.

The primary scientific objective remains progressive withdrawal of dependency
on the original parent computation. A secondary product hypothesis is that
HybridCLM could become a competitive post-training/model-evolution mechanism.
That opportunity is conditional on a controlled comparison and is not claimed
by this release.

Current engineering evidence is recorded by the PCU Hybrid Reattachment 001
line. It is not a formal HybridCLM validation and does not establish standalone
conversion, generic MoE transfer, continual-learning success, or superiority to
PEFT methods.

See [the evidence boundary](EVIDENCE_BOUNDARY.md), [roadmap](ROADMAP.md), and
the [public HybridCLM documentation](../../../docs/hybrid-clm/overview.md).
