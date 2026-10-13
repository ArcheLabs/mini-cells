# HybridCLM examples

These examples use only the public API. They require an immutable Granite Hub
revision and an artifact directory containing a compatible Cell mutation.

```bash
pip install "mini-cells[hybrid]"
python research/projects/cellular-language-models/examples/hybrid_clm/inspect_granite.py --revision <commit>
python research/projects/cellular-language-models/examples/hybrid_clm/attach_mutation.py mutation/ --revision <commit>
python research/projects/cellular-language-models/examples/hybrid_clm/alpha_sweep.py mutation/ --revision <commit>
```

The examples do not run PCU experiment code or formal seeds. Alpha values are
expression settings; they are not dependency-withdrawal claims.
