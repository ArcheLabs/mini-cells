# HybridCLM quickstart

Install the optional model dependencies:

```bash
pip install "mini-cells[hybrid]"
```

Use an immutable foundation revision and an explicit placement:

```python
from minicells import CellMutation, CellPlacement, HybridCLM

hybrid = HybridCLM.from_pretrained(
    "ibm-granite/granite-3.1-1b-a400m-base",
    revision="<immutable-hub-commit>",
)
print(hybrid.inspect())

hybrid.cellularize(CellPlacement(layer=7, experts="all"))
mutation = CellMutation.from_pretrained("archelabs-org/granite-3.1-1b-hybrid-cell-l7-k64")
hybrid.attach(mutation)
hybrid.set_alpha(mutation, 0.75)

# call hybrid.generate(...) or hybrid(...)
with hybrid.mutation_enabled(mutation, alpha=0.0):
    baseline_like_output = hybrid(**inputs)
hybrid.detach(mutation)
assert hybrid.verify_restoration().passed
```

The revision, architecture signature, module signature, tensor shapes, and
SHA256 are checked before attach. A mismatch is a hard error by default.
