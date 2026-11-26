# MiniCells Native MoM SDK 0.1.0 Developer Preview candidate

Python SDK for managing independent neural states on a frozen MoM 1.3B
backbone. The supported preview profile uses one visible NVIDIA Tesla T4,
FP16, and Fused Recurrent calls of at most 64 tokens. All Cells in a
runtime share one set of weights. Python 3.11 or newer and CUDA PyTorch
are required.

Build the wheel from the candidate checkout.

```bash
python -m pip wheel --no-deps ./apps/mom-native-cells -w dist
python -m pip install dist/minicells_native_sdk-0.1.0-py3-none-any.whl
```

Run [the minimal example](examples/preview.py) with a single visible T4.

```python
from minicells_native import MiniCells

with MiniCells.open_preview("./minicells_data") as mc:
    cell = mc.cells.create("research")
    mc.cells.write(cell.id, "The internal code for Project Ardent is LIME-173.")
    mc.cells.append(cell.id, "The project is active.")
    result = mc.query("What is the project code?", cells=[cell.id])
    print(result.text)
    mc.cells.export(cell.id, "research.cell")
```

Omitting `cells` or passing `[]` uses an empty working cache. Pass one Cell
ID to read its working copy. `cells="auto"` explicitly enables the
experimental selector. The preview can maintain several independent
Cells and selects at most one for generation. The existing
`MiniCells.open(...)` remains a research interface.

Append continues from persisted recurrence and starts a fresh short
convolution stream. To correct existing knowledge, append an explicit
correction, or create a replacement Cell and rebuild it from the desired
source records. Deleting a Cell deletes the entire bundle. A recurrent
state does not expose editable individual records.

A Cell bundle contains its schema, identity and revisions, cumulative
`tokens_written` offset, and checksummed recurrent states. Import
rejects incompatible model, tokenizer, precision, execution-profile and
schema identities, and rejects existing IDs. Use Fork to make an
independent copy with a new ID. Query uses copied state and does not
commit temporary writes.

See the [M0 audit](../../docs/python-sdk/M0-001-AUDIT.md) and the
[M0/M1 acceptance protocol](../../research/projects/continual-learning/experiments/minicells-next-stage/PROTOCOL.md).
Memory quality, automatic retrieval quality, and fusion gains remain
research questions. The package is a release candidate pending the
exact-commit T4 and full-suite receipts.
