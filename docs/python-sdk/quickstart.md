# Quickstart

```python
from minicells import MiniCells

with MiniCells.open(storage="./memory") as mc:
    a = mc.cells.create("founder")
    b = mc.cells.create("prize")
    mc.cells.write(a.id, "The Green Belt Movement was founded by Wangari Maathai.")
    mc.cells.write(b.id, "Wangari Maathai won the Nobel Peace Prize in 2004.")
    result = mc.query("Who founded the Green Belt Movement?", max_new_tokens=16)
    print(result.text, result.selected_cell_ids, result.cell_scores)
    mc.cells.export_collection("project.cells.zip")

with MiniCells.open(storage="./another-memory") as mc:
    mc.cells.import_collection("project.cells.zip")
    print(mc.query("Who founded the Green Belt Movement?", cells=[a.id]).text)
```

Keep the two storage directories separate. Imported Cells keep their IDs.
Explicitly pass `cells=[]` for ordinary model inference. The frozen backbone is
a base language model. Inspect generation text and scores for your workload.
See [examples](../../apps/mom-native-cells/examples) for separate workflows.
