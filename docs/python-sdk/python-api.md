# Python API

| API | Result |
| --- | --- |
| `MiniCells.open(model=..., storage=..., device="cuda", dtype="auto", max_cells=8, write_chunk_size="auto", selection_threshold=None, selection_top_k=2)` | Runtime and exclusive registry |
| `mc.close()` or context manager exit | Releases registry lock and model reference |
| `mc.cells.create(name)`, `get(id)`, `list()` | Immutable Cell record or list |
| `mc.cells.write(id, text, chunk_size="auto", progress=None, cancelled=None, resume=True)` | New revision or `None` on cancellation |
| `mc.cells.append(...)` | Append alias for write |
| `mc.cells.fork(id, name=None)`, `revisions(id)`, `delete(id)` | Fork, revision metadata, or deletion |
| `mc.cells.export(id, path)`, `export_collection(path, cell_ids=None)` | Destination path |
| `mc.cells.import_cell(path)`, `import_collection(path)` | Imported Cell record or list |
| `mc.query(prompt, cells="auto", max_new_tokens=128, temperature=0)` | QueryResult |

`Cell` exposes `id`, `name`, `revision`, timestamps, written token count and
`parent_cell_id`. Public errors are available from `minicells.exceptions`.
Example error types include `CellNotFoundError`, `CellAlreadyExistsError`,
`IncompatibleModelError`, `StorageBusyError`, and `UnsupportedChunkSizeError`.
Operations on a closed runtime raise `RuntimeClosedError`.
