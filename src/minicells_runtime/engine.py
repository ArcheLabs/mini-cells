"""Multi-Cell orchestration with revision-checked, functional updates."""
from dataclasses import replace
from threading import RLock
import uuid

import torch


class MultiCellRuntime:
    """The adapter supplies encode(ids, state) and logits(hidden).

    Encode owns its returned session state. Memory uses the existing Titans
    NeuralMemory interface. Registry entries are replaced only at commit.
    """

    def __init__(self, backbone, memory, projection, gate):
        self.backbone = backbone
        self.memory = memory
        self.projection = projection
        self.gate = gate
        self._cells = {}
        self._lock = RLock()

    def _insert(self, name, cell):
        if not name or name in self._cells:
            raise ValueError('Cell name is empty or already exists')
        self._cells[name] = cell

    def create(self, name):
        with self._lock:
            self._insert(name, self.memory.create(1, 'online_adapt').detached())

    def get(self, name):
        with self._lock:
            return self._cells[name].detached()

    def list(self):
        with self._lock:
            return tuple(self._cells)

    def delete(self, name):
        with self._lock:
            del self._cells[name]

    def clone(self, source, target):
        with self._lock:
            cell = self._cells[source].detached()
            self._insert(target, replace(cell, metadata={
                **cell.metadata, 'cell_id': uuid.uuid4().hex, 'parent': None,
            }))

    def fork(self, source, target):
        with self._lock:
            self._insert(target, self.memory.fork(self._cells[source]))

    @torch.no_grad()
    def generate(self, input_ids, *, read_cells=(), update_cell=None,
                 learning_payload=None, runtime_state=None, max_new_tokens=1):
        """Greedy token generation and optional explicit key/value learning."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError('Expected one nonempty token stream')
        if max_new_tokens < 1 or len(set(read_cells)) != len(read_cells):
            raise ValueError('Require a positive generation length and distinct read Cells')
        if (update_cell is None) != (learning_payload is None):
            raise ValueError('An update requires an explicit Cell and learning payload')
        with self._lock:
            readers = {name: self._cells[name] for name in read_cells}
            original = self._cells[update_cell] if update_cell is not None else None
        tokens = []
        state = runtime_state
        for _ in range(max_new_tokens):
            hidden, state = self.backbone.encode(input_ids, state)
            hidden = hidden[:, -1:]
            active = [cell for cell in readers.values() if cell.update_step]
            if active:
                query = self.memory.project(hidden.float())[2]
                recalled = torch.stack([self.memory.recall(query, cell) for cell in active]).mean(0)
                residual = self.gate(hidden.float()).sigmoid() * self.projection(recalled)
                hidden = hidden + residual.to(hidden.dtype)
            logits = self.backbone.logits(hidden)
            if not logits.isfinite().all():
                raise FloatingPointError('Nonfinite generation logits')
            input_ids = logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(input_ids)
        if original is not None:
            keys, values = learning_payload
            k = self.memory.project(self.backbone.encode(keys, None)[0].float())[0]
            v = self.memory.project(self.backbone.encode(values, None)[0].float())[1]
            updated, _ = self.memory.learn(k, v, original, 'online_adapt')
            if not all(t.isfinite().all() for t in (*updated.fast.values(), *updated.momentum.values())):
                raise FloatingPointError('Nonfinite Cell update')
            with self._lock:
                if self._cells.get(update_cell) is not original:
                    raise RuntimeError('Cell changed during request; retry from the original session')
                self._cells[update_cell] = updated.detached()
        return dict(tokens=torch.cat(tokens, dim=1), logits=logits, runtime_state=state,
                    pending_token=tokens[-1],
                    read_revisions={name: cell.metadata['revision'] for name, cell in readers.items()})

    def export_cell(self, name, path, *, identity, exporter):
        """Persist a stable revision through the existing portable Cell codec."""
        cell = self.get(name)
        return exporter(cell, path, identity, str(uuid.uuid4()))

    def import_cell(self, name, path, *, identity, importer):
        """Validate the entire artifact before publishing a registry entry."""
        cell = importer(path, self, identity)
        with self._lock:
            self._insert(name, cell.detached())

    def parameters(self):
        return self.memory.parameters()

    def export_bundle(self, state, names, path, *, identity, exporter):
        """Capture one consistent set of Cell revisions with an independent session."""
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError('Duplicate bundle Cell names')
        with self._lock:
            cells = [self._cells[name] for name in names]
        with TemporaryDirectory(dir=path.parent) as directory:
            stage = Path(directory) / 'bundle'
            stage.mkdir()
            self.backbone.export_runtime_state(state, stage / 'runtime')
            index = []
            for number, (name, cell) in enumerate(zip(names, cells)):
                artifact = stage / f'cell-{number}'
                digest = exporter(cell, artifact, identity, str(uuid.uuid4()))
                index.append(dict(name=name, revision=cell.metadata['revision'], sha256=digest))
            (stage / 'manifest.json').write_text(json.dumps(dict(schema=1, cells=index), indent=2))
            stage.rename(path)

    def import_bundle(self, path, *, identity, importer):
        """Restore all entries or leave the registry unchanged."""
        import json
        from pathlib import Path

        path = Path(path)
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest['schema'] != 1:
            raise ValueError('Unsupported bundle schema')
        state = self.backbone.import_runtime_state(path / 'runtime')
        entries = {}
        for index, record in enumerate(manifest['cells']):
            name = record['name']
            artifact = path / f'cell-{index}'
            integrity = json.loads((artifact / 'integrity.json').read_text())
            if not name or name in entries or integrity['artifact_sha256'] != record['sha256']:
                raise ValueError('Invalid bundle Cell index')
            cell = importer(artifact, self, identity)
            if cell.metadata['revision'] != record['revision']:
                raise ValueError('Bundle Cell revision differs')
            entries[name] = cell.detached()
        with self._lock:
            if self._cells.keys() & entries.keys():
                raise ValueError('Bundle would overwrite an existing Cell')
            self._cells.update(entries)
        return state
