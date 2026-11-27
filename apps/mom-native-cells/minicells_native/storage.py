"""SDK registry and collection transactions over native Cell bundles."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from safetensors import SafetensorError

from .backend import sha
from .cells import Cells
from .errors import (
    CellAlreadyExistsError, CellArchiveCorruptedError, CellExportError, CellImportError,
    CellNotFoundError, CellWriteError, IncompatibleModelError,
)

READ_ERRORS = (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, SafetensorError)
COLLECTION_FORMAT = "minicells-collection"
MAX_BUNDLE_BYTES = 151_000_000


@dataclass(frozen=True)
class Cell:
    id: str
    name: str
    revision: int
    created_at: str
    updated_at: str
    tokens_written: int
    parent_cell_id: str | None
    state_completeness: str = "recurrent_only"

    @classmethod
    def from_manifest(cls, meta):
        return cls(meta["cell_id"], meta["name"], meta["revision"], meta["created_at"],
                   meta["updated_at"], meta["tokens_written"], meta.get("parent_cell_id"),
                   meta.get("state_completeness", "recurrent_only"))


def revision(meta):
    keys = ("revision", "parent_revision", "parent_cell_id", "updated_at", "tokens_written",
            "tensor_sha256")
    return {key: meta.get(key) for key in keys}


class RevisionStore(Cells):
    def store(self, metadata, tensors):
        metadata = dict(metadata)
        path = self.path(metadata["cell_id"])
        if path.exists():
            with zipfile.ZipFile(path) as bundle:
                previous = json.loads(bundle.read("manifest.json"))
            metadata["sdk_revisions"] = [
                *previous.get("sdk_revisions", ()), revision(previous),
            ]
        else:
            metadata.pop("sdk_revisions", None)
        if len(json.dumps({**metadata, "tensor_sha256": "0" * 64}).encode()) > 100_000:
            raise CellWriteError("Revision metadata exceeds the native manifest size limit")
        return super().store(metadata, tensors)

    def decode(self, path):
        try:
            meta, tensors = super().decode(path)
            self.path(meta["cell_id"])
            Cell.from_manifest(meta)
            if type(meta["revision"]) is not int or meta["revision"] < 0:
                raise ValueError("Invalid revision")
            if type(meta["tokens_written"]) is not int or meta["tokens_written"] < 0:
                raise ValueError("Invalid token count")
            return meta, tensors
        except FileNotFoundError as exc:
            raise CellNotFoundError(str(path)) from exc
        except READ_ERRORS as exc:
            if str(exc) == "Incompatible backbone or tokenizer":
                raise IncompatibleModelError(f"{path}: {exc}") from exc
            raise CellArchiveCorruptedError(f"{path}: {exc}") from exc


@contextmanager
def atomic_output(destination):
    path = Path(destination)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
    try:
        yield pending
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


class CellManager:
    def __init__(self, runtime, store):
        self._runtime = runtime
        self._store = store
        self._registry = {}
        self._transaction = store.root / ".minicells-import.json"
        if self._transaction.exists():
            for cell_id in json.loads(self._transaction.read_text(encoding="utf-8")):
                store.path(cell_id).unlink(missing_ok=True)
            self._transaction.unlink()
        for path in sorted(store.root.glob("*.zip")):
            try:
                store.path(path.stem)
            except ValueError:
                continue  # Unrelated archive, outside the UUID-named registry.
            meta, _ = store.decode(path)
            if path != store.path(meta["cell_id"]):
                raise CellArchiveCorruptedError(f"Registry filename does not match Cell ID: {path}")
            self._register(meta)

    def _register(self, meta):
        self._registry[meta["cell_id"]] = meta
        return Cell.from_manifest(meta)

    def _get(self, cell_id):
        try:
            return self._registry[cell_id]
        except (KeyError, TypeError) as exc:
            raise CellNotFoundError(f"Cell does not exist: {cell_id!r}") from exc

    def _state(self, cell_id):
        self._get(cell_id)
        return self._store.read(cell_id)[1]

    def create(self, name, *, full_state=False):
        with self._runtime._operation():
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
                raise ValueError("Cell name must contain 1–120 characters")
            if type(full_state) is not bool or full_state and not self._runtime._preview:
                raise ValueError("Full-state Cells require the pinned Developer Preview")
            return self._register(self._store.create(name.strip(), full_state=full_state))

    def get(self, cell_id):
        with self._runtime._operation():
            return Cell.from_manifest(self._get(cell_id))

    def list(self):
        with self._runtime._operation():
            return [Cell.from_manifest(self._registry[key]) for key in sorted(self._registry)]

    def write(self, cell_id, text, *, chunk_size="auto", progress=None, cancelled=None, resume=True):
        with self._runtime._operation():
            self._get(cell_id)
            chunk = self._runtime._chunk_size(chunk_size)
            if not isinstance(text, str) or not text.strip():
                raise CellWriteError("Write requires nonempty text")
            try:
                item = self._store.write(cell_id, text, chunk, progress or (lambda *_: None),
                                         cancelled or (lambda: False), resume=resume)
                return self._register(item) if item is not None else None
            except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
                raise CellWriteError(f"Writing Cell {cell_id}: {exc}") from exc

    append = write

    def fork(self, cell_id, name=None):
        with self._runtime._operation():
            self._get(cell_id)
            if name is not None and (not isinstance(name, str) or not 1 <= len(name.strip()) <= 120):
                raise ValueError("Cell name must contain 1–120 characters")
            return self._register(self._store.fork(cell_id, name=name.strip() if name else None))

    def revisions(self, cell_id):
        with self._runtime._operation():
            meta = self._get(cell_id)
            return [dict(entry) for entry in (*meta.get("sdk_revisions", ()), revision(meta))]

    def delete(self, cell_id):
        with self._runtime._operation():
            self._get(cell_id)
            path = self._store.path(cell_id)
            path.with_suffix(".stream").unlink(missing_ok=True)
            path.unlink()
            del self._registry[cell_id]

    def _check_destination(self, path):
        path = Path(path).resolve()
        if path.parent == self._store.root:
            try:
                self._store.path(path.stem)
            except ValueError:
                pass
            else:
                raise CellExportError("Export would use a reserved Cell registry filename")
            if path.name.startswith(".minicells"):
                raise CellExportError("Export would use a reserved Cell registry filename")
        return path

    def export(self, cell_id, path):
        with self._runtime._operation():
            self._get(cell_id)
            path = self._check_destination(path)
            try:
                with atomic_output(path) as staged:
                    shutil.copyfile(self._store.path(cell_id), staged)
            except OSError as exc:
                raise CellExportError(f"Could not export Cell {cell_id}: {exc}") from exc
            return path

    def export_collection(self, path, *, cell_ids=None):
        with self._runtime._operation():
            ids = sorted(self._registry) if cell_ids is None else list(cell_ids)
            if len(ids) != len(set(ids)):
                raise ValueError("Collection requires distinct Cell IDs")
            for cell_id in ids:
                self._get(cell_id)
            path = self._check_destination(path)
            try:
                with atomic_output(path) as staged, zipfile.ZipFile(staged, "w") as archive:
                    entries = []
                    for cell_id in ids:
                        file = self._store.path(cell_id)
                        name = "cells/" + file.name
                        archive.write(file, name)
                        entries.append(dict(cell_id=cell_id, path=name, sha256=sha(file)))
                    archive.writestr("manifest.json", json.dumps(dict(
                        format=COLLECTION_FORMAT, format_version=1, cells=entries)))
            except OSError as exc:
                raise CellExportError(f"Could not export collection: {exc}") from exc
            return path

    def _commit(self, staged):
        ids = [meta["cell_id"] for meta, _ in staged]
        if len(ids) != len(set(ids)):
            raise CellArchiveCorruptedError("Duplicate Cell IDs in collection")
        for cell_id in ids:
            if cell_id in self._registry or self._store.path(cell_id).exists():
                raise CellAlreadyExistsError(f"Cell {cell_id} already exists")
        with atomic_output(self._transaction) as marker:
            marker.write_text(json.dumps(ids), encoding="utf-8")
        try:
            for meta, file in staged:
                path = self._store.path(meta["cell_id"])
                file.replace(path)
            self._transaction.unlink()
        except BaseException:
            for cell_id in ids:
                self._store.path(cell_id).unlink(missing_ok=True)
            self._transaction.unlink(missing_ok=True)
            raise
        return [self._register(meta) for meta, _ in staged]

    def import_cell(self, path, *, on_conflict="error"):
        with self._runtime._operation():
            self._check_policy(on_conflict)
            with tempfile.TemporaryDirectory(dir=self._store.root) as temp:
                file = Path(temp) / "cell.zip"
                try:
                    if Path(path).stat().st_size > MAX_BUNDLE_BYTES:
                        raise CellArchiveCorruptedError("Cell bundle exceeds size limit")
                    shutil.copyfile(path, file)
                    meta, _ = self._store.decode(file)
                    return self._commit([(meta, file)])[0]
                except OSError as exc:
                    raise CellImportError(f"Could not import {path}: {exc}") from exc

    def import_collection(self, path, *, on_conflict="error"):
        with self._runtime._operation():
            self._check_policy(on_conflict)
            try:
                with tempfile.TemporaryDirectory(dir=self._store.root) as temp:
                    with zipfile.ZipFile(path) as archive:
                        if archive.getinfo("manifest.json").file_size > 1_000_000:
                            raise ValueError("Collection manifest exceeds size limit")
                        manifest = json.loads(archive.read("manifest.json"))
                        if (manifest["format"], manifest["format_version"]) != (COLLECTION_FORMAT, 1):
                            raise ValueError("Unsupported collection format")
                        entries = manifest["cells"]
                        if sorted(archive.namelist()) != sorted(["manifest.json", *(item["path"] for item in entries)]):
                            raise ValueError("Unexpected collection entries")
                        staged = []
                        for i, item in enumerate(entries):
                            name = "cells/" + self._store.path(item["cell_id"]).name
                            if item["path"] != name or archive.getinfo(name).file_size > MAX_BUNDLE_BYTES:
                                raise ValueError(f"Invalid collection entry: {name}")
                            file = Path(temp) / f"{i}.zip"
                            with archive.open(name) as stream, file.open("wb") as output:
                                shutil.copyfileobj(stream, output)
                            if sha(file) != item["sha256"]:
                                raise ValueError(f"Checksum mismatch: {name}")
                            meta, _ = self._store.decode(file)
                            if meta["cell_id"] != item["cell_id"]:
                                raise ValueError(f"Embedded Cell ID mismatch: {name}")
                            staged.append((meta, file))
                    return self._commit(staged)
            except READ_ERRORS as exc:
                raise CellArchiveCorruptedError(f"Could not import collection {path}: {exc}") from exc

    @staticmethod
    def _check_policy(policy):
        if policy != "error":
            raise CellImportError("Supported conflict strategy: error")
