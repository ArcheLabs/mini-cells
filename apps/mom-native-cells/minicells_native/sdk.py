"""Python-first runtime with a single shared native MoM backbone."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from threading import RLock

from .errors import (
    CellLimitError, InvalidCellSelectionError, RuntimeClosedError,
    StorageBusyError, UnsupportedChunkSizeError,
)
from .storage import Cell, CellManager, RevisionStore

DEFAULT_MODEL = "linear-moe-hub/MoM-Gated-Deltanet-340M"


@dataclass(frozen=True)
class QueryResult:
    text: str
    mode: str
    candidate_cell_ids: list[str]
    selected_cell_ids: list[str]
    cell_scores: dict[str, float]
    fallback_reason: str | None
    latency: float
    generation_tokens: int
    diagnostics: dict


class MiniCells:
    """Open one exclusive local registry and one frozen native MoM runtime."""

    @classmethod
    def open(cls, model=DEFAULT_MODEL, storage="./minicells_data", device="cuda", dtype="auto",
             max_cells=8, write_chunk_size="auto", selection_threshold=None, *, selection_top_k=2,
             cache_dir=None, source=None, tokenizer=None):
        from .loader import load_backbone

        config = dict(max_cells=max_cells, write_chunk_size=write_chunk_size,
                      selection_threshold=selection_threshold, selection_top_k=selection_top_k)
        with cls._storage_lock(storage) as (root, lock):
            backbone = load_backbone(model, device, dtype, cache_dir, source, tokenizer)
            return cls(backbone, root, lock, **config)

    @staticmethod
    @contextmanager
    def _storage_lock(storage):
        from filelock import FileLock, Timeout

        root = Path(storage).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(root / ".minicells.lock"), thread_local=False)
        try:
            lock.acquire(timeout=0)
        except Timeout as exc:
            raise StorageBusyError(f"Cell storage already open: {root}") from exc
        try:
            yield root, lock
        except BaseException:
            lock.release()
            raise
        # On success the runtime owns the lock until close().

    @classmethod
    def _from_backbone(cls, backbone, storage, **options):
        with cls._storage_lock(storage) as (root, lock):
            return cls(backbone, root, lock, **options)

    def __init__(self, backbone, storage, lock, *, max_cells=8, write_chunk_size="auto",
                 selection_threshold=None, selection_top_k=2):
        if type(max_cells) is not int or max_cells < 1:
            raise ValueError("max_cells must be a positive integer")
        if type(selection_top_k) is not int or selection_top_k < 1:
            raise ValueError("selection_top_k must be a positive integer")
        if (selection_threshold is not None and
                (not math.isfinite(selection_threshold) or selection_threshold < 0)):
            raise ValueError("selection_threshold must be finite and nonnegative")
        self._mutex, self._lock, self._active = RLock(), lock, False
        self._backbone = backbone
        self._max_cells, self._top_k = max_cells, selection_top_k
        self._threshold = selection_threshold
        self._native_limit = 2048 if getattr(backbone, "native_bf16", False) else 64
        self._write_chunk = 64
        self._write_chunk = self._chunk_size(write_chunk_size)
        self.cells = CellManager(self, RevisionStore(storage, backbone))
        self.model_info = {**backbone.identity, "parameters": getattr(backbone, "parameters", None)}

    @contextmanager
    def _operation(self):
        with self._mutex:
            if self._backbone is None:
                raise RuntimeClosedError("MiniCells runtime is closed")
            if self._active:
                raise RuntimeError("A runtime callback cannot start another Cell operation")
            self._active = True
            try:
                yield
            finally:
                self._active = False

    def _chunk_size(self, value):
        if value == "auto":
            return self._write_chunk
        if type(value) is not int or not 1 <= value <= self._native_limit:
            raise UnsupportedChunkSizeError(f"Write chunk must be auto or 1–{self._native_limit}")
        return value

    def close(self):
        with self._mutex:
            if self._active:
                raise RuntimeError("Cannot close MiniCells during a runtime callback")
            if self._backbone is not None:
                self._backbone = None
                self.cells._store.backbone = None
                self._lock.release()

    def __enter__(self):
        with self._operation():
            return self

    def __exit__(self, *exc):
        self.close()

    def query(self, prompt, cells="auto", *, max_new_tokens=128, temperature=0):
        from .query import run_query

        with self._operation():
            return run_query(self, prompt, cells, max_new_tokens, temperature)

    def _weighted_query(self, prompt, ids, weights, max_new_tokens, temperature):
        from .query import run_query

        with self._operation():
            return run_query(self, prompt, ids, max_new_tokens, temperature, weights)

    def _candidate_ids(self, cells):
        if isinstance(cells, str) and cells == "auto":
            ids = sorted(self.cells._registry)
            if len(ids) > self._max_cells:
                raise CellLimitError(f"{len(ids)} registered Cells exceed max_cells={self._max_cells}")
            return "auto", ids
        if (not isinstance(cells, (list, tuple)) or any(not isinstance(c, str) for c in cells)
                or len(set(cells)) != len(cells)):
            raise InvalidCellSelectionError("Use 'auto' or a list of distinct Cell IDs")
        for cell_id in cells:
            self.cells._get(cell_id)
        return "manual", list(cells)


__all__ = ["Cell", "MiniCells", "QueryResult"]
