"""Errors raised by the Python SDK."""


class MiniCellsError(Exception):
    pass


class ModelLoadError(MiniCellsError):
    pass


class CellNotFoundError(MiniCellsError, FileNotFoundError):
    pass


class CellAlreadyExistsError(MiniCellsError):
    pass


class CellLimitError(MiniCellsError):
    pass


class CellImportError(MiniCellsError):
    pass


class CellExportError(MiniCellsError):
    pass


class CellArchiveCorruptedError(CellImportError):
    pass


class IncompatibleModelError(CellImportError):
    pass


class InvalidCellSelectionError(MiniCellsError, ValueError):
    pass


class UnsupportedChunkSizeError(MiniCellsError, ValueError):
    pass


class RuntimeClosedError(MiniCellsError):
    pass


class CellWriteError(MiniCellsError):
    pass


class StorageBusyError(MiniCellsError):
    pass


__all__ = ['MiniCellsError', 'ModelLoadError', 'CellNotFoundError', 'CellAlreadyExistsError', 'CellLimitError', 'CellImportError', 'CellExportError', 'CellArchiveCorruptedError', 'IncompatibleModelError', 'InvalidCellSelectionError', 'UnsupportedChunkSizeError', 'RuntimeClosedError', 'CellWriteError', 'StorageBusyError']
