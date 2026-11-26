"""Minimal state lifecycle on the supported Developer Preview profile."""
from tempfile import TemporaryDirectory
from pathlib import Path

from minicells_native import MiniCells

with TemporaryDirectory() as temp:
    root = Path(temp)
    bundle = root / "sample.cell"
    with MiniCells.open_preview(root / "write") as mc:
        cell = mc.cells.create("Project Ardent")
        mc.cells.write(cell.id, "The internal code for Project Ardent is LIME-173.")
        mc.cells.export(cell.id, bundle)
    with MiniCells.open_preview(root / "restore") as mc:
        cell = mc.cells.import_cell(bundle)
        print(mc.query("What is the internal code for Project Ardent?", cells=[cell.id]).text)
