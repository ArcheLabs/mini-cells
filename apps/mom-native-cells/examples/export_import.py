"""Move a collection from one registry to another."""
from pathlib import Path
from tempfile import TemporaryDirectory

from minicells import MiniCells


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "collection.zip"
        with MiniCells.open(storage=root / "source") as source:
            cell = source.cells.create("people")
            source.cells.write(cell.id, "Alice was born in London.")
            source.cells.export_collection(archive)
        with MiniCells.open(storage=root / "destination") as destination:
            destination.cells.import_collection(archive)
            print(destination.query("Where was Alice born?", cells=[cell.id], max_new_tokens=16).text)


if __name__ == "__main__":
    main()
