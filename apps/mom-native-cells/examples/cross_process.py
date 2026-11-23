"""Export in one process, then import and append in a separate process."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from minicells import MiniCells


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
        with MiniCells.open(storage=root / "receiver") as mc:
            cell = mc.cells.import_cell(root / "people.zip")
            print(mc.query("Where was Alice born?", cells=[cell.id], max_new_tokens=16).text)
            mc.cells.append(cell.id, "Alice works in Edinburgh.")
            print(mc.query("Where does Alice work?", cells=[cell.id], max_new_tokens=16).text)
        return
    with TemporaryDirectory() as directory:
        root = Path(directory)
        with MiniCells.open(storage=root / "sender") as mc:
            cell = mc.cells.create("people")
            mc.cells.write(cell.id, "Alice was born in London.")
            mc.cells.export(cell.id, root / "people.zip")
        subprocess.run([sys.executable, __file__, str(root)], check=True)


if __name__ == "__main__":
    main()
