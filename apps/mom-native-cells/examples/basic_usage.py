"""Open a local runtime, append a fact and query it."""
from tempfile import TemporaryDirectory

from minicells import MiniCells


def main():
    with TemporaryDirectory() as root, MiniCells.open(storage=root) as mc:
        cell = mc.cells.create("people")
        mc.cells.write(cell.id, "Alice was born in London.")
        print(mc.query("Where was Alice born?", max_new_tokens=16).text)


if __name__ == "__main__":
    main()
