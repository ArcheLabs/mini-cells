"""Use exactly one specified Cell."""
from tempfile import TemporaryDirectory

from minicells import MiniCells


def main():
    with TemporaryDirectory() as root, MiniCells.open(storage=root) as mc:
        person = mc.cells.create("person")
        mc.cells.write(person.id, "Alice was born in London.")
        mc.cells.write(mc.cells.create("sibling").id, "Robert was born in Milan.")
        result = mc.query("Where was Alice born?", cells=[person.id], max_new_tokens=16)
        print(result.text, result.selected_cell_ids)


if __name__ == "__main__":
    main()
