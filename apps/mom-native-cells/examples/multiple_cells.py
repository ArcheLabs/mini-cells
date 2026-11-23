"""Score independent Cells and automatically select context."""
from tempfile import TemporaryDirectory

from minicells import MiniCells


def main():
    with TemporaryDirectory() as root, MiniCells.open(storage=root) as mc:
        facts = {"person": "Alice was born in London.", "date": "Alice was born in 1995."}
        for name, text in facts.items():
            mc.cells.write(mc.cells.create(name).id, text)
        result = mc.query("When and where was Alice born?", max_new_tokens=24)
        print(result.text, result.selected_cell_ids, result.cell_scores)


if __name__ == "__main__":
    main()
