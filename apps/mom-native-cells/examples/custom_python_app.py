"""Use MiniCells as a Python application dependency."""
from dataclasses import asdict
import json
import sys
from tempfile import TemporaryDirectory

from minicells import MiniCells


def answer(mc: MiniCells, prompt: str) -> dict:
    return asdict(mc.query(prompt, max_new_tokens=24))


def main():
    with TemporaryDirectory() as root, MiniCells.open(storage=root) as mc:
        cell = mc.cells.create("application notes")
        mc.cells.write(cell.id, "The library opens at nine on weekdays.")
        prompt = " ".join(sys.argv[1:]) or "When does the library open?"
        print(json.dumps(answer(mc, prompt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
