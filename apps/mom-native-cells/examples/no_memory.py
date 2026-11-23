"""Run the frozen model with an empty working cache."""
from tempfile import TemporaryDirectory

from minicells import MiniCells


def main():
    with TemporaryDirectory() as root, MiniCells.open(storage=root) as mc:
        result = mc.query("What is the capital of France?", cells=[], max_new_tokens=16)
        print(result.text, result.fallback_reason)


if __name__ == "__main__":
    main()
