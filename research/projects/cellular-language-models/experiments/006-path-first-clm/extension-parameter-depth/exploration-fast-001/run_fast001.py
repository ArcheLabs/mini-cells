from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
for path in (
    ROOT / "src",
    ROOT / "shared/python",
    BASE.parent.parent / "implementation",
    BASE.parent / "implementation",
    BASE / "implementation",
):
    sys.path.insert(0, str(path))
from fast001.entry import main  # noqa: E402

if __name__ == "__main__":
    main()
