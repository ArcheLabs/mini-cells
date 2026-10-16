from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
PD = BASE.parent.parent / "006-path-first-clm/extension-parameter-depth"
for path in (
    ROOT / "src",
    ROOT / "shared/python",
    PD.parent / "implementation",
    PD / "implementation",
    PD / "exploration-fast-001/implementation",
    BASE.parent / "implementation",
    BASE.parent / "r2/implementation",
    BASE / "implementation",
):
    sys.path.insert(0, str(path))
from clm001_r3.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
