from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
for path in (BASE / "implementation", BASE.parent / "implementation", ROOT / "src", ROOT / "shared/python"):
    sys.path.insert(0, str(path))
from pd001.runner import main  # noqa: E402 -- establish canonical research import roots first

if __name__ == "__main__":
    main()
