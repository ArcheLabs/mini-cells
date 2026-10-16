import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated"
PD = BASE.parent / "006-path-first-clm/extension-parameter-depth"
for p in (
    ROOT / "src",
    ROOT / "shared/python",
    PD.parent / "implementation",
    PD / "implementation",
    PD / "exploration-fast-001/implementation",
    BASE / "implementation",
    BASE / "r2/implementation",
    BASE / "r3/implementation",
):
    sys.path.insert(0, str(p))
