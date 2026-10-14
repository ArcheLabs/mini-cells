"""Use the repository's identity, RNG, atomic JSON and checkpoint components."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
ROOT = next(p for p in BASE.parents if (p / 'pyproject.toml').is_file())
for path in (
    ROOT / 'src', ROOT / 'shared/python',
    ROOT / 'research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation',
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from minicells.reproducibility import set_global_seed, environment_info
from experiments.common.train import write_json as _write_json
from exp005.checkpoint import canonical_sha256, file_sha256, save_checkpoint, restore_checkpoint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)


def code_digest():
    files = list((BASE / 'src').rglob('*.py')) + [BASE / 'run.py',
        ROOT / 'src/minicells/reproducibility.py',
        ROOT / 'shared/python/experiments/common/train.py',
        ROOT / 'shared/python/experiments/common/models.py',
        ROOT / 'research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/exp005/checkpoint.py',
        ROOT / 'research/projects/cellular-language-models/context/08-hybrid-clm/scripts/research/publish_experiment_results.py',
        ROOT / 'research/projects/continual-learning/context/04-continual-learning/scripts/research/publish_core_validation_007.py']
    return canonical_sha256({str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(files)})
