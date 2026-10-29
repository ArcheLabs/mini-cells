import copy
import runpy
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(
    str(ROOT / "research/projects/cellular-language-models/experiments/jct-001/run.py"),
    run_name="test_bootstrap",
)["bootstrap"]()


@pytest.fixture
def config():
    from minicells_jct001.common import configuration

    torch.set_num_threads(1)
    return copy.deepcopy(configuration())


@pytest.fixture
def identity():
    return dict(
        experiment_id="JCT-001",
        source_sha="a" * 40,
        config_sha256="b" * 64,
        scope="SYNTHETIC_CPU_ENGINEERING_ONLY",
        run_id="c" * 32,
    )
