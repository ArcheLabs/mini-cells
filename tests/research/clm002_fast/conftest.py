import copy
import runpy
from pathlib import Path

import pytest
import torch

BASE = (
    Path(__file__).resolve().parents[3]
    / "research/projects/cellular-language-models/experiments/clm-002-fast-001"
)
runpy.run_path(str(BASE / "run.py"), run_name="fast_tests")["bootstrap"]()


@pytest.fixture
def config():
    from minicells_fast001.common import configuration

    torch.set_num_threads(1)
    return copy.deepcopy(configuration())
