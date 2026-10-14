from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = (
    ROOT
    / "research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth"
)
for path in (BASE / "implementation", BASE.parent / "implementation", ROOT / "src", ROOT / "shared/python"):
    sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def proposal():
    from pd001.configs import load

    return load()[0]


@pytest.fixture
def tiny(proposal):
    from pd001.runner import fixture_config

    return fixture_config(proposal, "P50-A32")
