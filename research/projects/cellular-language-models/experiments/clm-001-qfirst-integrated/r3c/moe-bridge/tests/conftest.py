from pathlib import Path
import runpy
import pytest
import torch

BASE = Path(__file__).resolve().parents[1]
runpy.run_path(str(BASE / "run.py"), run_name="r3c_tests")


@pytest.fixture
def config():
    from minicells_r3c_moe_bridge.common import synthetic_configuration

    torch.set_num_threads(1)
    return synthetic_configuration()
