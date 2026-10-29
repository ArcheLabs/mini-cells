import copy
import runpy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(
    str(
        ROOT
        / "research/projects/cellular-language-models/experiments/clm-002-cell-differentiation-joint-value/run.py"
    ),
    run_name="clm002_cpu_tests",
)["bootstrap"]()


@pytest.fixture
def config():
    from minicells_clm002.common import configuration

    torch.set_num_threads(1)
    return copy.deepcopy(configuration())


@pytest.fixture
def ident():
    return {
        "experiment_id": "CLM-002-DIFF-JOINT-VALUE-001",
        "source_sha": "a" * 40,
        "config_sha256": "b" * 64,
        "code_sha256": "c" * 64,
        "scope": "GATE0_SYNTHETIC_CPU_ONLY",
        "scientific_training_authorized": False,
        "run_id": "d" * 32,
    }
