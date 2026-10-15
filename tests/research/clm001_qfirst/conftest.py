import sys
from pathlib import Path
import pytest

R = Path(__file__).resolve().parents[3]
B = R / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated"
PD = B.parent / "006-path-first-clm/extension-parameter-depth"
for p in (
    R / "src",
    R / "shared/python",
    PD.parent / "implementation",
    PD / "implementation",
    PD / "exploration-fast-001/implementation",
    B / "implementation",
):
    sys.path.insert(0, str(p))
from clm001.common import load  # noqa: E402


@pytest.fixture
def config():
    c = load()
    c["synthetic"] = True
    c["seed"] = 123
    c["model"].update(
        width=8,
        heads=2,
        accepted_cells=2,
        reserve_slots=1,
        cell_hidden=12,
        vocab_size=16,
        max_sequence_length=8,
        budget=3,
        router_hidden=8,
        step_width=4,
        edge_rank=3,
    )
    c["training"].update(
        steps=2, effective_batch=1, microbatch=1, sequence_length=4, target_tokens=8, warmup_steps=1
    )
    c["policy"]["lambda_cost"] = 0.01
    c["oracle"]["suffix_samples"] = 2
    c["shadow"].update(
        path_plan=[0, 3, 4],
        private_train_steps=1,
        minimum_admission_stories=2,
        min_cost_aware_improvement=0.001,
        nll_regression_tolerance=0.1,
        bootstrap_replicates=100,
    )
    return c
