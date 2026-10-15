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
    B / "r2/implementation",
):
    sys.path.insert(0, str(p))
from clm001_r2.common import load  # noqa: E402


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
    c["oracle"]["depth_bucket_width"] = 1
    c["oracle"]["max_counterfactual_operator_calls_per_step"] = 96
    c["shadow"]["max_upper95_nll_regression"] = 0.1
    return c


from clm001_r2.data import select, build, load_assets  # noqa: E402


class Tokenizer:
    def get_vocab_size(self):
        return 16

    def token_to_id(self, x):
        return 1

    def encode(self, text):
        class Encoded:
            pass

        e = Encoded()
        e.ids = [2 + ord(v) % 14 for v in text]
        return e

    def save(self, path):
        Path(path).write_text('{"synthetic":true}')


@pytest.fixture
def assets(config, tmp_path):
    for i, role in enumerate(("train", "calibration", "admission", "development")):
        config["data"][role + "_range"] = [2 * i, 2 * i + 2]
    config["evaluation"].update(
        calibration_windows=2,
        admission_windows=2,
        development_windows=2,
        oracle_states=1,
        oracle_samples=2,
    )
    texts = [
        "violet velvet valley variation",
        "brass bridge bears balloons",
        "cyan circus cloud castle",
        "amber autumn acorn animal",
        "silver snow silent swan",
        "golden garden geese green",
        "purple planet people painting",
        "scarlet sailing salty sea",
    ]

    def source():
        for t in texts:
            yield {"text": t}
        raise AssertionError("read outside selected nonconfirmation data")

    selected, audit = select(source(), config)
    folder = tmp_path / "assets"
    build(config, selected, audit, folder, "synthetic-shard", Tokenizer())
    return folder, load_assets(config, folder)
