"""Tiny synthetic engineering workload; no public dataset, GPU, or R3b authorization."""

from pathlib import Path
from clm001_r3.common import load
from clm001_r3.data import build, select


class SyntheticTokenizer:
    def get_vocab_size(self):
        return 16

    def token_to_id(self, value):
        return 1

    def encode(self, text):
        class Encoded:
            pass

        result = Encoded()
        result.ids = [2 + ord(v) % 14 for v in text]
        return result

    def save(self, path):
        Path(path).write_text('{"synthetic":true,"fixture":"continuation-v1"}')


def configuration():
    c = load()
    c.update(synthetic=True, seed=123)
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
        steps=10,
        effective_batch=1,
        microbatch=1,
        sequence_length=4,
        target_tokens=40,
        warmup_steps=1,
        checkpoint_interval=1,
        evaluation_interval=5,
    )
    c["policy"]["lambda_cost"] = 0.01
    c["oracle"].update(
        suffix_samples=2, depth_bucket_width=1, max_counterfactual_operator_calls_per_step=96
    )
    c["shadow"].update(
        path_plan=[0, 3, 4],
        private_train_steps=1,
        minimum_admission_stories=2,
        min_cost_aware_improvement=0.001,
        nll_regression_tolerance=0.1,
        bootstrap_replicates=100,
        max_upper95_nll_regression=0.1,
        cell_proposal_step=3,
        connection_path_step=6,
        admission_step=8,
    )
    c["competition"].update(events_per_step=4, candidate_cells=2, depth_bucket_width=1, beta=0.1)
    c["alignment"].update(
        minimum_pairs=1,
        minimum_stories=2,
        grid_stories=2,
        depths=[0, 1, 2],
        cell_pairs=[[0, 1]],
        replicates=100,
    )
    c["evaluation"].update(
        calibration_windows=2,
        admission_windows=2,
        development_windows=2,
        oracle_states=1,
        oracle_samples=2,
        periodic_diagnostics_enabled=True,
        snapshots=[0, 3, 6, 8, 10],
        counterfactual_snapshots=[0, 10],
    )
    for i, role in enumerate(("train", "calibration", "admission", "development")):
        c["data"][role + "_range"] = [2 * i, 2 * i + 2]
    return c


def prepare(destination, c):
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
    chosen, audit = select(({"text": t} for t in texts), c)
    build(c, chosen, audit, destination, "synthetic-continuation-fixture-v1", SyntheticTokenizer())
    return destination
