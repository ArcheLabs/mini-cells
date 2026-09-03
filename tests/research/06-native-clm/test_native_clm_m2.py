from __future__ import annotations

from pathlib import Path

import torch

from minicells.native_clm_m2 import (
    NativeCLMM2Config,
    _train_phase,
    compare_arms,
    evaluate_domain,
    frozen_state_sha256,
)
from minicells.native_clm_v0 import NativeCLM, NativeCLMConfig


def tiny_model() -> NativeCLM:
    config = NativeCLMConfig(
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        initial_cells=4,
        active_cells=2,
        cellular_layer_index=0,
        certificate_max_rank=4,
    )
    model = NativeCLM(config)
    for index, cell in enumerate(model.cellular.cells):
        vector = torch.zeros(config.d_model)
        vector[index] = 1.0
        assert cell.add_certificate_vector(vector)
    return model


def write_corpus(path: Path, word: str) -> None:
    path.write_text((f"{word} alpha beta gamma delta.\n" * 200), encoding="utf-8")


def test_m2_cell_only_protected_and_unsafe_paths(tmp_path: Path) -> None:
    torch.manual_seed(7)
    protected = tiny_model()
    unsafe = tiny_model()
    unsafe.load_state_dict(protected.state_dict())
    corpus = tmp_path / "domain.txt"
    write_corpus(corpus, "domain")
    config = NativeCLMM2Config(
        batch_size=2,
        steps_per_phase=3,
        eval_batches=1,
        log_interval=1,
        warmup_steps=1,
        certificate_update_interval=0,
        precision="fp32",
    )

    frozen_before = frozen_state_sha256(protected)
    p = _train_phase(
        protected,
        corpus,
        device=torch.device("cpu"),
        config=config,
        seed=1,
        protected=True,
        phase="B",
    )
    u = _train_phase(
        unsafe,
        corpus,
        device=torch.device("cpu"),
        config=config,
        seed=1,
        protected=False,
        phase="B",
    )
    assert frozen_state_sha256(protected) == frozen_before
    assert p["learner_replay_bytes"] == 0
    assert p["projection_ratio_min"] < 1.0
    assert u["projection_ratio_min"] == 1.0
    metrics = evaluate_domain(protected, corpus, device=torch.device("cpu"), config=config)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["active_fraction_vs_dense"] == 0.5


def _fake_arm(*, protected: bool, final_a: float, final_b: float, final_c: float):
    def m(loss):
        return {
            "loss": loss,
            "perplexity": 2.0,
            "route_entropy": 0.5,
            "active_fraction_vs_dense": 0.25,
            "cell_usage_share": [0.125] * 8,
        }

    return {
        "seed": 1,
        "protected": protected,
        "parent_checkpoint_sha256": "abc",
        "cell_only_writes": True,
        "shared_and_router_frozen": True,
        "learner_replay_bytes": 0,
        "cell_count": 8,
        "evaluation_matrix": {
            "initial": {"A": m(1.0), "B": m(2.0), "C": m(2.0), "D": m(2.0)},
            "after_B": {"A": m(1.0), "B": m(1.0), "C": m(2.0), "D": m(2.0)},
            "after_C": {"A": m(1.0), "B": m(1.0), "C": m(1.0), "D": m(2.0)},
            "after_D": {"A": m(final_a), "B": m(final_b), "C": m(final_c), "D": m(1.0)},
        },
    }


def test_registered_comparison_can_distinguish_retention() -> None:
    protected = _fake_arm(protected=True, final_a=1.05, final_b=1.05, final_c=1.05)
    unsafe = _fake_arm(protected=False, final_a=1.30, final_b=1.30, final_c=1.30)
    result = compare_arms(
        protected,
        unsafe,
        thresholds={
            "max_active_fraction": 0.30,
            "min_phase_gain": 0.05,
            "max_A_regression": 0.20,
            "min_unsafe_mean_forgetting": 0.03,
            "min_retention_advantage": 0.02,
            "min_plasticity_ratio_vs_unsafe": 0.80,
        },
    )
    assert result["pass"] is True
    assert result["retention_advantage"] > 0.20
