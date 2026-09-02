from __future__ import annotations

import math
from pathlib import Path

import torch

from minicells.native_clm_train import NativeCLMTrainConfig, train_m1
from minicells.native_clm_v0 import NativeCLM, NativeCLMConfig


def tiny_config() -> NativeCLMConfig:
    return NativeCLMConfig(
        max_seq_len=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        initial_cells=4,
        active_cells=2,
        cellular_layer_index=0,
        certificate_max_rank=8,
    )


def test_sparse_forward_backward_certificate_and_spawn(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = NativeCLM(tiny_config())
    tokens = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    output = model(tokens, targets, return_info=True)

    assert torch.isfinite(output["loss"])
    assert output["cell_info"]["active_fraction_vs_dense"] == 0.5
    output["loss"].backward()
    assert any(p.grad is not None for p in model.parameter_groups()["router"])
    assert any(p.grad is not None for p in model.parameter_groups()["cells"])

    added = model.update_certificates(output["cell_info"])
    assert added > 0
    ratios = model.project_cell_gradients_()
    assert len(ratios) == 4
    assert all(0.0 <= value <= 1.000001 for value in ratios.values())

    with torch.no_grad():
        route_state = output["cell_info"]["route_input"].mean(dim=(0, 1))
        route_key = model.cellular.query_proj(route_state)
    child = model.spawn_cell(parent_id=0, route_key=route_key)
    assert child == 4
    assert model.cell_count == 5

    checkpoint = tmp_path / "native.pt"
    model.save_checkpoint(checkpoint, extra={"child": child})
    restored, extra = NativeCLM.load_checkpoint(checkpoint)
    assert restored.cell_count == 5
    assert extra == {"child": 4}
    assert torch.isfinite(restored(tokens, targets)["loss"])


def test_canonical_m1_is_roughly_twelve_million_parameters() -> None:
    model = NativeCLM(NativeCLMConfig())
    counts = model.parameter_count()
    assert 10_000_000 <= counts["total"] <= 15_000_000
    assert counts["cells"] > 0
    assert counts["router"] > 0
    assert counts["shared"] > counts["cells"]


def test_tiny_m1_training_runtime(tmp_path: Path) -> None:
    text = (
        "Once upon a time a tiny Cell learned the next byte from a short story.\n" * 120
    )
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text(text, encoding="utf-8")
    validation_path.write_text(text, encoding="utf-8")

    train_config = NativeCLMTrainConfig(
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=3,
        eval_interval=3,
        eval_batches=1,
        log_interval=1,
        checkpoint_interval=0,
        warmup_steps=1,
        certificate_update_interval=2,
        precision="fp32",
        generation_tokens=4,
    )
    summary = train_m1(
        model_config=tiny_config(),
        train_config=train_config,
        train_path=train_path,
        validation_path=validation_path,
        output_dir=tmp_path / "output",
        device="cpu",
    )

    assert summary["scientific_decision"] is False
    assert math.isfinite(summary["initial_eval"]["loss"])
    assert math.isfinite(summary["final_eval"]["loss"])
    assert summary["max_observed_router_grad_norm"] > 0
    assert summary["max_observed_cell_grad_norm"] > 0
    assert (tmp_path / "output" / "summary.json").exists()
    assert (tmp_path / "output" / "final-model.pt").exists()
