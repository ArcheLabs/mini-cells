from __future__ import annotations

import torch

from minicells.write_addressability import WriteAddressabilityConfig
from minicells.write_addressability_experiment import (
    make_edit_schedule,
    oracle_exact_zero_check,
)
from minicells.write_addressability_models import (
    SparseFunctionalModel,
    apply_addressed_write,
    infer_write_address,
)


def tiny_config() -> WriteAddressabilityConfig:
    return WriteAddressabilityConfig(
        observation_dim=12,
        num_features=24,
        active_features=3,
        output_dim=5,
        latent_dim=10,
        latent_topk=3,
        edit_count=4,
        edit_examples=8,
        affected_examples=16,
        invariant_examples=16,
        retention_examples_per_edit=4,
        oracle_probe_examples=32,
        pretrain_steps=2,
        pretrain_examples=32,
        pretrain_batch_size=8,
    )


def test_oracle_aligned_write_is_exactly_local() -> None:
    result = oracle_exact_zero_check()
    assert result["max_invariant_change"] == 0.0


def test_address_inference_uses_shared_coordinate() -> None:
    config = tiny_config()
    model = SparseFunctionalModel(config)
    # Make encode deterministic: first 10 observation coordinates are latent coordinates.
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.weight[:, :10] = torch.eye(10)
        model.writer.weight.zero_()
    x = torch.zeros(8, config.observation_dim)
    x[:, 4] = torch.linspace(0.5, 1.2, 8)
    # Distractors vary and are not shared across all edit examples.
    for row in range(8):
        x[row, row % 4] = 0.25 + row * 0.03
    delta = torch.tensor([0.3, -0.2, 0.4, 0.1, -0.5])
    y = x[:, 4, None] * delta[None, :]
    result = infer_write_address(model, x, y, config)
    assert result["address"] == 4
    assert result["active_fraction"] == 1.0


def test_permutation_control_changes_write_destination_not_forward_mapping() -> None:
    config = tiny_config()
    model = SparseFunctionalModel(config)
    x = torch.randn(6, config.observation_dim)
    writer_before = model.writer.weight.detach().clone()
    encoder_before = model.encoder.weight.detach().clone()
    delta = torch.ones(config.output_dim) * 0.1
    destination = apply_addressed_write(model, address=2, destination=3, delta=delta)
    assert destination == 3
    assert torch.equal(model.writer.weight[:, 2], writer_before[:, 2])
    assert torch.allclose(model.writer.weight[:, 3], writer_before[:, 3] + delta)
    # No feature/read mapping was mutated by the control itself.
    assert torch.equal(model.encoder.weight, encoder_before)


def test_recovery_load_and_schedule_are_deterministic() -> None:
    config = tiny_config()
    assert config.superposition_load == 2.0
    assert config.recovery_load > 0
    left = make_edit_schedule(config, seed=123)
    right = make_edit_schedule(config, seed=123)
    assert [task.target_feature for task in left] == [task.target_feature for task in right]
    assert torch.equal(left[0].delta, right[0].delta)


def test_world_sampler_has_exact_support_and_respects_conditioning() -> None:
    from minicells.write_addressability import SuperpositionWorld

    config = tiny_config()
    world = SuperpositionWorld(config, seed=9)
    generator = torch.Generator().manual_seed(10)
    s = world.sample_latents(
        64,
        generator=generator,
        include_feature=7,
        exclude_feature=None,
        forced_distractor=8,
    )
    assert torch.all(s.ne(0).sum(dim=1) == config.active_features)
    assert torch.all(s[:, 7].ne(0))
    assert torch.all(s[:, 8].ne(0))
    generator = torch.Generator().manual_seed(11)
    invariant = world.sample_latents(64, generator=generator, exclude_feature=7)
    assert torch.all(invariant[:, 7].eq(0))
