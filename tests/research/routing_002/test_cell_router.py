from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from experiments.cell_internal_routing_002.models.cell_router import (
    CellInternalRouter,
    LanguageModel,
)
from experiments.runners.models import LanguageModel as Experiment001LanguageModel
from experiments.cell_internal_routing_002.train import validate_config
from experiments.cell_internal_routing_002.report import matrix_report, seed_report


@pytest.fixture
def config():
    return {
        'vocab_size': 32,
        'width': 16,
        'sequence_length': 4,
        'layers': 2,
        'heads': 4,
        'experts': 3,
        'top_k': 2,
        'expert_hidden': 8,
        'max_hops': 3,
        'path_samples': 8,
    }


def test_soft_transitions_are_differentiable_and_record_executed_cost(config):
    torch.manual_seed(42)
    model = LanguageModel(config, 'cell-internal')
    tokens = torch.randint(0, config['vocab_size'], (2, config['sequence_length']))
    logits, balance = model(tokens)
    loss = logits.square().mean() + 0.01 * balance + 0.001 * model.compute_penalty()
    loss.backward()

    for router in model.routing_modules():
        assert router.cells[0].router.weight.grad is not None
        assert torch.isfinite(router.cells[0].router.weight.grad).all()
        assert router.cells[1].router.weight.grad is not None
        assert router.cells[1].router.weight.grad.abs().sum() > 0

    for router in model.routing_modules():
        router.reset_stats()
        router.collect_stats = True
    model.eval()
    with torch.no_grad():
        model(tokens)
    for router in model.routing_modules():
        report = router.stats()
        assert report['mode'] == 'soft'
        assert report['validation_tokens'] == tokens.numel()
        assert report['executed_cell_visits_per_token'] == 1 + config['experts'] * (config['max_hops'] - 1)
        assert report['executed_router_visits_per_token'] == 1 + config['experts'] * (config['max_hops'] - 2)
        assert report['unique_router_weights_per_token'] == config['experts']
        assert report['expected_path_length'] <= config['max_hops']
        assert sum(report['path_length_distribution']) == pytest.approx(1.0)


def test_hard_top1_stops_early_and_executes_only_selected_cell(config):
    router = CellInternalRouter(config).eval()
    for cell in router.cells:
        with torch.no_grad():
            cell.router.weight.zero_()
            cell.router.weight[-1].fill_(1.0)  # deterministic STOP transition
    calls = [0] * config['experts']
    hooks = [cell.compute.register_forward_hook(
        lambda _module, _inputs, _output, index=index: calls.__setitem__(index, calls[index] + 1))
        for index, cell in enumerate(router.cells)]
    router.mode = 'hard'
    router.collect_stats = True
    hidden = torch.ones(2, 3, config['width'])
    with torch.no_grad():
        output, _ = router(hidden)
    for hook in hooks:
        hook.remove()

    report = router.stats()
    assert calls == [1, 0, 0]  # one batched module call for all six selected tokens
    assert report['executed_cell_visits_per_token'] == 1
    assert report['expected_path_length'] == 1
    assert report['learned_stop_fraction'] == 1
    assert report['forced_stop_fraction'] == 0
    assert report['executed_router_visits_per_token'] == 1
    assert report['unique_router_weights_per_token'] == 1
    assert report['sample_paths'] == [[0], [0], [0], [0], [0], [0]]
    assert output.shape == hidden.shape


def test_forced_final_hop_skips_router_projection(config):
    router = CellInternalRouter(config).eval()
    calls = [0] * config['experts']
    hooks = []
    for index, cell in enumerate(router.cells):
        with torch.no_grad():
            cell.router.weight.zero_()  # tied logits choose destination 0, never STOP
        hooks.append(cell.router.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(index, calls[index] + 1)))
    router.mode = 'hard'
    router.collect_stats = True
    hidden = torch.ones(2, 3, config['width'])
    with torch.no_grad():
        router(hidden)
    for hook in hooks:
        hook.remove()

    report = router.stats()
    assert calls == [config['max_hops'] - 1, 0, 0]
    assert report['executed_cell_visits_per_token'] == config['max_hops']
    assert report['executed_router_visits_per_token'] == config['max_hops'] - 1
    assert report['forced_stop_fraction'] == 1


def test_top1_training_fails_closed(config):
    router = CellInternalRouter(config).train()
    router.mode = 'hard'
    with pytest.raises(RuntimeError, match='not differentiable training'):
        router(torch.randn(1, 2, config['width']))


def test_registered_arms_match_capacity_within_preregistered_tolerance():
    config_path = Path(__file__).resolve().parents[3] / 'shared/python/experiments/cell_internal_routing_002/configs/experiment.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    counts = validate_config(config)
    assert all(50_000_000 <= count <= 150_000_000 for count in counts)
    assert max(counts) / min(counts) - 1 <= 0.01


def test_experiment001_model_import_and_checkpoint_layout_remain_stable(config):
    old_config = {key: value for key, value in config.items() if key not in {'max_hops', 'path_samples'}}
    torch.manual_seed(42)
    legacy = Experiment001LanguageModel(old_config, 'global-router')
    torch.manual_seed(42)
    shared = LanguageModel(old_config, 'global-router')
    assert list(legacy.state_dict()) == list(shared.state_dict())
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, shared.state_dict()[name])
    assert len(legacy.routing_modules()) == old_config['layers']


def test_missing_seed_evidence_cannot_be_reported_as_a_result(tmp_path):
    report = seed_report(tmp_path)
    assert report['status'] == 'DEVELOPMENT_INCOMPLETE'
    assert report['scientific_decision'] is False

    matrix = matrix_report([tmp_path / str(seed) for seed in (42, 123, 456)])
    assert matrix['status'] == 'DEVELOPMENT_INCOMPLETE'
    assert matrix['scientific_decision'] is False
