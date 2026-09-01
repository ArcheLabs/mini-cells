from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.real_representation_006_config import CoreValidation006Config
from minicells.real_representation_006_core import CellSystem, fit_functional_delta, protected_basis
from minicells.real_representation_006_experiment import prepare_seed, run_seed
from minicells.real_representation_006_io import FrozenSequence

PROTOCOL = (
    Path(__file__).resolve().parents[3]
    / "research"
    / "validations"
    / "core-006-real-representation-continual-plasticity"
    / "protocol.json"
)


def test_protocol_is_frozen_real_bridge() -> None:
    cfg = CoreValidation006Config.from_protocol(PROTOCOL)
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert raw["status_before_run"] == "PROTOCOL_FROZEN_UNRUN"
    assert cfg.model_id == "EleutherAI/pythia-160m"
    assert cfg.model_revision == "step143000"
    assert cfg.dataset_id == "DKYoon/SlimPajama-6B"
    assert cfg.formal_seeds == (80611, 80612, 80613)
    assert cfg.transactions == 56
    assert cfg.addresses == 32
    assert cfg.base_cells == 8


def test_safe_fit_is_orthogonal_to_protected_basis() -> None:
    g = torch.Generator().manual_seed(1)
    d = 8
    old = torch.randn(10, d, generator=g, dtype=torch.float64)
    cov = old.T @ old
    z = torch.randn(6, d, generator=g, dtype=torch.float64)
    desired = torch.randn(6, d, generator=g, dtype=torch.float64)
    fit = fit_functional_delta(
        z,
        desired,
        cov=cov,
        certificate_energy=0.9,
        ridge=1e-4,
        safe=True,
    )
    q = protected_basis(cov, energy=0.9)
    assert torch.linalg.norm(fit.delta @ q).item() < 1e-8


def test_mitosis_is_function_preserving_and_partitions_dependency() -> None:
    d = 6
    system = CellSystem.initialize(
        dim=d,
        num_addresses=4,
        base_cells=2,
        address_owner={0: 0, 1: 0, 2: 1, 3: 1},
        certificate_energy=0.95,
    )
    g = torch.Generator().manual_seed(2)
    system.cells[0].a.copy_(torch.randn(d, d, generator=g, dtype=torch.float64))
    z0 = torch.randn(5, d, generator=g, dtype=torch.float64)
    z1 = torch.randn(7, d, generator=g, dtype=torch.float64)
    system.register(0, z0)
    system.register(1, z1)
    parent_before = system.cells[0].a.clone()
    cov_before = system.cell_covariance(0).clone()

    split = system.split_address(1, transaction=3)
    child = split["child_id"]
    assert torch.equal(system.cells[0].a, parent_before)
    assert torch.equal(system.cells[child].a, parent_before)
    assert system.address_owner[1] == child
    assert torch.allclose(cov_before, system.cell_covariance(0) + system.cell_covariance(child))
    assert torch.allclose(z1 @ parent_before.T, z1 @ system.cells[child].a.T)


def _mock_cfg() -> CoreValidation006Config:
    return CoreValidation006Config(
        model_id="mock",
        model_revision="mock",
        dataset_id="mock",
        dataset_revision="mock",
        dataset_split="train",
        sources=("A", "B"),
        sequence_length=9,
        router_bootstrap_sequences_per_source=4,
        train_sequences_per_source=4,
        eval_sequences_per_source=2,
        sequences_per_transaction=2,
        addresses=4,
        base_cells=2,
        kmeans_iterations=4,
        cell_dim=4,
        certificate_energy=0.9,
        ridge=1e-3,
        functional_step=0.01,
        maximum_delta_norm=0.2,
        split_conflict_threshold=0.2,
        maximum_splits_per_transaction=1,
        replay_buffer_sequences=4,
        replay_sequences_per_transaction=2,
        retention_checkpoint_every_transactions=1,
        minimum_midstream_reuse_ratio=0.0,
        maximum_midstream_energy_rank_fraction=1.0,
        maximum_registered_regression_ratio_vs_unsafe=100.0,
        minimum_gain_ratio_vs_replay=0.0,
        minimum_split_conflict_reduction=0.0,
        maximum_spawned_fraction_of_addresses=1.0,
        minimum_child_reuse_transactions=0,
        formal_seeds=(1,),
        smoke_seed=1,
    )


def _mock_sequences(cfg: CoreValidation006Config) -> tuple[list[FrozenSequence], torch.Tensor]:
    g = torch.Generator().manual_seed(4)
    seqs = []
    vocab = 16
    hidden_dim = 8
    for source in cfg.sources:
        for partition, count in (
            ("router", cfg.router_bootstrap_sequences_per_source),
            ("train", cfg.train_sequences_per_source),
            ("eval", cfg.eval_sequences_per_source),
        ):
            for index in range(count):
                seqs.append(
                    FrozenSequence(
                        partition=partition,
                        source=source,
                        hidden=torch.randn(
                            cfg.sequence_length - 1, hidden_dim, generator=g
                        ).to(torch.float16),
                        labels=torch.randint(
                            0, vocab, (cfg.sequence_length - 1,), generator=g
                        ),
                        document_sha256=f"{source}-{partition}-{index}",
                        token_sha256=f"tok-{source}-{partition}-{index}",
                    )
                )
    head = torch.randn(vocab, hidden_dim, generator=g) * 0.1
    return seqs, head


def test_real_bridge_state_machine_executes_without_old_access_for_candidate() -> None:
    cfg = _mock_cfg()
    seqs, head = _mock_sequences(cfg)
    run = run_seed(
        seqs,
        cfg,
        seed=1,
        lm_head_weight=head,
        device=torch.device("cpu"),
    )
    growth = run["gate_summary"]["variant_summaries"]["certificate_mitosis"]
    replay = run["gate_summary"]["variant_summaries"]["replay"]
    assert growth["learner_old_sample_accesses"] == 0
    assert growth["learner_old_label_accesses"] == 0
    assert replay["learner_old_sample_accesses"] > 0
    assert len(run["checkpoint_records"]) > 0
    assert len(run["causal_records"]) > 0


def test_router_is_frozen_and_seed_deterministic() -> None:
    cfg = _mock_cfg()
    seqs, _ = _mock_sequences(cfg)
    u1, c1, owners1, p1 = prepare_seed(seqs, cfg, seed=7)
    u2, c2, owners2, p2 = prepare_seed(seqs, cfg, seed=7)
    assert torch.equal(u1, u2)
    assert torch.equal(c1, c2)
    assert owners1 == owners2
    assert [x.address for x in p1] == [x.address for x in p2]
