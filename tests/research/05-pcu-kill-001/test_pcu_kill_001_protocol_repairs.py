"""Regression tests for the final PCU-KILL-001 protocol-wiring repairs.

These tests are deliberately low-cost and never use a formal seed.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from minicells.pcu_kill_001.artifacts import file_sha256, load_tensor_artifact, save_tensor_artifact
from minicells.pcu_kill_001.backends import make_toy_model
from minicells.pcu_kill_001.cellular import GraniteArchitectureInspector
from minicells.pcu_kill_001.composition import compose_cellular_experts
from minicells.pcu_kill_001.evaluation import evaluate_samples
from minicells.pcu_kill_001.lora import LoRAConfig, MatchedLoRAExperts
from minicells.pcu_kill_001.metrics import decide
from minicells.pcu_kill_001.model import cellularize_model
from minicells.pcu_kill_001.registry import bind_fork_artifact, fork_registry, make_foundation_registry, validate_fork_artifacts
from minicells.pcu_kill_001.synthetic import audit_dataset, generate_world
from minicells.pcu_kill_001.task import TailTrainingCache
from minicells.pcu_kill_001.task_training import train_cached_lora_branch


FORMAL_SEEDS = {26090511, 26090512, 26090513}


def test_training_prompts_never_contain_their_answer_identifier() -> None:
    world = generate_world(26090501, 16)
    assert all(sample.answer not in sample.prompt for sample in world.splits["A_train"])
    assert all(sample.answer not in sample.prompt for sample in world.splits["B_train"])
    audit = audit_dataset(world)
    assert audit.passed
    assert audit.checks["A_answer_absent_from_prompt"]
    assert audit.checks["B_answer_absent_from_prompt"]


def test_dataset_audit_rejects_target_leakage_in_training_prompt() -> None:
    world = generate_world(26090501, 8)
    original = world.splits["A_train"][0]
    world.splits["A_train"][0] = replace(original, prompt=f"{original.prompt} {original.answer}")
    audit = audit_dataset(world)
    assert not audit.passed
    assert not audit.checks["A_answer_absent_from_prompt"]


class _GeneratedTokenizer:
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [7]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        mapping = {20: "VAAAA", 21: "WAAAB", 22: "WAAAA"}
        return " ".join(mapping[value] for value in ids if value in mapping)


class _ScriptedModel:
    def __init__(self, generated: list[int], vocab: int = 32) -> None:
        self.generated = list(generated)
        self.calls = 0
        self.vocab = vocab

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        index = min(self.calls, len(self.generated) - 1)
        next_id = self.generated[index]
        self.calls += 1
        logits = torch.full((*input_ids.shape, self.vocab), -1000.0, device=input_ids.device)
        logits[:, -1, next_id] = 1000.0
        return type("Output", (), {"logits": logits})()


def _sample(answer: str, sample_id: str = "sample"):
    return type("Sample", (), {"sample_id": sample_id, "prompt": "prompt", "answer": answer})()


def test_multitoken_identifier_evaluation_rejects_wrong_identifier_after_shared_prefix() -> None:
    result = evaluate_samples(
        _ScriptedModel([20, 1]),
        _GeneratedTokenizer(),
        [_sample("VAAAB")],
        split="A_eval",
        device="cpu",
        max_new_tokens=2,
    )
    assert result.exact == 0.0


def test_composition_requires_both_relay_and_terminal() -> None:
    wrong_terminal = evaluate_samples(
        _ScriptedModel([20, 21, 1]),
        _GeneratedTokenizer(),
        [_sample("VAAAA WAAAA")],
        split="AB_eval",
        device="cpu",
        max_new_tokens=3,
    )
    assert wrong_terminal.relay_exact == 1.0
    assert wrong_terminal.terminal_exact == 0.0
    assert wrong_terminal.both_exact == 0.0

    correct = evaluate_samples(
        _ScriptedModel([20, 22, 1]),
        _GeneratedTokenizer(),
        [_sample("VAAAA WAAAA")],
        split="AB_eval",
        device="cpu",
        max_new_tokens=3,
    )
    assert correct.both_exact == 1.0


def test_training_execution_does_not_imply_capability_pass() -> None:
    decision = decide(
        {
            "g0_exact_embedding": 1.0,
            "context_oracle_accuracy": 1.0,
            "base_a": 0.0,
            "base_b": 0.0,
            "acc_a": 0.2,
            "acc_b": 0.3,
        },
        {"protocol": True},
    )
    assert decision.status == "LOCAL_CELL_MUTATION_UNSUPPORTED"


def test_context_oracle_floor_is_a_testbed_gate() -> None:
    decision = decide(
        {"g0_exact_embedding": 1.0, "context_oracle_accuracy": 0.5},
        {"protocol": True},
    )
    assert decision.status == "TESTBED_COMPOSITION_CAPACITY_INADEQUATE"


def _minimal_cache(split: str) -> TailTrainingCache:
    return TailTrainingCache(
        mlp_input=torch.zeros(1, 1, 4),
        pre_mlp_residual=torch.zeros(1, 1, 4),
        top_k_index=torch.zeros(1, 1, dtype=torch.long),
        top_k_weights=torch.ones(1, 1),
        input_ids=torch.zeros(1, 1, dtype=torch.long),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        labels=torch.full((1, 1), -100, dtype=torch.long),
        loss_mask=torch.zeros(1, 1, dtype=torch.bool),
        sample_ids=("one",),
        split=split,
        identity={},
    )


def test_lora_branch_cannot_consume_the_other_branch_split() -> None:
    with pytest.raises(ValueError, match="cannot consume cache split"):
        train_cached_lora_branch(
            None,
            None,
            _minimal_cache("B_train"),
            ("L0:E0:C0",),
            layer=0,
            branch="A",
            rank=1,
            config=type("Config", (), {})(),
        )


def test_exact_lora_runtime_uses_additive_weight_deltas() -> None:
    model = make_toy_model()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    cellular, _ = cellularize_model(model, inspector)
    parent = cellular.model.layers[0].block_sparse_moe.experts
    branch_a = MatchedLoRAExperts(parent, {0: [0]}, LoRAConfig(rank=2))
    branch_b = MatchedLoRAExperts(parent, {0: [0]}, LoRAConfig(rank=2))
    with torch.no_grad():
        for module, scale in ((branch_a, 0.10), (branch_b, -0.07)):
            cell = module.cells[0].cells[0]
            cell.gate_a.fill_(scale)
            cell.gate_b.fill_(scale * 0.5)
            cell.up_a.fill_(scale * 0.8)
            cell.up_b.fill_(scale * 0.4)
            cell.down_a.fill_(scale * 0.6)
            cell.down_b.fill_(scale * 0.3)

    branches = {
        "A": {index: expert for index, expert in enumerate(branch_a.cells)},
        "B": {index: expert for index, expert in enumerate(branch_b.cells)},
    }
    merged = compose_cellular_experts(parent, branches, ("A", "B"))
    assert type(merged).__name__ == "ExactMergedLoRAExperts"

    parent_cell = parent.cells[0].cells[0]
    a_cell = branch_a.cells[0].cells[0]
    b_cell = branch_b.cells[0].cells[0]
    x = torch.randn(5, inspector.hidden_size)
    gate_w = parent_cell.gate_weight + a_cell.effective_deltas()["gate"] + b_cell.effective_deltas()["gate"]
    up_w = parent_cell.up_weight + a_cell.effective_deltas()["up"] + b_cell.effective_deltas()["up"]
    down_w = parent_cell.down_weight + a_cell.effective_deltas()["down"] + b_cell.effective_deltas()["down"]
    expected = F.linear(
        F.silu(F.linear(x, gate_w, parent_cell.gate_bias)) * F.linear(x, up_w, parent_cell.up_bias),
        down_w,
    )
    torch.testing.assert_close(merged.cells[0].cells[0](x), expected, rtol=1e-5, atol=1e-6)


def test_bound_fork_weight_hash_is_revalidated_after_restart(tmp_path: Path) -> None:
    artifact, _ = save_tensor_artifact({"delta": torch.arange(4.0)}, tmp_path, "branch")
    digest = file_sha256(artifact)
    base = make_foundation_registry(
        layer=0,
        experts=1,
        cells_per_expert=1,
        cell_width=4,
        foundation_model="m",
        foundation_revision="r",
        foundation_hash="foundation",
        protocol_sha256="protocol",
    )
    branch = fork_registry(base, ["L0:E0:C0"], "A")
    branch = bind_fork_artifact(branch, "A", str(artifact), digest)
    validate_fork_artifacts(branch)
    record = branch.fork_records[0]
    assert record.weight_hash == digest
    assert record.provenance["weight_hash_kind"] == "artifact_sha256"

    artifact.write_bytes(artifact.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_fork_artifacts(branch)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_tensor_artifact(artifact, digest)


def test_formal_executor_has_no_k_or_lr_search() -> None:
    from minicells.pcu_kill_001 import experiment

    source = inspect.getsource(experiment.run_formal_execution)
    assert "allow_search=False" in source
    assert "ENGINEERING_LR_CANDIDATES" not in source
    assert "CAPACITY_LADDER" not in source


def test_scientifically_negative_formal_result_is_still_a_valid_run() -> None:
    script = Path(__file__).resolve().parents[3] / "scripts/research/run_pcu_kill_001.py"
    spec = importlib.util.spec_from_file_location("pcu_run_script_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._formal_run_is_valid({
        "status": "LOCAL_CELL_MUTATION_UNSUPPORTED",
        "scientific_evidence": True,
        "valid_formal_run": True,
    })
    assert not module._formal_run_is_valid({
        "status": "INVALID_FORMAL_RUN",
        "scientific_evidence": False,
        "valid_formal_run": False,
    })


def test_no_formal_seed_is_used_by_this_test_module() -> None:
    assert 26090501 not in FORMAL_SEEDS
