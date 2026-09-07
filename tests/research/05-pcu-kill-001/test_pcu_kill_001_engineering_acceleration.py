"""Regression coverage for the resource-only PCU engineering accelerations."""

from __future__ import annotations

import inspect

import torch
from torch import nn

from minicells.pcu_kill_001.evaluation import evaluate_samples
from minicells.pcu_kill_001.overlay import ExpertsOverlayModel
from minicells.pcu_kill_001 import pipeline_guard


class _BatchTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [2, 3] if add_special_tokens else [3]

    def __call__(self, texts, return_tensors: str, padding: bool, truncation: bool):
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is False
        assert self.padding_side == "left"
        rows = [[2, 3] for _ in texts]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 2), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        mapping = {5: "VAAAA", 6: "WAAAA"}
        return " ".join(mapping[value] for value in ids if value in mapping)


class _GenerateModel:
    def __init__(self) -> None:
        self.calls = 0
        self.max_batch = 0
        self.use_cache = None
        self.do_sample = None

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        self.calls += 1
        self.max_batch = max(self.max_batch, int(input_ids.shape[0]))
        self.use_cache = kwargs.get("use_cache")
        self.do_sample = kwargs.get("do_sample")
        suffix = torch.full((input_ids.shape[0], 1), 5, dtype=torch.long, device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


class _Sample:
    def __init__(self, index: int) -> None:
        self.sample_id = f"sample-{index}"
        self.prompt = f"prompt {index}"
        self.answer = "VAAAA"


def test_evaluation_batches_hf_generate_and_enables_kv_cache() -> None:
    tokenizer = _BatchTokenizer()
    model = _GenerateModel()
    result = evaluate_samples(
        model,
        tokenizer,
        [_Sample(index) for index in range(32)],
        split="A_eval",
        device="cpu",
        max_new_tokens=4,
        batch_size=16,
    )
    assert result.exact == 1.0
    assert model.calls == 2
    assert model.max_batch == 16
    assert model.use_cache is True
    assert model.do_sample is False
    assert tokenizer.padding_side == "right"


class _Experts(nn.Module):
    def __init__(self, marker: int) -> None:
        super().__init__()
        self.marker = marker


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _Experts(1)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_sparse_moe = _Block()


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


class _OverlayGenerateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.seen_marker = None

    def generate(self, input_ids: torch.Tensor, **kwargs):
        self.seen_marker = self.model.layers[0].block_sparse_moe.experts.marker
        return input_ids


def test_overlay_generate_keeps_branch_experts_for_whole_generate_call_and_restores() -> None:
    model = _OverlayGenerateModel()
    resident = model.model.layers[0].block_sparse_moe.experts
    branch = _Experts(7)
    view = ExpertsOverlayModel(model, "model.layers.0.block_sparse_moe", branch)
    view.generate(input_ids=torch.tensor([[1, 2]], dtype=torch.long))
    assert model.seen_marker == 7
    assert model.model.layers[0].block_sparse_moe.experts is resident


def test_pipeline_guard_installs_acceleration_only_for_engineering() -> None:
    source = inspect.getsource(pipeline_guard._run_with_optional_engineering_acceleration)
    assert 'kwargs.get("phase")' in source
    assert '!= "engineering"' in source
    assert "maybe_engineering_acceleration" in source
    assert "DualGPUContextOracle" in source


def test_generation_acceleration_has_fail_closed_equivalence_gate() -> None:
    source = inspect.getsource(pipeline_guard._verify_generation_acceleration)
    assert "fast_evaluate_samples" in source
    assert "greedy_generate" in source
    assert "GENERATION_ACCELERATION_SEMANTICS_INVALID" in source
    assert "exact_match" in source


def test_formal_scientific_worker_does_not_import_engineering_accelerator_directly() -> None:
    from minicells.pcu_kill_001 import resource_runtime

    source = inspect.getsource(resource_runtime.run_formal_execution)
    assert "engineering_accel" not in source
    assert "maybe_engineering_acceleration" not in source
