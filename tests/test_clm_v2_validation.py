from __future__ import annotations

from pathlib import Path

import torch

from minicells.clm_v2_training import latest_stage_checkpoint, save_v2_checkpoint
from minicells.clm_v2_validation import (
    V2RoutingRecorder,
    make_v2_decision,
    replay_v2_masks,
    static_mask,
)
from minicells.language_models import TextNCALM
from minicells.textnca_to_clm_v2 import convert_textnca_to_clm_v2


def _model():
    source = TextNCALM(
        vocab_size=29, max_context=10, dim=16, heads=4, ffn_dim=32,
        windows=(2, 4, 8), iterations=(1, 1, 1),
    )
    model = convert_textnca_to_clm_v2(source)
    model.set_scaffold_alpha(0)
    return model


def test_v2_mask_record_and_replay_parity() -> None:
    model = _model()
    inputs = torch.randint(0, 29, (2, 8))
    with V2RoutingRecorder(model) as recorder:
        expected = model(inputs).logits
    with replay_v2_masks(model, recorder.masks):
        actual = model(inputs).logits
    torch.testing.assert_close(actual, expected)


def test_v2_static_mask_is_deterministic_topk() -> None:
    masks = [[torch.tensor([[[1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]]], dtype=torch.float32)]]
    first = static_mask(masks, 6)
    second = static_mask(masks, 6)
    assert int(first.sum()) == 6
    torch.testing.assert_close(first, second)


def test_stage_checkpoint_discovery_supports_resume(tmp_path: Path) -> None:
    save_v2_checkpoint(tmp_path / "r0-stage-1.pt", {"stage_index": 1})
    save_v2_checkpoint(tmp_path / "r0-stage-3.pt", {"stage_index": 3})
    assert latest_stage_checkpoint(tmp_path, 0) == tmp_path / "r0-stage-3.pt"


def test_v2_decision_requires_causal_controls_not_only_capacity() -> None:
    workers = [
        {"replicate": index, "status": "CLMV2_SCAFFOLD_HANDOFF_SIGNAL", "quality_safe_k": 5}
        for index in range(3)
    ]
    arms = []
    for replicate in range(3):
        for arm, nll in (("dense", 2.0), ("dynamic", 2.01), ("static", 2.02),
                         ("shuffled", 2.02)):
            arms.append({"replicate": replicate, "arm": arm, "nll": nll,
                         "ppl": 7.46, "sample_variation": 0.06, "receptor_ratio": 0.02})
    assert make_v2_decision(workers, arms, teacher_nll=2.0)["diagnosis"] == (
        "CLMV2_PROGRAM_CONDITIONALITY_SIGNAL"
    )
    for row in arms:
        if row["arm"] in ("static", "shuffled"):
            row["nll"] = 2.011
    assert make_v2_decision(workers, arms, teacher_nll=2.0)["diagnosis"] == (
        "CLMV2_CONDITIONAL_CAPACITY_WITHOUT_CAUSAL_ROUTING"
    )
