from __future__ import annotations

import inspect

from minicells.constructive_clm_005 import (
    EndogenousCellModel,
    endogenous_control_smoke,
    learned_controllers,
)


def test_constructive_clm_005_controller_meta_smoke() -> None:
    result = endogenous_control_smoke(601)
    assert result["pass"] is True
    assert result["route_smoke_accuracy"] == 1.0
    diagnostics = result["controller_diagnostics"]
    assert diagnostics["router"]["heldout_meta_accuracy"] >= 0.98
    assert diagnostics["growth"]["heldout_meta_accuracy"] >= 0.98
    assert diagnostics["write"]["heldout_meta_accuracy"] >= 0.98
    assert diagnostics["meta_training_uses_formal_seed_data"] is False
    assert diagnostics["meta_training_uses_hidden_ids_as_targets"] is False
    assert len(result["controller_state_sha256"]) == 64


def test_constructive_clm_005_learner_api_has_no_hidden_ids() -> None:
    bundle = learned_controllers()
    assert bundle.diagnostics["meta_training_uses_hidden_ids_as_targets"] is False
    assert list(inspect.signature(EndogenousCellModel.observe_operator).parameters) == [
        "self",
        "route_context",
        "hidden",
        "residual",
    ]
    assert list(inspect.signature(EndogenousCellModel.growth_probability).parameters) == [
        "self",
        "route_context",
        "hidden",
        "target_residual",
    ]
    assert "hidden_id" not in inspect.getsource(EndogenousCellModel)
    assert "novelty_flag" not in inspect.getsource(EndogenousCellModel)
