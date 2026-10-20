import inspect
import numpy as np
import pytest
import torch
from minicells_r3c_moe_bridge.common import configuration
from minicells_r3c_moe_bridge.model import BridgeModel
from minicells_r3c_moe_bridge.evaluation import cell_values, permutation_intervention
from minicells_r3c_moe_bridge.groups import freeze_groups, apply_groups, gate_utility
from minicells_r3c_moe_bridge.statistics import (
    story_interval,
    paired_seed_comparison,
    seed_variation,
    crossfit_oracle,
    conditional_groups,
    align_cells,
    temporal_summary,
)
from minicells_r3c_moe_bridge.release import require_development, choose_balance
from minicells_r3c_moe_bridge.runner import gpu_smoke
from minicells_r3c_moe_bridge.data import fixed_windows


@pytest.mark.parametrize("suffix", ["FIXED_SUFFIX", "REPLANNED_SUFFIX"])
@pytest.mark.parametrize("arm", list("ABCDE"))
def test_counterfactuals_target_free_inference(config, arm, suffix):
    m = BridgeModel(config, arm, 123)
    x = torch.tensor([[2, 3, 4, 5]])
    y = torch.tensor([[3, 4, 5, 2]])
    report = cell_values(m, x, y, stage=1, seed=11, suffix=suffix)
    assert report["budget"] == "4x2_cells_plus_one_context"
    assert report["values"].shape == (x.numel(), config["routing"]["cells"])
    assert "y" not in inspect.signature(m.forward).parameters
    assert "targets" not in inspect.signature(permutation_intervention([1, 0, 3, 2])).parameters


def test_calibration_discovery_independent_development():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    rule = freeze_groups(x, ["a", "b", "c", "d"], role="calibration", seed=1)
    groups = apply_groups(x, rule, ["e", "f", "g", "h"], role="development")
    assert set(groups) == {0, 1} and not rule["uses_targets"]
    with pytest.raises(ValueError, match="overlap"):
        apply_groups(x, rule, ["a", "f", "g", "h"], role="development")
    with pytest.raises(ValueError):
        freeze_groups(x, ["a", "b", "c", "d"], role="development", seed=1)


def test_independent_oracle_and_group_statistics():
    a = np.array([[1.0, 3.0, 4.0], [3.0, 1.0, 4.0]])
    with pytest.raises(ValueError):
        crossfit_oracle(a, a, selection_stream="same", validation_stream="same")
    result = crossfit_oracle(
        a, a + 0.1, selection_stream="selection", validation_stream="validation"
    )
    assert result["oracle_potential_gap"] > 0 and not result["inference_uses_oracle"]
    report = conditional_groups(
        a,
        [0, 1],
        a + 0.1,
        [0, 1],
        selection_role="calibration",
        validation_role="development",
        selection_ids=["a", "b"],
        validation_ids=["c", "d"],
    )
    assert report["calibration_selected_cells"] == {"0": 0, "1": 1}
    assert align_cells(a, a[::-1]) == [1, 0]
    utility = gate_utility(
        [[0, 1], [1, 0]],
        a,
        a + 0.1,
        selection_stream="a",
        validation_stream="b",
        identifiable_margin=0.01,
    )
    assert utility["ranking_agreement"] == 1 and utility["regret"] == 0


def test_story_and_seed_units():
    arms = {
        a: {str(s): {"mean_nll": float(s) + (0 if a == "C" else 1)} for s in range(4)}
        for a in "ABCDE"
    }
    report = paired_seed_comparison(arms, 123, 100)
    assert report["comparisons"]["C-A"]["stories"] == 4
    assert all("holm_exploratory_p" in report["comparisons"][key] for key in ("C-A", "C-B", "C-D"))
    with pytest.raises(ValueError):
        seed_variation([report, report])
    assert seed_variation([report])["independent_training_seeds"] == 1
    assert story_interval([1], 123)["ci95"] is None
    t = np.ones((3, 2, 4))
    assert "CORRELATION_ONLY" in temporal_summary(t, t, t)["claim"]


def test_gpu_denied_before_any_device_or_asset_access(monkeypatch, tmp_path):
    def forbidden(*a, **k):
        raise AssertionError("real work reached")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(RuntimeError, match="NOT_AUTHORIZED"):
        gpu_smoke(configuration(), tmp_path, None, None)
    with pytest.raises(RuntimeError, match="NOT_RELEASED"):
        require_development(configuration(), None)
    with pytest.raises(RuntimeError, match="NOT_FROZEN"):
        choose_balance(configuration(), {})


def test_balance_selection_and_no_development(config):
    config["balance"]["minimum_cell_share"] = 0.1
    rows = {
        v: {"role": "calibration", "cell_shares": np.full((2, 4, 4), 0.25).tolist()}
        for v in config["balance"]["candidates"]
    }
    assert choose_balance(config, rows)["coefficient"] == 0
    rows[0.0]["role"] = "development"
    with pytest.raises(ValueError, match="Calibration"):
        choose_balance(config, rows)


def test_confirmation_evaluation_rejected(config):
    with pytest.raises(ValueError):
        list(fixed_windows({}, config, "confirmation", "cpu"))
