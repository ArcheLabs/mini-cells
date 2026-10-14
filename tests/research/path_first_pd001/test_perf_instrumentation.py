import copy
import pytest
import torch
from pd001.perf_audit import flags
from pd001.performance import settings, retrain_review
from pd001.models import PDCLM
from pd001.accounting import touched_weights, Counter
from performance_helpers import original, fixture


def test_ablations_one_factor_at_a_time():
    keys = ["capture", "diagnostics", "counter", "dispatch", "probe_interval"]
    for i, key in enumerate(keys):
        a, b = flags("B" + str(i)), flags("B" + str(i + 1))
        assert [k for k in a if a[k] != b[k]] == [key]


def test_cached_accounting_matches_original(proposal):
    _, _, m, _ = fixture(proposal)
    model = PDCLM(m, "P50-F16", 17).train()
    x = torch.tensor([[1, 2]])
    with Counter(model) as ledger:
        out = model(x, replay=torch.tensor([[[0, 0, 2, 1, 0], [1, 1, 2, 0, 1]]]))
    assert touched_weights(model, out) == original("accounting").touched_weights(model, out)
    assert ledger.report(2)["trace_verification"]["verified"]


def test_profile_validation_and_retrain_fail_closed(proposal):
    c = copy.deepcopy(proposal)
    assert settings(c)["training_profile"] == "light"
    c["performance"]["full_path_capture_each_step"] = True
    with pytest.raises(ValueError):
        settings(c)
    c["performance"].update(training_profile="instrumented", full_path_capture_each_step=False)
    with pytest.raises(RuntimeError, match="at most 100"):
        settings(c)
    with pytest.raises(RuntimeError, match="review pending"):
        retrain_review(proposal, "a" * 40)
