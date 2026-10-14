import torch
from performance_helpers import fixture, original


def test_actual_selected_operator_calls_repeated_cells_and_context(proposal):
    _, _, c, _ = fixture(proposal)
    x = torch.tensor([[1, 2]])
    replay = torch.tensor([[[0, 0, 0, 2, 0], [1, 1, 1, 2, 1]]])
    ledgers = []
    for module in (original("models"), __import__("pd001.models", fromlist=["PDCLM"])):
        model = module.PDCLM(c, "P50-F16", 17)
        calls = {}
        handles = []
        for name, operator in [
            ("cell0", model.cells[0]),
            ("cell1", model.cells[1]),
            ("ctx", model.context),
        ]:

            def hook(m, i, o, key=name):
                calls[key] = calls.get(key, 0) + i[0].shape[0]

            handles.append(operator.register_forward_hook(hook))
        out = model(x, replay=replay, capture=True)
        assert (out.lengths == 5).all() and out.forced_stop.all()
        repeated = [r for r in out.state_records if r["action"] == 0]
        assert not torch.equal(repeated[0]["before"], repeated[1]["before"])
        ledgers.append(calls)
        for handle in handles:
            handle.remove()
    assert ledgers == [{"cell0": 4, "cell1": 4, "ctx": 2}] * 2
