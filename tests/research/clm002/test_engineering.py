import numpy as np
import pytest
import torch
from minicells_r3e.common import fingerprint
from minicells_clm002.native import build, masks, classify, treatment, public_fingerprint
from minicells_clm002.engine import Engine
from minicells_clm002.data import PairedStream, unit_pools, allocation_audit
from minicells_clm002.measurement import (
    replay,
    nll,
    contributions,
    specialization,
    gradient_exposure,
    js_divergence,
    route_switch,
    arithmetic_generation,
)


def setup(config, ident, tmp_path, arm="B11-H"):
    model, r = build(config, 7001, True)
    base = Engine(model, r, "BASE", ident, tmp_path / "base.jsonl", unit=True).state()
    model, r = build(config, 7001, True)
    engine = Engine(model, r, arm, ident, tmp_path / (arm + ".jsonl"), base, unit=True)
    pools = unit_pools(config)
    stream = PairedStream(pools, 7001, "H", r["training"]["tokens_per_step"])
    return engine, stream, base, pools


def test_native_class_and_real_budget(config):
    model, r = build(config, 7001, False)
    assert model.__class__.__module__ == "minicells_r3h.routing"
    assert len(model.cells) == 6 and model.steps == 8
    assert r["training"]["steps"] == 500 and r["training"]["tokens_per_step"] == 512
    assert model.embedding.weight.shape == (4096, 128)


@pytest.mark.parametrize("arm", ["A-S", "A-H", "B00-S", "B10-H", "B01-S", "B11-H"])
def test_parameter_boundaries(config, arm):
    model, _ = build(config, 7001, True)
    flags = masks(model, arm)
    cells, connection = treatment(arm)
    assert all(
        value == dict(cells=cells, connection=connection, public=False)[classify(name)]
        for name, value in flags.items()
    )
    assert not model.embedding.weight.requires_grad and not model.final.weight.requires_grad


def test_fixed_path_residual_parity_and_unused_cell(config):
    model, _ = build(config, 7001, True)
    x = unit_pools(config)["Story"][0]["x"]
    before = fingerprint(model.state_dict())
    full = model(x)
    assert int(full.operator_rows.sum()) == 8 * x.numel()
    assert torch.allclose(full.logits, replay(model, x, full.actions).logits, rtol=1e-5, atol=1e-7)
    paths = torch.zeros_like(full.actions)
    baseline = replay(model, x, paths)
    unused = replay(model, x, paths, 5)
    assert torch.equal(baseline.logits, unused.logits)
    ablated = replay(model, x, paths, 0)
    h, _ = model.initial(x)
    assert torch.allclose(ablated.logits, model.head(h.reshape(1, x.shape[1], -1)))
    assert torch.equal(ablated.actions, paths) and fingerprint(model.state_dict()) == before
    with pytest.raises(ValueError):
        replay(model, x, paths[:, :2])


def test_target_weighting_and_padding():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    labels = torch.tensor([[0, 0]])
    one = nll(logits, labels, torch.tensor([[1, 0]]))
    assert float(one) == pytest.approx(
        float(torch.nn.functional.cross_entropy(logits[:, 0], labels[:, 0]))
    )
    with pytest.raises(ValueError):
        nll(logits, labels, torch.zeros_like(labels))


def test_pairing_effective_token_fairness_restore(config):
    pools = unit_pools(config)
    a = PairedStream(pools, 88, "H", 8)
    b = PairedStream(pools, 88, "H", 8)
    s = PairedStream(pools, 88, "S", 8)
    for _ in range(3):
        rows = a.next()
        assert fingerprint(rows) == fingerprint(b.next())
        assert sum(int(r["valid"].sum()) for r in rows if r["task"] == "Story") == 4
        assert sum(int(r["valid"].sum()) for r in rows if r["task"] == "Arithmetic") == 4
        assert all(r["task"] == "Story" for r in s.next())
    saved = a.state()
    expected = fingerprint(a.next())
    a.restore(saved)
    assert fingerprint(a.next()) == expected


@pytest.mark.parametrize("arm", ["B00-H", "B10-H", "B01-H", "B11-H"])
def test_actual_native_updates_and_restore(config, ident, tmp_path, arm):
    tr, stream, base, pools = setup(config, ident, tmp_path, arm)
    initial = public_fingerprint(tr.model)
    path = tmp_path / "state.pt"
    tr.update(stream, path, [pools["Story"][0]["x"]])
    tr.save(tmp_path / "boundary.pt", stream)
    first = fingerprint((tr.state(), stream.state()))
    tr.update(stream, path, [pools["Story"][0]["x"]])
    expected = fingerprint((tr.state(), stream.state()))
    tr.restore(tmp_path / "boundary.pt", stream)
    assert fingerprint((tr.state(), stream.state())) == first
    tr.update(stream, path, [pools["Story"][0]["x"]])
    assert fingerprint((tr.state(), stream.state())) == expected
    assert public_fingerprint(tr.model) == initial
    for group, active in zip(("cells", "connection"), treatment(arm)):
        assert any(row["parameter_delta"][group] > 0 for row in tr.curve) == active
    assert tr.tokens == 16 and tr.journal.summary()["pending_attempts"] == 0
    if treatment(arm)[1]:
        assert tr.journal.summary()["executed_operator_work"]["cell_rows"] > 128


def test_forbidden_public_mutation_stops_before_update(config, ident, tmp_path):
    tr, stream, _, pools = setup(config, ident, tmp_path)
    with torch.no_grad():
        tr.model.embedding.weight.add_(1)
    with pytest.raises(RuntimeError, match="BEFORE_UPDATE"):
        tr.update(stream, tmp_path / "state.pt", [pools["Story"][0]["x"]])
    assert tr.journal.summary()["completed_optimizer_calls"] == 0


def test_failed_attempt_does_not_erase_real_work(config, ident, tmp_path):
    tr, stream, _, pools = setup(config, ident, tmp_path, "B10-H")
    path = tmp_path / "state.pt"
    tr.save(path, stream)

    def fail(_):
        raise RuntimeError("INJECTED")

    with pytest.raises(RuntimeError, match="INJECTED"):
        tr.update(stream, path, [pools["Story"][0]["x"]], failure_hook=fail)
    tr.restore(path, stream)
    tr.update(stream, path, [pools["Story"][0]["x"]])
    assert tr.step == 1 and tr.journal.summary()["completed_optimizer_calls"] == 2


def test_measurements_are_readonly_and_task_labels_not_router_inputs(config):
    model, _ = build(config, 7001, True)
    masks(model, "B00-H")
    pools = unit_pools(config)
    rows = [dict(pools[t][0], task=t) for t in ("Story", "Arithmetic")]
    before = fingerprint(model.state_dict())
    flags = [p.requires_grad for p in model.parameters()]
    matrix = contributions(model, rows)
    probe = gradient_exposure(model, rows)
    assert set(matrix) == set(probe) == {"Story", "Arithmetic"}
    assert matrix["Story"]["tokens"] == 4 and len(matrix["Story"]["D"]) == 6
    assert fingerprint(model.state_dict()) == before and flags == [
        p.requires_grad for p in model.parameters()
    ]
    assert all(p.grad is None for p in model.parameters())
    x = pools["Story"][0]["x"]
    paths = model(x).actions
    assert route_switch(paths, paths, "probe", "probe")["path_change"] == 0
    with pytest.raises(ValueError):
        route_switch(paths, paths, "old", "new")


def test_bidirectional_specialization_requires_exposure():
    tasks = dict(
        Story=dict(D=[0.04, 0, 9.0], call_share=[0.5, 0.5, 0]),
        Arithmetic=dict(D=[0, 0.05, 0], call_share=[0.5, 0.5, 0]),
    )
    result = specialization(tasks)
    assert result["S"] == pytest.approx(0.04) and result["story_preferring_exposed_cells"] == [0]
    tasks["Story"]["call_share"][0] = 0
    assert specialization(tasks)["S"] == 0
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(np.log(2))


def test_arithmetic_generation_stops_at_context_limit(config):
    model, _ = build(config, 7001, True)
    result = arithmetic_generation(
        model, [dict(prompt_tokens=[1], answer_tokens=[2])], eos_id=0, max_new_tokens=2
    )
    assert len(result["records"]) == 1
    with pytest.raises(ValueError, match="CONTEXT_BUDGET"):
        arithmetic_generation(model, [dict(prompt_tokens=[1, 2, 3], answer_tokens=[])], 0, 2)


def test_content_and_token_conflicts_are_not_hidden():
    rows = [
        dict(
            role="train",
            task="Story",
            text="Ｆoo   bar",
            tokens=[1, 2, 3, 4],
            tokenizer_sha256="same",
        ),
        dict(
            role="dev", task="Story", text="foo bar", tokens=[1, 2, 3, 4], tokenizer_sha256="same"
        ),
    ]
    proof = allocation_audit(rows)
    assert proof["status"] == "CONFLICT"
    assert (
        "CROSS_ROLE_NORMALIZED_TEXT" in proof["errors"]
        and "EXACT_TOKEN_WINDOW_OVERLAP" in proof["errors"]
    )
    assert not proof["global_scientific_isolation_pass"]


def test_arithmetic_group_boundaries_and_unknown_coverage():
    rows = [
        dict(
            role="train",
            task="Arithmetic",
            text="a b c d e",
            tokens=[1, 2, 3, 4],
            tokenizer_sha256="t",
            template_family="a",
            operator_combination="+",
            numeric_range=[0, 10],
        ),
        dict(
            role="dev",
            task="Arithmetic",
            text="u v w x y",
            tokens=[5, 6, 7, 8],
            tokenizer_sha256="t",
            template_family="b",
            operator_combination="*",
            numeric_range=[10, 20],
        ),
    ]
    assert allocation_audit(rows)["status"] == "UNKNOWN_HISTORICAL_COVERAGE"
    rows[1]["template_family"] = "a"
    assert "ARITHMETIC_TEMPLATE_FAMILY_OVERLAP" in allocation_audit(rows)["errors"]


def test_parent_hash_rejected_before_pickle(config, tmp_path, monkeypatch):
    from minicells_clm002.audit import inspect_parent

    path = tmp_path / "not-a-parent.pt"
    path.write_bytes(b"NOT_TRUSTED")

    def forbidden(*args, **kwargs):
        raise AssertionError("untrusted deserialization")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(RuntimeError, match="BEFORE_DESERIALIZATION"):
        inspect_parent(config, path, tmp_path)


def test_stage_a_preserves_native_frozen_connection_and_dynamic_decisions(config, monkeypatch):
    assert treatment("A-S") == treatment("A-H") == (True, False)
    model, _ = build(config, 7001, True)
    masks(model, "B10-H")
    original = fingerprint(model.connection.state_dict())
    monkeypatch.setattr(model.gate, "forward", lambda h, t: h[:, :6])
    a = torch.zeros(1, 8)
    a[0, 0] = 1
    b = torch.zeros(1, 8)
    b[0, 1] = 1
    previous = torch.tensor([6])
    assert int(model.select(a, 0, previous)[0]) != int(model.select(b, 0, previous)[0])
    assert fingerprint(model.connection.state_dict()) == original
    assert not any(p.requires_grad for p in model.connection.parameters())
