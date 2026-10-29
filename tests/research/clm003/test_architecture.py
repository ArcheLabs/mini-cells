"""Bounded Stage1 fixture; no scientific seeds, datasets or budgets registered."""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import screening as sc
import torch
from minicells_r3e.common import fingerprint
from tokenizers import Tokenizer

SPEC = importlib.util.spec_from_file_location(
    "architecture_benchmark", sc.BASE / "architecture-benchmark/implementation/benchmark.py"
)
ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ab)


class Stream:
    def __init__(self, rows):
        self.rows, self.step = rows, 0

    def next(self):
        self.step += 1
        return [dict(r, task=("Story" if i else "Arithmetic")) for i, r in enumerate(self.rows)]

    def state(self):
        return {"step": self.step, "data_hash": fingerprint(self.rows)}

    def restore(self, state):
        assert state["data_hash"] == fingerprint(self.rows)
        self.step = state["step"]


def tokenizer():
    path = (
        sc.BASE.parent
        / "clm-001-qfirst-integrated/r3m/registration/artifact-recovery-001/independent-review-002/tokenizer.json"
    )
    assert (
        ab.digest(path.read_bytes())
        == "612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff"
    )
    return Tokenizer.from_file(str(path))


def setup():
    torch.set_num_threads(1)
    rows, identity = ab.tiny_data(tokenizer())
    store = ab.RoleStore(rows)
    models, runtime, calibration = ab.make_models(
        73001, [r["x"] for r in store.read("calibration")]
    )
    return store, identity, models, runtime, calibration


def test_masks_and_confirmation_gate():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    y = torch.zeros(2, 3, dtype=torch.long)
    valid = torch.tensor([[1, 0, 1], [0, 1, 0]])
    value = ab.sequence_loss(logits, y, valid)
    (value["loss_sum"].sum() / value["valid_count"].sum()).backward()
    assert (logits.grad[valid == 0] == 0).all()
    assert value["valid_count"].tolist() == [2, 1]
    with pytest.raises(ValueError):
        ab.sequence_loss(logits, y, torch.zeros_like(valid))

    class Forbidden:
        def __call__(self, x):
            raise AssertionError("MODEL_MUST_NOT_BE_CALLED")

    with pytest.raises(RuntimeError, match="CONFIRMATION_SEALED"):
        ab.evaluate(Forbidden(), ab.RoleStore({}), "confirmation")


def test_one_bounded_cpu_trial_and_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    store, data_identity, models, runtime, calibration = setup()
    init = ab.initialization_report(models)
    assert init["passed"]
    assert all(
        torch.isfinite(models["CLM"].state_dict()[k])
        for k in ["route_switch_margin", "route_numeric_tolerance"]
    )
    rng_before = torch.get_rng_state().clone()
    repeat, _, _ = ab.make_models(73001, [r["x"] for r in store.read("calibration")])
    assert all(
        fingerprint(m.state_dict()) == fingerprint(repeat[a].state_dict())
        for a, m in models.items()
    )
    assert torch.equal(rng_before, torch.get_rng_state())
    causal = {}
    for arm, model in models.items():
        x = store.read("train")[0]["x"]
        changed = x.clone()
        changed[:, -1] = (changed[:, -1] + 1) % 4096
        with torch.no_grad():
            error = (model(x).logits[:, :-1] - model(changed).logits[:, :-1]).abs().max().item()
        assert error < 1e-6
        causal[arm] = error
    native = models["CLM"]
    engine = ab.make_clm_trainer(native, runtime, store.read("calibration"), tmp_path)
    assert not engine.cell_optimizer.state
    stream = Stream(store.read("train"))
    latest = tmp_path / "latest.pt"
    engine.update(stream, latest, [])
    engine.update(stream, latest, [])
    assert engine.step == 2 and engine.tokens == 32
    assert engine.proposal_calls == 1
    saved_optimizer = fingerprint(engine.cell_optimizer.state_dict())
    saved = fingerprint(engine.model.state_dict())
    with torch.no_grad():
        next(engine.model.cells.parameters()).add_(0.5)
    engine.restore(latest, stream)
    assert saved == fingerprint(engine.model.state_dict())
    assert saved_optimizer == fingerprint(engine.cell_optimizer.state_dict())
    decision_path = next((tmp_path / "proposals").glob("*/ADMISSION_DECISION.json"))
    candidate_path = tmp_path / "private/proposals/2/CANDIDATE.json"
    admission = json.loads(decision_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    assert len(admission["attempts"]) == 22
    assert candidate["actual_optimizer_calls"] == 1
    costs = {"CLM": ab.clm_cost(engine, candidate, admission)}
    costs["CLM"]["total_optimizer_calls_main_plus_candidate"] = 3
    costs["CLM"]["candidate_evaluations"] = len(admission["attempts"])
    costs["CLM"]["candidate_receipt"] = candidate

    curves = {"CLM": engine.curve}
    # Same rows/order/token budget; fresh independent AdamW for each reference.
    for arm in ["MoE", "Dense-AllCell-Mean"]:
        model = models[arm]
        trainer = ab.EngineeringReferenceTrainer(model, runtime)
        assert not trainer.optimizer.state
        for _ in range(2):
            trainer.update(store.read("train"))
        checkpoint = trainer.state()
        with torch.no_grad():
            next(model.cells.parameters()).add_(0.5)
        trainer.optimizer.param_groups[0]["lr"] = 9.0
        trainer.restore(checkpoint)
        assert fingerprint(trainer.state()) == fingerprint(checkpoint)
        assert fingerprint(model.state_dict()) == fingerprint(checkpoint["model"])
        with pytest.raises(RuntimeError, match="BUDGET"):
            trainer.update(store.read("train"))
        assert [r["input_sha256"] for r in trainer.curve] == [
            r["input_sha256"] for r in engine.curve
        ]
        curves[arm] = trainer.curve
        costs[arm] = trainer.cost()
    metrics = {a: ab.profile(m, store.read("development")[0]) for a, m in models.items()}
    assert metrics["Dense-AllCell-Mean"]["cell_calls_per_target"] == 48
    assert metrics["MoE"]["cell_calls_per_target"] == 16
    assert metrics["CLM"]["cell_calls_per_target"] == 8
    for arm, model in models.items():
        body = model if arm == "CLM" else model.native
        assert fingerprint(body.cells.state_dict()) != init["cell_hashes"][arm]
        assert (
            fingerprint(
                {
                    n: p
                    for n, p in body.state_dict().items()
                    if not n.startswith(("cells.", "connection.", "gate."))
                }
            )
            == init["public_hashes"][arm]
        )
    evaluations = {a: ab.evaluate(m, store, "development") for a, m in models.items()}
    with pytest.raises(RuntimeError, match="CONFIRMATION_SEALED"):
        ab.evaluate(native, store, "confirmation")
    assert "confirmation" not in store.accesses
    out = os.environ.get("ARCHITECTURE_STAGE1_REPORT")
    if out:
        dest = Path(out)
        dest.mkdir(parents=True, exist_ok=True)
        reports = {
            "ARCHITECTURE_CONFIGS": {
                "native_model": native.c["model"],
                "native_routing": native.c["routing"],
                "CLM_training": engine.c["training"],
                "CLM_connection": engine.fast["connection"],
                "common_optimizer": runtime["engine"]["training"],
                "reference_definitions": ab.manifest()["MoE"],
                "scope": "ENGINEERING_ONLY",
            },
            "ADMISSION_DECISION": admission,
            "INITIALIZATION_MANIFEST": init,
            "DATA_MANIFEST": data_identity,
            "CALIBRATION": calibration,
            "METRICS": metrics,
            "COST_LEDGER": costs,
            "ENGINEERING_DEVELOPMENT_NLL": evaluations,
            "TRAINING_CURVES": curves,
            "ACCESS_GATE": {
                "confirmation_model_reads": 0,
                "causality_max_errors": causal,
                "accessed_roles": sorted(set(store.accesses)),
            },
        }
        for name, value in reports.items():
            (dest / (name + ".json")).write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n"
            )
    for h in engine._work_handles:
        h.remove()
