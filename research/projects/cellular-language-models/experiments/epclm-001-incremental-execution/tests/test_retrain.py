import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "implementation"))
from epclm001.retrain import configuration, manifest, run, validate_resume
from epclm001.retrain_guard import ResourceStop, TrainingGuard


def test_authorization_and_immutable_sources():
    c = configuration()
    assert list(c["models"]) == ["T1", "X3", "X4"] and c["seed"] == 91002
    assert c["model_seeds"] == {"T1": 91003, "X3": 91002, "X4": 91002}
    assert c["steps"] == 10000 and c["gpu_budget_seconds"] == 7200
    assert "M4" not in c["models"]


def test_manifest_refuses_config_drift(tmp_path):
    c = configuration()
    m = manifest(tmp_path, c)
    assert m["models"] == {"T1": "NOT_STARTED", "X3": "NOT_STARTED", "X4": "NOT_STARTED"}
    changed = dict(c, seed=91003)
    with pytest.raises(RuntimeError, match="CONFIG_MISMATCH"):
        manifest(tmp_path, changed)


def test_r0_blocked_never_starts_worker(tmp_path, monkeypatch):
    from epclm001 import retrain

    c = configuration()
    m = manifest(tmp_path, c)
    m["status"] = "R0_BLOCKED_NO_TRAINING"
    monkeypatch.setattr(retrain, "preflight", lambda *args: m)
    monkeypatch.setattr(
        retrain.subprocess, "run", lambda *a, **k: pytest.fail("Worker launched before R0")
    )
    assert run(tmp_path, tmp_path / "cache")["status"] == "R0_BLOCKED_NO_TRAINING"


def test_resource_guard_saves_before_stop(tmp_path):
    guard = TrainingGuard(0, tmp_path / "resource.json", "a" * 40, "b" * 64, "run")
    payload = {}

    def save(value, path):
        path.write_text(json.dumps(value))

    with pytest.raises(ResourceStop):
        guard.checkpoint(payload, tmp_path / "saved.json", save)
    assert json.loads((tmp_path / "saved.json").read_text())["training_source_sha"] == "a" * 40


def test_safe_boundary_and_explicit_optimizer_defaults(tmp_path):
    guard = TrainingGuard(100, tmp_path / "r.json", "a" * 40, "b" * 64, "run")
    calls = []
    fake_optimizer = SimpleNamespace(step=lambda: None)
    fake_torch = SimpleNamespace(
        optim=SimpleNamespace(AdamW=lambda *a, **k: calls.append(k) or fake_optimizer),
        amp=SimpleNamespace(GradScaler=lambda *a, **k: SimpleNamespace(update=lambda: None)),
    )
    proxy = guard.proxy(fake_torch)
    proxy.optim.AdamW([]).step()
    assert guard.actual_updates == 1 and calls[0]["eps"] == 1e-8 and calls[0]["fused"] is None
    saved = []
    frame = SimpleNamespace(
        f_locals={"step": 100, "checkpoint_set": {1000000}, "save_resume": saved.append}
    )
    guard.after_update(frame)
    assert saved == [100]
    frame.f_locals["step"] = 1000
    guard.after_update(frame)
    assert saved == [100]


def test_resume_rejects_cross_model_seed_and_source(tmp_path):
    import torch

    c = configuration()
    m = manifest(tmp_path, c)
    path = tmp_path / "resume.pt"
    torch.save(
        {"model": c["models"]["T1"], "seed": 91003, "step": 10000, "consumed_tokens": 10000000},
        path,
    )
    with pytest.raises(RuntimeError, match="RESUME_IDENTITY"):
        validate_resume(path, c, m, c["models"]["T1"])


def test_guard_observation_preserves_synthetic_cpu_step(tmp_path):
    import torch
    from epclm001.common import native
    from epclm001.publication import parent

    base, _, _ = native()
    _, bind = parent()
    corpus = SimpleNamespace(
        tokenizer_path=tmp_path / "tokenizer",
        train_path=tmp_path / "train",
        validation_path=tmp_path / "val",
        manifest={},
    )
    profile = base.Profile("fixture", 8, 32, 16, (8,), 1)

    def build(name, vocab):
        return torch.nn.Sequential(torch.nn.Embedding(vocab, 4), torch.nn.Linear(4, vocab))

    overrides = {
        "VOCAB_SIZE": 32,
        "TOKENS_PER_STEP": 8,
        "BATCH_SIZE": 2,
        "TRAIN_SEQUENCE_LENGTH": 4,
        "CONTEXT_LENGTH": 4,
        "load_tokenizer": lambda p: SimpleNamespace(get_vocab_size=lambda: 32),
        "open_memmap": lambda p: list(range(32)),
        "fixed_validation_starts": lambda *a, **kw: (),
        "memmap_batch": lambda *a: (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]]),
        ),
        "evaluate": lambda *a: {
            "validation_nll": 1.0,
            "validation_ppl": 2.718,
            "validation_tokens": 8,
        },
        "build_model": build,
        "model_config": lambda *a: {},
        "estimate_flops": lambda *a, **kw: {},
    }
    bind(base.train_one, **overrides)(
        "fixture",
        corpus=corpus,
        profile=profile,
        output_dir=tmp_path / "plain",
        seed=91002,
        device=torch.device("cpu"),
    )
    guard = TrainingGuard(100, tmp_path / "resource.json", "a" * 40, "b" * 64, "fixture")

    def save(payload, path):
        guard.checkpoint(payload, path, base._atomic_save)

    bind(base.train_one, **dict(overrides, torch=guard.proxy(torch), _atomic_save=save))(
        "fixture",
        corpus=corpus,
        profile=profile,
        output_dir=tmp_path / "guarded",
        seed=91002,
        device=torch.device("cpu"),
    )
    plain = torch.load(tmp_path / "plain/fixture/resume.pt", map_location="cpu", weights_only=True)
    observed = torch.load(
        tmp_path / "guarded/fixture/resume.pt", map_location="cpu", weights_only=True
    )
    assert guard.actual_updates == 1 and observed["actual_optimizer_updates"] == 1
    for key in plain["model_state_dict"]:
        torch.testing.assert_close(
            plain["model_state_dict"][key], observed["model_state_dict"][key], atol=0, rtol=0
        )
    assert plain["batch_rng_state"] == observed["batch_rng_state"]


def test_report_publication_excludes_weights(tmp_path):
    from epclm001.retrain_publication import payload, snapshot

    c = configuration()
    manifest(tmp_path, c)
    snapshot(tmp_path, c)
    (tmp_path / "resume.pt").write_bytes(b"private")
    _, files, _, _ = payload(tmp_path)
    assert "resume.pt" not in files
    assert "TRAINING_MANIFEST.json" in files


def test_pristine_report_manifest_does_not_hide_remote_controller(tmp_path, monkeypatch):
    import huggingface_hub
    from epclm001.retrain_assets import restore_controller
    from huggingface_hub.utils import EntryNotFoundError

    c = configuration()
    manifest(tmp_path, c)
    monkeypatch.setenv("HF_TOKEN", "synthetic-test-credential")
    called = []

    def missing(*args, **kwargs):
        called.append(True)
        raise EntryNotFoundError("Synthetic missing index")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    restore_controller(tmp_path, c)
    assert called == [True]


def test_worker_requires_controller_reservation(tmp_path):
    from epclm001.retrain import worker

    c = configuration()
    m = manifest(tmp_path, c)
    m["status"] = "R0_PASS"
    (tmp_path / "TRAINING_MANIFEST.json").write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="WORKER_GATE"):
        worker(tmp_path, "T1", 100)


def test_failed_independent_readback_never_marks_available(tmp_path, monkeypatch):
    import shutil
    import subprocess

    import huggingface_hub
    from epclm001 import retrain_assets

    c = configuration()
    directory = tmp_path / "models" / c["models"]["T1"]
    directory.mkdir(parents=True)
    names = (
        "resume.pt",
        "inference.pt",
        "IDENTITY.json",
        "summary.json",
        "checkpoints.csv",
        "QUALITY.json",
        "tokenizer.json",
        "corpus-manifest.json",
        "REAL_WEIGHT_PARITY.json",
    )
    for name in names:
        (directory / name).write_bytes(b"synthetic fixture")
    monkeypatch.setattr(retrain_assets, "token", lambda: "synthetic-test-credential")
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda **kw: SimpleNamespace(upload_folder=lambda **kw: SimpleNamespace(oid="a" * 40)),
    )

    def fetch(repo, path, **kwargs):
        target = Path(kwargs["local_dir"]) / Path(path).name
        shutil.copyfile(directory / target.name, target)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(retrain_assets.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        retrain_assets.publish_model(tmp_path, c, "test-run", "T1")
    assert not (directory / "PERSISTENCE.json").exists()


def test_probe_compatible_with_hub_036_and_checks_real_write(tmp_path, monkeypatch):
    import shutil

    import huggingface_hub
    from epclm001 import retrain_assets

    calls = []

    def auth(repo_id, *, repo_type=None, token=None):
        calls.append("access")

    def upload(**kwargs):
        calls.append("write")
        return SimpleNamespace(oid="a" * 40)

    def download(repo, path, **kwargs):
        assert kwargs["force_download"] is True and kwargs["revision"] == "a" * 40
        calls.append("readback")
        copy = Path(kwargs["local_dir"]) / "probe.json"
        shutil.copyfile(tmp_path / "PERSISTENCE_PROBE.json", copy)
        return str(copy)

    monkeypatch.setattr(retrain_assets, "token", lambda: "synthetic-test-credential")
    monkeypatch.setattr(huggingface_hub, "auth_check", auth)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: SimpleNamespace(upload_file=upload))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert (
        retrain_assets.probe(tmp_path, configuration(), "test")["status"]
        == "PERSISTENCE_PREFLIGHT_PASS"
    )
    assert calls == ["access", "write", "readback"]


def test_zero_progress_manifest_refreshes_control_source_only(tmp_path):
    c = configuration()
    m = manifest(tmp_path, c)
    previous = m["source_sha"]
    m["control_source_files"] = {"old": "old"}
    (tmp_path / "TRAINING_MANIFEST.json").write_text(json.dumps(m))
    updated = manifest(tmp_path, c)
    assert updated["control_source_files"] != {"old": "old"}
    assert updated["pretraining_engineering_revisions"][-1]["source_sha"] == previous
    updated["models"]["T1"] = "TRAINED"
    updated["control_source_files"] = {"trained-old": "preserve"}
    (tmp_path / "TRAINING_MANIFEST.json").write_text(json.dumps(updated))
    assert manifest(tmp_path, c)["control_source_files"] == {"trained-old": "preserve"}


def test_persistence_api_error_is_reported_as_blocked_r0(tmp_path, monkeypatch):
    import torch
    from epclm001 import retrain, retrain_assets, retrain_data

    c = configuration()

    def model(name, *args):
        key = next(k for k, v in c["models"].items() if v == name)
        return SimpleNamespace(
            parameters=lambda: [SimpleNamespace(numel=lambda: c["parameter_counts"][key])]
        )

    monkeypatch.setattr(
        retrain,
        "native",
        lambda: (
            SimpleNamespace(build_model=model, environment_record=dict),
            None,
            SimpleNamespace(build_phase_c=model),
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(retrain_assets, "restore_controller", lambda *args: None)
    monkeypatch.setattr(retrain_data, "prepare", lambda *args: None)

    def broken(*args):
        raise TypeError("Synthetic incompatible persistence API")

    monkeypatch.setattr(retrain_assets, "probe", broken)
    result = retrain.preflight(tmp_path, tmp_path / "cache")
    assert result["status"] == "R0_BLOCKED_NO_TRAINING"
    assert result["preflight"]["persistence"]["code"] == "TypeError"
    assert result["models"] == {"T1": "NOT_STARTED", "X3": "NOT_STARTED", "X4": "NOT_STARTED"}
