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
    with pytest.raises(RuntimeError, match="UNREVIEWED_CONTROL_SOURCE_MIGRATION"):
        manifest(tmp_path, c)


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


def accounting_fixture(actual=9999):
    import torch

    return {
        "step": 10000,
        "consumed_tokens": 10000000,
        "actual_optimizer_updates": actual,
        "optimizer_state_dict": {
            "param_groups": [{"params": [0, 1]}],
            "state": {
                i: {
                    "step": torch.tensor(float(actual)),
                    "exp_avg": torch.zeros(2),
                    "exp_avg_sq": torch.zeros(2),
                }
                for i in (0, 1)
            },
        },
        "scaler_state_dict": {"scale": 32768.0},
    }


@pytest.mark.parametrize("actual", [9999, 10000, 9998])
def test_amp_budget_counts_attempts_and_successes_separately(actual):
    from epclm001.retrain_accounting import completed_budget

    value = completed_budget(accounting_fixture(actual))
    assert (
        value["training_steps"]
        == value["successful_optimizer_updates"] + value["skipped_optimizer_updates"]
        == 10000
    )
    assert value["training_tokens"] == 10000000 and value["compensation_updates"] == 0
    assert value["actual_optimizer_updates"] == actual


@pytest.mark.parametrize("actual", [0, -1, 10001, "9999"])
def test_amp_budget_rejects_invalid_actual_counts(actual):
    from epclm001.retrain_accounting import completed_budget

    value = accounting_fixture()
    value["actual_optimizer_updates"] = actual
    with pytest.raises(RuntimeError, match="UPDATE_COUNT"):
        completed_budget(value)


def test_amp_budget_requires_optimizer_corroboration_and_finite_state():
    import torch
    from epclm001.retrain_accounting import completed_budget

    value = accounting_fixture()
    value["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(10000.0)
    with pytest.raises(RuntimeError, match="DISAGREES"):
        completed_budget(value)
    value = accounting_fixture()
    value["optimizer_state_dict"]["state"][0]["exp_avg"][0] = float("inf")
    with pytest.raises(RuntimeError, match="MOMENT_STATE"):
        completed_budget(value)
    value = accounting_fixture()
    value["step"] = 9999
    with pytest.raises(RuntimeError, match="LOOP_BUDGET"):
        completed_budget(value)


def completed_manifest_fixture(root):
    import hashlib

    import torch
    from epclm001.common import BASE, sha

    c = configuration()
    m = manifest(root, c)
    policy = json.loads((BASE / "amp_step_accounting.json").read_text())
    m.update(
        source_sha=policy["predecessor_commit"],
        control_source_files=policy["predecessor_control_source_files"],
        status="R2_PENDING",
        gpu_seconds_charged=224.544535773,
        cache=str(root / "cache"),
    )
    m["models"]["T1"] = "TRAINED"
    state = dict(
        accounting_fixture(),
        model=c["models"]["T1"],
        seed=91002,
        profile="baseline",
        asset_generation="CONTROLLED_RETRAIN_001",
        controlled_config_sha256=sha(c),
        retrain_run_id=m["run_id"],
        training_source_sha=m["source_sha"],
        corpus_manifest_sha256=hashlib.sha256(
            json.dumps(c["historical_corpus_manifest"], sort_keys=True).encode()
        ).hexdigest(),
    )
    path = root / "models" / c["models"]["T1"] / "resume.pt"
    path.parent.mkdir(parents=True)
    torch.save(state, path)
    (root / "TRAINING_MANIFEST.json").write_text(json.dumps(m))
    return c, m, path


def test_completed_controller_migration_preserves_original_evidence(tmp_path):
    from epclm001.common import digest

    c, old, path = completed_manifest_fixture(tmp_path)
    before = digest(path)
    new = manifest(tmp_path, c)
    assert new["run_id"] == old["run_id"] and new["source_sha"] == old["source_sha"]
    assert new["gpu_seconds_charged"] == old["gpu_seconds_charged"]
    assert new["model_training_sources"]["T1"] == old["source_sha"]
    assert new["controller_revisions"][-1]["reason"] == "HISTORICAL_FP16_STEP_ACCOUNTING"
    assert digest(path) == before
    assert len(manifest(tmp_path, c)["controller_revisions"]) == 1


def test_partial_controller_migration_fails_closed(tmp_path):
    import torch

    c, _, path = completed_manifest_fixture(tmp_path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state.update(step=9000, consumed_tokens=9000000)
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="PARTIAL_TRAINING"):
        manifest(tmp_path, c)


def test_completed_t1_is_verified_before_any_new_worker(tmp_path, monkeypatch):
    from epclm001 import retrain, retrain_assets, retrain_data

    _c, _, path = completed_manifest_fixture(tmp_path)
    monkeypatch.setattr(retrain, "source", lambda: "b" * 40)
    monkeypatch.setattr(retrain_data, "load_verified", lambda *a: None)
    monkeypatch.setattr(retrain_assets, "persist_controller", lambda *a: None)
    events = []

    def verified(root, key):
        events.append(("verify", key))
        assert key == "T1"
        current = json.loads((root / "TRAINING_MANIFEST.json").read_text())
        current["models"][key] = "ASSET_AVAILABLE"
        (root / "TRAINING_MANIFEST.json").write_text(json.dumps(current))

    monkeypatch.setattr(retrain, "verify_and_publish", verified)

    class StopBeforeActualWorker(Exception):
        pass

    def stopped(args, **kwargs):
        events.append(("worker", args[args.index("--model") + 1]))
        raise StopBeforeActualWorker()

    monkeypatch.setattr(retrain.subprocess, "run", stopped)
    with pytest.raises(StopBeforeActualWorker):
        run(tmp_path, tmp_path / "cache")
    assert events == [("verify", "T1"), ("worker", "X3")]
    assert path.exists()


def test_quality_hash_export_and_publication_retry(tmp_path, monkeypatch):
    from epclm001 import retrain, retrain_assets, retrain_data, retrain_validation
    from epclm001.common import digest

    c, m = configuration(), manifest(tmp_path, configuration())
    m["cache"] = str(tmp_path / "cache")
    (tmp_path / "TRAINING_MANIFEST.json").write_text(json.dumps(m))
    directory = tmp_path / "models" / c["models"]["T1"]
    directory.mkdir(parents=True)
    (directory / "resume.pt").write_bytes(b"unchanged synthetic checkpoint fixture")
    (tmp_path / "ASSET_MANIFEST.json").write_text(
        json.dumps({"assets": [{"model": k, "status": "PENDING"} for k in c["models"]]})
    )
    monkeypatch.setattr(retrain, "validate_resume", lambda *a: {})
    monkeypatch.setattr(retrain_data, "load_verified", lambda *a: None)
    monkeypatch.setattr(retrain_assets, "persist_controller", lambda *a: None)
    calls = []

    def verify(*args):
        calls.append("verify")
        (directory / "inference.pt").write_bytes(b"synthetic inference fixture")
        (directory / "QUALITY.json").write_text(json.dumps({"status": "QUALITY_VALID"}))
        (directory / "IDENTITY.json").write_text(
            json.dumps(
                {
                    "resume_sha256": digest(directory / "resume.pt"),
                    "inference_sha256": digest(directory / "inference.pt"),
                    "quality_sha256": digest(directory / "QUALITY.json"),
                }
            )
        )
        return {"status": "QUALITY_VALID"}

    def publish(*args):
        calls.append("publish")
        if calls.count("publish") == 1:
            raise OSError("Synthetic interrupted upload")
        return {"model": "T1", "status": "ASSET_AVAILABLE"}

    monkeypatch.setattr(retrain_validation, "verify", verify)
    monkeypatch.setattr(retrain_assets, "publish_model", publish)
    with pytest.raises(OSError):
        retrain.verify_and_publish(tmp_path, "T1")
    retrain.verify_and_publish(tmp_path, "T1")
    assert calls == ["verify", "publish", "publish"]
    assert (directory / "resume.pt").read_bytes() == b"unchanged synthetic checkpoint fixture"


def test_actual_export_writes_quality_digest_and_preserves_resume(tmp_path, monkeypatch):
    import hashlib

    import torch
    from epclm001 import retrain_validation
    from epclm001.common import digest, native
    from epclm001.validation import engineering_model

    base, _, _ = native()
    model = engineering_model("T1")
    c = configuration()
    c["parameter_counts"] = dict(
        c["parameter_counts"], T1=sum(p.numel() for p in model.parameters())
    )
    directory = tmp_path / "models" / c["models"]["T1"]
    directory.mkdir(parents=True)
    state = dict(
        accounting_fixture(),
        model=c["models"]["T1"],
        seed=91002,
        profile="baseline",
        training_source_sha="a" * 40,
        asset_generation="CONTROLLED_RETRAIN_001",
        corpus_manifest_sha256=hashlib.sha256(json.dumps({}, sort_keys=True).encode()).hexdigest(),
        model_state_dict=model.state_dict(),
        batch_rng_state=(),
        torch_rng_state=torch.get_rng_state(),
        cuda_rng_state_all=[],
        metrics=[
            {"consumed_tokens": n, "validation_nll": 1.0, "validation_ppl": 2.718}
            for n in c["checkpoints"]
        ],
    )
    torch.save(state, directory / "resume.pt")
    before = digest(directory / "resume.pt")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("synthetic unit fixture")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "validation_inputs.json").write_text(
        json.dumps({"tokens": [[i % 32 for i in range(128)]]})
    )
    monkeypatch.setattr(retrain_validation, "BASE", inputs)
    monkeypatch.setattr(base, "build_model", lambda *args: engineering_model("T1"))
    monkeypatch.setattr(base, "open_memmap", lambda *args: list(range(1024)))
    monkeypatch.setattr(base, "fixed_validation_starts", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        base,
        "evaluate",
        lambda *args: {"validation_nll": 1.0, "validation_ppl": 2.718, "validation_tokens": 8},
    )
    result = retrain_validation.verify(
        tmp_path,
        c,
        "T1",
        SimpleNamespace(
            manifest={}, tokenizer_path=tokenizer, validation_path=tmp_path / "validation"
        ),
        device="cpu",
    )
    identity = json.loads((directory / "IDENTITY.json").read_text())
    assert identity["quality_sha256"] == digest(directory / "QUALITY.json")
    assert result["training_accounting"]["actual_optimizer_updates"] == 9999
    assert digest(directory / "resume.pt") == before
