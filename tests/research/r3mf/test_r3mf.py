# ruff: noqa: E402
"""Synthetic CPU only; tokenizer identity uses restored public bytes, no private model access."""

import copy
from pathlib import Path
import runpy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = (
    ROOT / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m-f"
)
runpy.run_path(str(BASE / "run.py"), run_name="r3mf_test_bootstrap")
from minicells_r3mf import (
    protocol,
    arithmetic_tasks,
    task_audit,
    data,
    learnability,
    specialization,
    runner,
    report,
)
from minicells_r3me.common import atomic, canonical, digest, fingerprint, read
from minicells_r3m import anchor, training
from minicells_r3h.routing import Model
from minicells_r3h.trainer import Trainer
from minicells_r3c_moe_bridge.common import synthetic_configuration
from minicells_r3j.storage import Store


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def c():
    return protocol.load()


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(
        str(
            BASE.parent
            / "r3m/registration/artifact-recovery-001/independent-review-002/tokenizer.json"
        )
    )


def fixture(c):
    runtime = anchor.runtime(c)
    runtime["engine"] = synthetic_configuration()
    runtime["engine"]["routing"]["cells"] = 6
    runtime["training"]["tokens_per_step"] = 8
    c = copy.deepcopy(c)
    c.update(tokens_per_step=8, continuation_steps=2, barriers=[0, 1, 2], execution_phase="tiny")
    c["tiny"].update(steps=2, barriers=[0, 1, 2])
    c["training"].update(tokens_per_step=8, effective_batch=2, microbatch=1, sequence_length=4)
    c["model"]["sequence_length"] = 4
    c["evaluation"]["autoregressive_answer_token_cap"] = 2
    assets = {
        "train": np.tile(np.arange(2, 12, dtype=np.int32), 4),
        "windows": {"train": {"starts": [0, 4, 8, 12]}},
    }
    facts = [
        dict(
            group=str(i),
            ids=[2, 3, 4, 5],
            prompt_ids=[2, 3],
            answer_positions=[2, 3],
            answer_tokens=[4, 5],
            answer="23",
            template=0,
            sampling_weight=1,
        )
        for i in range(2)
    ]
    r = {
        "story": {"development": [dict(group="cpu", windows=[[2, 3, 4, 5, 6]])]},
        "tiny": facts,
        "a1": {"train": facts, "dev": facts, "train_eval": facts},
        "copy": {"train": facts, "dev": facts},
    }
    tok = SimpleNamespace(decode=lambda x, **kw: "0")
    return c, runtime, assets, r, tok


def parent(runtime, seed):
    model = Model(runtime, seed, "B-H")
    model.route_switch_margin.fill_(1e-6)
    model.route_numeric_tolerance.fill_(1e-7)
    tr = Trainer(model, runtime, "B-H")
    tr.parent_identity = {"scope": "SYNTHETIC_CPU_ONLY"}
    tr.step = 300
    tr.data_cursor = 2400
    return tr


def test_protocol_and_budgets(c):
    gate = protocol.cpu_gate(c)
    assert gate["pilot_updates"] == 8100 and gate["pilot_tokens"] == 4147200
    assert gate["tiny_updates"] == 200 and gate["tiny_tokens"] == 102400
    assert protocol.task_for(0, "F1-COPY") == "copy"
    assert protocol.task_for(0, "F2-A1") == "a1"
    assert all(protocol.task_for(i, "F0-STORY") == "story" for i in range(900))
    assert protocol.schedule(0.001, 0.06, 899, c) == pytest.approx((0.0002, 0.02))


def test_no_authorization_or_phase_bypass(c, tmp_path):
    with pytest.raises(RuntimeError, match="INDEPENDENT_REVIEW"):
        runner.run(c, "unused", tmp_path, tmp_path, "pilot")
    c["authorization"]["full_training"] = True
    with pytest.raises(ValueError, match="FROZEN_CONFIG_CHANGED"):
        protocol.validate(c)


def test_real_tokenizer_fact_roles_and_exact_numeric_masks(c, tokenizer):
    assert (
        digest(
            BASE.parent
            / "r3m/registration/artifact-recovery-001/independent-review-002/tokenizer.json"
        )
        == c["model"]["tokenizer_sha256"]
    )
    r = arithmetic_tasks.generate(c, tokenizer)
    a = task_audit.audit(c, r)
    assert a["facts"]["a1"]["train"] == 44 and a["facts"]["a1"]["dev"] == 11
    assert a["facts"]["copy"]["train"] == 15 and a["facts"]["copy"]["dev"] == 4
    assert len(r["tiny"]) == 8
    for task in ("a1", "copy"):
        for role in ("train", "dev"):
            for row in r[task][role]:
                assert row["prompt_ids"] == tokenizer.encode(row["prompt"]).ids
                assert row["ids"][: len(row["prompt_ids"])] == row["prompt_ids"]
                assert tokenizer.decode(row["answer_tokens"]) == row["answer"]
                assert all(
                    row["text"][a:b].isdigit()
                    for i, (a, b) in enumerate(tokenizer.encode(row["text"]).offsets)
                    if i in row["answer_positions"]
                )
                assert 1 not in row["ids"]
    # Multi-token answers are actually exercised with the original tokenizer.
    assert any(len(row["answer_tokens"]) > 1 for row in r["a1"]["dev"])
    assert 0 < a["total_variation_distance"] < 1
    assert a["confirmation"] == "SEALED" and not a["r3m_main_candidates_accessed"]


def test_swapped_facts_never_cross_roles(c, tokenizer):
    r = arithmetic_tasks.generate(c, tokenizer)
    roles = {}
    for role in ("train", "dev"):
        for row in r["a1"][role]:
            fact = tuple(sorted(row["operands"]))
            assert roles.setdefault(fact, role) == role
    r["a1"]["dev"].append(r["a1"]["train"][0])
    with pytest.raises(ValueError, match="LEAKAGE"):
        task_audit.audit(c, r)


@pytest.mark.parametrize(
    "a,b,expected", [(23, 14, "NONE"), (27, 18, "UNITS"), (60, 50, "TENS"), (78, 65, "BOTH")]
)
def test_future_carry_classification_only_no_future_training(a, b, expected, c):
    assert arithmetic_tasks.carry_class(a, b) == expected
    assert not c["arithmetic"]["later_stages_authorized"]


def test_generation_does_not_receive_answer_and_handles_eos():
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(1))
            self.c = {"model": {"max_sequence_length": 64}}
            self.calls = []

        def forward(self, x):
            self.calls.append(x.tolist()[0])
            t = [7, 8, 1][len(self.calls) - 1]
            logits = torch.zeros(1, len(x[0]), 10)
            logits[0, -1, t] = 10
            return SimpleNamespace(logits=logits)

    m = Decoder()
    tokens, ended = learnability.generate(m, [2, 3], 8)
    assert tokens == [7, 8] and ended
    assert m.calls == [[2, 3], [2, 3, 7], [2, 3, 7, 8]]


def test_fact_aggregation_and_intervals_not_variant_sample_size():
    rows = [
        dict(
            group="f",
            answer="10",
            template=i,
            exact_match=(i == 0),
            first_token_correct=True,
            answer_nll=1.0,
            full_nll=2.0,
        )
        for i in range(3)
    ]
    a = learnability.aggregate(rows)
    assert a["facts"] == 1 and a["exact_match"] == pytest.approx(1 / 3)
    assert a["strict_all_variants_successes"] == 0
    assert a["strata"]["two_digit"]["facts"] == 1
    assert a["strict_fact_wilson95"][0] == 0


def test_eos_delimiters_and_exact_batches(c):
    q, runtime, assets, r, tok = fixture(c)
    tr = parent(runtime, 151001301)
    for arm in ("TINY-A1", "F0-STORY", "F1-COPY", "F2-A1"):
        for step in (0, 1):
            batches, task = data.batches(tr, assets, r, step, arm, q)
            assert sum(x.numel() for x, y in batches) == 8
            assert all(x.shape == y.shape == (1, 4) for x, y in batches)
    assert c["data"]["main_candidates_forbidden"] and c["data"]["trigger_pilot_ranges_forbidden"]


def test_full_state_restore_exact_next_update_and_no_connection_event(c, tmp_path):
    q, runtime, assets, r, tok = fixture(c)
    tr = parent(runtime, 151001301)
    state = copy.deepcopy(tr.state())
    ident = dict(
        experiment_id="SYNTHETIC_R3MF",
        run_id="cpu",
        source_sha="cpu",
        config_sha256=canonical(q),
        seed=151001301,
        arm="TINY-A1",
    )
    root = tmp_path / "private-checkpoints/cpu/jobs/151001301/TINY-A1"
    execute = runner.training_runner(q, tmp_path, ident, 151001301, "TINY-A1")
    final, aux = execute(
        q,
        runtime,
        SimpleNamespace(state=lambda: copy.deepcopy(state)),
        ident,
        root,
        assets,
        r,
        "N",
        "STABLE",
        "cpu",
    )
    assert aux["main_updates"] == 2 and aux["transactions"] == [] and aux["shadow_updates"] == 0
    assert fingerprint(final.model.connection.state_dict()) == fingerprint(
        state["model"] if False else tr.model.connection.state_dict()
    )
    result = runner.restore_check(q, runtime, root, ident, "cpu")
    assert result["exact_full_state"]
    store = Store(root, ident)
    boundary = store.verified(read(store.receipt_file(1)))
    left = training.restore(runtime, copy.deepcopy(boundary["trainer"]), 151001301, "cpu")
    right = training.restore(runtime, copy.deepcopy(boundary["trainer"]), 151001301, "cpu")
    for candidate in (left, right):
        batches, _ = data.batches(candidate, assets, r, 1, "TINY-A1", q)
        runner.advance(candidate, batches, q, boundary["aux"]["schedule"])
    assert fingerprint(training.semantic(left.state())) == fingerprint(
        training.semantic(right.state())
    )
    assert fingerprint(left.state()["model"]) == fingerprint(final.state()["model"])


def test_nine_paired_cpu_jobs_and_reports(c, tmp_path):
    q, runtime, assets, r, tok = fixture(c)
    q["execution_phase"] = "pilot"
    q["arms"] = list(protocol.ARMS)
    ident = dict(
        experiment_id="SYNTHETIC_R3MF",
        run_id="matrix",
        source_sha="cpu",
        config_sha256=canonical(q),
    )
    for seed in q["seeds"]:
        tr = parent(runtime, seed)
        state = copy.deepcopy(tr.state())
        initial = []
        for arm in q["arms"]:
            root = runner.location(tmp_path, ident, seed, arm)
            aid = dict(ident, seed=seed, arm=arm)
            runner.training_runner(q, tmp_path, ident, seed, arm)(
                q,
                copy.deepcopy(runtime),
                SimpleNamespace(state=lambda: copy.deepcopy(state)),
                aid,
                root,
                assets,
                r,
                "N",
                "STABLE",
                "cpu",
            )
            initial.append(
                Store(root, aid).verified(read(Store(root, aid).receipt_file(0)))["trainer"]
            )
            runner.restore_check(q, runtime, root, aid, "cpu")
            result = runner.analyze_job(
                q, tmp_path, runtime, assets, r, tok, ident, seed, arm, "cpu"
            )
            assert (
                result["main_updates"] == 2
                and result["tokens"] == 16
                and result["cell_calls"] == 128
            )
            for e in result["evaluations"].values():
                assert len(e["interventions"]["rows"]) == 18 and len(e["gradient"]["cells"]) == 6
                assert set(e["metrics"]["exposure"]) == {"story", "copy", "a1"}
                assert np.asarray(e["metrics"]["exposure"]["a1"]).shape == (8, 6)
        assert len({fingerprint(training.semantic(x)) for x in initial}) == 1
    runner.training_barrier(q, tmp_path, ident)
    # Synthetic templates differ from production; use a fixture held-out index.
    q["arithmetic"]["heldout_template"] = 0
    manifest = {"task_audit": {"frequency_baseline": {"dev_fact_accuracy": 0.0}}}
    out = report.write(q, tmp_path, ident, manifest)
    assert out["status"] != "PARTIAL"
    assert set(report.ARTIFACTS) <= {p.name for p in (tmp_path / "reports").iterdir()}
    for name, meta in out["artifact_files"].items():
        assert digest(tmp_path / "reports" / name) == meta["sha256"]
    missing = runner.location(tmp_path, ident, q["seeds"][0], q["arms"][0]) / "EVALUATION.json"
    missing.unlink()
    assert report.write(q, tmp_path, ident, manifest)["status"] == "PARTIAL"


def test_diagnostics_preserve_model_rng_work_and_full_signed_gradient_layout(c):
    q, runtime, assets, r, tok = fixture(c)
    tr = parent(runtime, 151001301)
    before = copy.deepcopy(tr.state())
    m = learnability.measure(tr.model, r, tok, q)
    assert m["a1"]["facts"] == 2
    g = specialization.gradient_conflict(tr, assets, r, 151001301, q)
    interventions = specialization.interventions(tr.model, r, tok, q)
    assert fingerprint(training.semantic(before)) == fingerprint(training.semantic(tr.state()))
    assert tr.model.work == before["work"]
    assert len(g["cells"]) == 6 and all(x["dimensions"] > 0 for x in g["cells"])
    assert set(g["cells"][0]["cosines"]) == {"story_copy", "story_a1", "copy_a1"}
    assert len(interventions["rows"]) == 18
    for row in interventions["rows"]:
        assert row["cell"] != row["replacement"] and row["calls_per_token"] == 8
        assert row["specificity"] == pytest.approx(
            row["normalized_delta"]["a1"] - row["normalized_delta"]["story"]
        )
    assert specialization.normalized(2, 3) == 0.5


def test_asset_load_blocks_relabeling_or_forbidden_role(c, tmp_path):
    r = {}
    manifest = dict(
        experiment_id=c["experiment_id"],
        config_sha256=canonical(c),
        research_payload_sha256=canonical(r),
        confirmation="SEALED",
        r3m_main_candidates_accessed=False,
    )
    atomic(tmp_path / "manifest.json", manifest)
    atomic(tmp_path / "research.json", r)
    assert data.load(tmp_path, c)[0] == {}
    manifest["r3m_main_candidates_accessed"] = True
    atomic(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        data.load(tmp_path, c)


def test_allowed_historical_roles_only_and_manifest_integrity(c, tokenizer, tmp_path, monkeypatch):
    import shutil

    called = []
    original = (
        BASE.parent / "r3m/registration/artifact-recovery-001/independent-review-002/tokenizer.json"
    )

    def train(runtime, backend, folder, roles):
        called.append(roles)
        assert roles == ("train",)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, folder / "tokenizer.json")
        return {"manifest": {"scope": "synthetic_train"}}

    def dev(runtime, backend, folder, roles):
        called.append(roles)
        assert roles == ("development",)
        return {
            "manifest": {"scope": "synthetic_dev"},
            "development": np.zeros(1024, dtype=np.int32),
            "windows": {
                "development": {"starts": list(range(64)), "groups": [i // 2 for i in range(64)]}
            },
        }

    monkeypatch.setattr(data, "r3h_train_assets", train)
    monkeypatch.setattr(data.historical, "r3h_assets", dev)
    train_assets, r, manifest, path = data.prepare(c, anchor.runtime(c), None, tmp_path)
    assert called == [("train",), ("development",)]
    assert manifest["forbidden_roles_materialized"] == []
    assert manifest["confirmation"] == "SEALED"
    assert data.load(tmp_path, c)[0] == r
    r["tiny"][0]["answer"] = "999"
    atomic(tmp_path / "research.json", r)
    with pytest.raises(ValueError):
        data.load(tmp_path, c)


def test_private_publication_namespace_and_cross_session_cpu_restore(c, tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "tests/engineering"))
    from test_hf_storage import FakeHub
    from minicells_r3mf.storage import SnapshotPublisher
    from minicells_r3m.storage import restore as restore_private

    q, runtime, assets, r, tok = fixture(c)
    q["seeds"] = [151001301]
    q["arms"] = ["TINY-A1"]
    ident = dict(
        experiment_id="SYNTHETIC_R3MF",
        run_id="hf-cpu",
        source_sha="cpu",
        config_sha256=canonical(q),
    )
    tr = parent(runtime, 151001301)
    state = copy.deepcopy(tr.state())
    arm = "TINY-A1"
    root = runner.location(tmp_path, ident, 151001301, arm)
    aid = dict(ident, seed=151001301, arm=arm)
    runner.training_runner(q, tmp_path, ident, 151001301, arm)(
        q,
        runtime,
        SimpleNamespace(state=lambda: copy.deepcopy(state)),
        aid,
        root,
        assets,
        r,
        "N",
        "STABLE",
        "cpu",
    )
    runner.restore_check(q, runtime, root, aid, "cpu")
    runner.analyze_job(q, tmp_path, runtime, assets, r, tok, ident, 151001301, arm, "cpu")
    hub = FakeHub(tmp_path / "hub")
    writer = SnapshotPublisher(
        hub, tmp_path / "private-checkpoints" / ident["run_id"], tmp_path, ident, "session-A"
    )
    ref = writer.publish(True)
    assert hub.optional("r3m-f/hf-cpu/latest.json", ref["hf_revision"])
    assert not hub.optional("r3m-e/hf-cpu/latest.json", ref["hf_revision"])
    target = tmp_path / "session-B"
    target.mkdir()
    restore_private(
        hub,
        ref,
        target / "private-checkpoints" / ident["run_id"],
        target,
        ident,
        "session-B",
        q["barriers"],
    )
    runner.training_barrier(q, target, ident)
    result = report.write(q, target, ident, {})
    assert result["status"] in ("TINY_OVERFIT_FAILED", "TINY_OVERFIT_PASS_REVIEW_REQUIRED")
    assert not result["full_pilot_authorized"]
    decision = read(target / "reports/TINY_OVERFIT_REPORT.json")
    assert decision["optimizer_updates"] == 2 and not decision["scientific_generalization_claim"]
    assert not decision["pilot_authorized"]


def test_tiny_threshold_never_becomes_pilot_release(c):
    metric = dict(facts=8, exact_match=0.75)
    result = dict(
        evaluations={
            "0": {"metrics": {"tiny": dict(facts=8, exact_match=0.0)}},
            "200": {"metrics": {"tiny": metric}},
        },
        curve=[{"cell_gradient_norms": [0, 1, 0, 0, 0, 0]}],
        parameter_updated=True,
        resume={"exact_full_state": True},
        main_updates=200,
    )
    c["continuation_steps"] = 200
    out = report.tiny_decision(result, c)
    assert out["passed"] and not out["pilot_authorized"]
    metric["exact_match"] = 0.625
    assert report.tiny_decision(result, c)["status"] == "TINY_OVERFIT_FAILED"


def test_f_publication_uses_current_main_and_only_metadata(tmp_path, monkeypatch):
    import subprocess
    from minicells_r3m import publication as inherited

    bare, seed = tmp_path / "remote.git", tmp_path / "main-seed"

    def command(*args, cwd=None):
        return subprocess.run(
            list(map(str, args)),
            cwd=cwd,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    command("git", "init", "--bare", bare)
    command("git", "init", seed)
    command("git", "symbolic-ref", "HEAD", "refs/heads/main", cwd=seed)
    command("git", "config", "user.name", "CPU Test", cwd=seed)
    command("git", "config", "user.email", "cpu@example.invalid", cwd=seed)
    (seed / "README.md").write_text("Synthetic publication fixture\n")
    command("git", "add", "README.md", cwd=seed)
    command("git", "commit", "-m", "main fixture", cwd=seed)
    command("git", "remote", "add", "origin", bare, cwd=seed)
    command("git", "push", "origin", "main", cwd=seed)
    main_sha = command("git", "rev-parse", "HEAD", cwd=seed).stdout.strip()
    output = tmp_path / "result"
    for name in report.ARTIFACTS:
        path = output / "reports" / name
        if path.suffix == ".json":
            atomic(path, {"scope": "SYNTHETIC_CPU_ONLY", "status": "COMPLETE_EXPLORATORY"})
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Synthetic CPU report\n")
    (output / "reports/not-for-publication.pt").write_bytes(b"private fixture")

    def git(checkout, *args):
        values = list(args)
        if values[:3] == ["remote", "add", "origin"]:
            values[3] = str(bare)
        return command("git", *values, cwd=checkout)

    monkeypatch.setattr(inherited, "git", git)
    monkeypatch.setattr(
        inherited,
        "push_results",
        lambda checkout, branch: git(checkout, "push", "origin", "HEAD:refs/heads/" + branch),
    )
    published = runner.publication(output)
    assert published["status"] == "PUBLISHED"
    remote_sha = command("git", "--git-dir", bare, "rev-parse", published["branch"]).stdout.strip()
    assert remote_sha == published["remote_commit_sha"]
    assert (
        command("git", "--git-dir", bare, "rev-parse", remote_sha + "^").stdout.strip() == main_sha
    )
    changed = command(
        "git", "--git-dir", bare, "diff", "--name-only", main_sha, remote_sha
    ).stdout.splitlines()
    assert {Path(path).name for path in changed} == set(report.ARTIFACTS)
    assert all("/r3m-f/reports/published/" in path for path in changed)
    assert runner.publication(output)["remote_commit_sha"] == remote_sha
