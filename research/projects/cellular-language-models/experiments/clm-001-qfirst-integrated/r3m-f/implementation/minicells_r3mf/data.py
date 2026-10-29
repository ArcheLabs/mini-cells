"""Only the pinned R3h Train/reused Development and new generated task identities."""

from pathlib import Path
from minicells_r3me import data as historical
from minicells_r3me.data import arithmetic_batches
from minicells_r3c_moe_bridge.data import train_batches
from .protocol import atomic, canonical, digest, read, task_for
from .arithmetic_tasks import generate
from .task_audit import audit

r3h_train_assets = historical.r3h_train_assets


def prepare(c, runtime, backend, folder):
    from tokenizers import Tokenizer

    folder = Path(folder)
    train = r3h_train_assets(runtime, backend, folder / "r3h-train", ("train",))
    dev = historical.r3h_assets(runtime, backend, folder / "r3h-development", ("development",))
    path = folder / "r3h-train/tokenizer.json"
    if path.is_symlink() or digest(path) != c["model"]["tokenizer_sha256"]:
        raise ValueError("ORIGINAL_TOKENIZER_CHANGED")
    if runtime["engine"]["input"]["data_config"]["revision"] != c["data"]["revision"]:
        raise ValueError("REVISION_CHANGED")
    research = generate(c, Tokenizer.from_file(str(path)))
    generated_audit = audit(c, research)
    research["story"] = {
        "development": historical._story_eval_assets(
            dev,
            c["model"]["sequence_length"],
            c["evaluation"]["story_groups"],
            c["evaluation"]["story_windows_per_group"],
        )
    }
    manifest = dict(
        experiment_id=c["experiment_id"],
        config_sha256=canonical(c),
        research_payload_sha256=canonical(research),
        train_manifest=train["manifest"],
        development_manifest=dev["manifest"],
        task_audit=generated_audit,
        tokenizer_sha256=digest(path),
        confirmation="SEALED",
        r3m_main_candidates_accessed=False,
        forbidden_roles_materialized=[],
        historical_development="REUSED_NOT_INDEPENDENT_CONFIRMATION",
    )
    old = folder / "manifest.json"
    if old.exists() and read(old) != manifest:
        raise ValueError("EXISTING_ASSETS_IDENTITY_CHANGED_USE_DISTINCT_DIRECTORY")
    atomic(folder / "research.json", research)
    atomic(old, manifest)
    return train, research, manifest, path


load = historical.load


def batches(tr, train, research, step, arm, c):
    task = task_for(step, arm)
    if task == "story":
        result = train_batches(
            train, tr.c["engine"], tr.data_rng, next(tr.model.parameters()).device
        )
    else:
        records = research["tiny"] if task == "tiny" else research[task]["train"]
        if task == "copy":
            records = [r for r in records for _ in range(r["sampling_weight"])]
        result = arithmetic_batches(tr.data_rng, records, c, tr.model)
    if sum(x.numel() for x, y in result) != c["tokens_per_step"]:
        raise ValueError("TOKEN_BUDGET_CHANGED")
    return result, task
