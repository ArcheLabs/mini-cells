"""R3h lineage assets plus fact-disjoint generated Arithmetic; never opens R3m candidates."""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from minicells_r3e.common import atomic, canonical, digest, read
from minicells_r3h.evaluation import assets as r3h_assets
from minicells_r3m.data import batches as r3m_batches
from minicells_r3m.data import historical_assets as r3h_train_assets
from .arithmetic import generate
from .protocol import task_for


def _story_eval_assets(assets, sequence_length, groups=32, windows_per_group=2):
    meta = assets["windows"]["development"]
    starts, group_ids = meta["starts"], meta["groups"]
    if len(starts) != len(group_ids):
        raise ValueError("R3H_DEVELOPMENT_WINDOW_METADATA_INVALID")
    tokens = assets["development"]
    grouped = {}
    selected_starts = {}
    for start, group in zip(starts, group_ids):
        group = int(group)
        lo = int(start)
        hi = lo + sequence_length + 1
        if lo < 0 or hi > len(tokens):
            raise ValueError("R3H_DEVELOPMENT_WINDOW_INVALID")
        grouped.setdefault(group, []).append(np.asarray(tokens[lo:hi], dtype=np.int64).tolist())
        selected_starts.setdefault(group, []).append(lo)
    if any(any(len(x) != sequence_length + 1 for x in w) for w in grouped.values()):
        raise ValueError("R3H_DEVELOPMENT_WINDOW_INVALID")
    eligible = [g for g in grouped if len(set(selected_starts[g])) >= windows_per_group]
    chosen = sorted(eligible, key=lambda g: hashlib.sha256(("R3ME-STORY-DEV:" + str(g)).encode()).hexdigest())[:groups]
    if len(chosen) != groups:
        raise ValueError("R3H_DEVELOPMENT_GROUPS_INSUFFICIENT")
    result = []
    for g in chosen:
        by_start = dict(zip(selected_starts[g], grouped[g]))
        chosen_starts = sorted(by_start)[:windows_per_group]
        result.append(dict(group=str(g), starts=chosen_starts, windows=[by_start[start] for start in chosen_starts]))
    return result


def prepare(c, runtime, backend, folder):
    """Materialize only R3h Train and its reused Development role, plus generated arithmetic."""
    from tokenizers import Tokenizer

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    train = r3h_train_assets(runtime, backend, folder / "r3h-train", ("train",))
    dev = r3h_assets(runtime, backend, folder / "r3h-development", ("development",))
    tokenizer_path = folder / "r3h-train" / "tokenizer.json"
    if not tokenizer_path.is_file() or tokenizer_path.is_symlink():
        raise ValueError("R3H_TOKENIZER_ASSET_MISSING")
    tokenizer_sha = digest(tokenizer_path)
    if tokenizer_sha != c["model"]["tokenizer_sha256"]:
        raise ValueError("R3ME_TOKENIZER_IDENTITY_CHANGED")
    if runtime["engine"]["input"]["data_config"]["revision"] != c["data"]["revision"]:
        raise ValueError("R3H_TRAIN_REVISION_CHANGED")
    story = _story_eval_assets(
        dev,
        runtime["engine"]["training"]["sequence_length"],
        c["evaluation"]["story_groups"],
        c["evaluation"]["story_windows_per_group"],
    )
    arithmetic, arithmetic_audit = generate(
        Tokenizer.from_file(str(tokenizer_path)),
        seed=c["arithmetic"]["generator_seed"],
        max_length=c["arithmetic"]["max_sequence_length"],
        eos_token_id=c["arithmetic"]["eos_token_id"],
    )
    arithmetic_audit["tokenizer_sha256"] = tokenizer_sha
    for role in ("train", "dev"):
        counts = Counter(row["answer"] for row in arithmetic[role])
        arithmetic_audit.setdefault("answer_frequency_baseline", {})[role] = {
            "majority_answer_fraction": max(counts.values()) / sum(counts.values()),
            "facts": sum(counts.values()),
            "scope": "DETERMINISTIC_LABEL_FREQUENCY_ONLY_NOT_MODEL_SELECTION",
        }
    from .evaluation import _fixed_arithmetic

    fixed_eval = _fixed_arithmetic(arithmetic["dev"], c["evaluation"]["arithmetic_groups"])
    arithmetic_audit["evaluation_fact_selection"] = "SHA256_GROUP_ORDER_WITH_EQUAL_CARRY_STRATA"
    arithmetic_audit["fixed_evaluation_facts"] = [
        {
            "group": row["group"],
            "sample_sha256": row["sample_sha256"],
            "carry": row["carry"],
            "answer": row["answer"],
        }
        for row in fixed_eval
    ]
    research = {"story": {"development": story}, "arithmetic": arithmetic}
    research_sha = canonical(research)
    manifest = {
        "schema": "minicells.r3me.input-identity.v1",
        "experiment_id": c["experiment_id"],
        "config_sha256": canonical(c),
        "dataset": c["data"]["dataset"],
        "dataset_revision": c["data"]["revision"],
        "r3h_train_manifest": train["manifest"],
        "r3h_development_manifest": dev["manifest"],
        "r3h_development_role": "REUSED_HISTORICAL_DEVELOPMENT_NOT_UNTOUCHED_TEST",
        "arithmetic": arithmetic_audit,
        "story_evaluation_selection": {
            "rule": c["evaluation"]["story_window_selection"],
            "groups_and_starts": [{"group": row["group"], "starts": row["starts"]} for row in story],
            "selected_payload_sha256": canonical(story),
        },
        "research_payload_sha256": research_sha,
        "forbidden_roles_materialized": [],
        "r3m_main_candidates_accessed": False,
        "confirmation": "SEALED",
        "scope": "R3H_LINEAGE_EXPLORATORY_DEVELOPMENT_WITH_DISCLOSED_HISTORICAL_OVERLAP",
    }
    atomic(folder / "research.json", research)
    atomic(folder / "manifest.json", manifest)
    return train, research, manifest, tokenizer_path


def load(folder, c):
    folder = Path(folder)
    manifest = read(folder / "manifest.json")
    research = read(folder / "research.json")
    if (
        manifest.get("experiment_id") != c["experiment_id"]
        or manifest.get("config_sha256") != canonical(c)
        or manifest.get("research_payload_sha256") != canonical(research)
        or manifest.get("confirmation") != "SEALED"
        or manifest.get("r3m_main_candidates_accessed") is not False
    ):
        raise ValueError("R3ME_INPUT_IDENTITY_CHANGED")
    return research, manifest


def batches(tr, train_assets, research, relative_step, arm, c):
    """Return exactly 8x64 tokens; task choice is fixed by relative update, never quality."""
    mixed = arm != "E0-STORY-STABLE"
    task = "story" if not mixed else task_for(relative_step)
    if task == "story":
        result = r3m_batches(tr, train_assets, research, "N", c)
    else:
        result = arithmetic_batches(tr.data_rng, research["arithmetic"]["train"], c, tr.model)
    count = sum(x.numel() for x, _ in result)
    if count != c["training"]["tokens_per_step"]:
        raise ValueError("R3ME_TOKEN_BUDGET_CHANGED")
    return result, task


def task_for(relative_step):
    if not 0 <= relative_step < 600:
        raise ValueError("R3ME_RELATIVE_STEP_INVALID")
    frac = 0.1 if relative_step < 100 else 0.3 if relative_step < 200 else 0.5
    return "arithmetic" if relative_step % 10 < int(frac * 10) else "story"


def arithmetic_batches(rng, records, c, model):
    q = c["training"]
    rows_per_update = q["effective_batch"]
    sequence = c["model"]["sequence_length"]
    device = next(model.parameters()).device
    output = []
    for _ in range(rows_per_update):
        ids = []
        while len(ids) < sequence + 1:
            row = records[int(rng.integers(len(records)))]
            ids.extend(row["ids"] + [int(c["arithmetic"]["eos_token_id"])])
        output.append(ids[: sequence + 1])
    tokens = torch.tensor(output, device=device, dtype=torch.long)
    return [
        (tokens[i : i + 1, :-1], tokens[i : i + 1, 1:])
        for i in range(rows_per_update)
    ]

