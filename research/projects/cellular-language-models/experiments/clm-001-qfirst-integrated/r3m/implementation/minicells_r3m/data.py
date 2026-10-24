"""Historical tokenizer/train identity, separate report roles and deterministic curriculum batches."""

from pathlib import Path
import hashlib
import itertools
import torch
from minicells_r3e.common import atomic, canonical, digest, read
from minicells_r3h.evaluation import assets as historical_assets
from minicells_r3c_moe_bridge.data import train_batches
from minicells_r3l.data_boundaries import REVISION, TOKENIZER
from .arithmetic import generate, isolation
from .protocol import task_for

HISTORY = {
    "train": (0, 8192),
    "r3k_probe": (8192, 8448),
    "r3k_holdout": (8448, 8960),
    "r3l_search": (8960, 9216),
    "r3l_select": (9216, 9472),
    "r3l_report": (9472, 9984),
    "r3l_path": (9984, 10048),
    "calibration": (50000, 50128),
    "admission": (50128, 50256),
    "development": (50256, 50512),
}


def boundaries(c):
    legacy_path = (
        Path(__file__).resolve().parents[4] / "005-compositional-cell-graph/config/registered.json"
    )
    legacy = read(legacy_path)["data"]
    if legacy["dataset_revision"] != REVISION:
        raise RuntimeError("GLOBAL_DATASET_REGISTRY_REVISION_CHANGED")
    historical = dict(HISTORY)
    if c.get("story_isolation_scope", "GLOBAL_HISTORY") == "GLOBAL_HISTORY":
        historical["experiment_005_registered_train"] = (0, legacy["train_stories"])
    elif c.get("story_isolation_scope") != "R3H_LINEAGE" or not c.get("data_registration_reviewed"):
        raise RuntimeError("STORY_ISOLATION_SCOPE_NOT_REGISTERED")
    prior = list(historical.values())
    for name, (a, b) in c["story_ranges"].items():
        if not a < b <= 60000 or any(a < y and x < b for x, y in prior):
            conflicts = [key for key, (x, y) in historical.items() if a < y and x < b]
            raise RuntimeError("STORY_RANGE_CONFLICT:" + name + ":" + ",".join(conflicts))
        prior.append((a, b))
    return dict(
        status="INDEX_DISJOINT",
        historical=historical,
        roles=c["story_ranges"],
        confirmation="SEALED",
        scope=c.get("story_isolation_scope", "GLOBAL_HISTORY"),
    )


def prepare(c, runtime, backend, folder, main=False):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from fast001.data import SHARD

    proof = boundaries(c)
    folder = Path(folder)
    historical = historical_assets(runtime, backend, folder / "historical", ("train",))
    config = runtime["engine"]["input"]["data_config"]
    if config["revision"] != REVISION or digest(folder / "historical/tokenizer.json") != TOKENIZER:
        raise RuntimeError("R3H_DATA_IDENTITY_CHANGED")
    source = hf_hub_download(
        repo_id=config["dataset"], repo_type="dataset", revision=REVISION, filename=SHARD
    )
    if digest(source) != historical["manifest"]["source_sha256"]:
        raise RuntimeError("HISTORICAL_SOURCE_HASH_CHANGED")
    roles = ("trigger", "main_report") if main else ("trigger", "pilot_report")
    ranges = {r: c["story_ranges"][r] for r in roles}
    hist_hashes = set()
    selected = {r: [] for r in roles}
    rows = load_dataset("parquet", data_files=source, split="train", streaming=True)
    # No Main Report text is materialized in the Pilot. No Confirmation text is accessed.
    for i, row in enumerate(itertools.islice(rows, max(b for a, b in HISTORY.values()))):
        historical_row = any(a <= i < b for a, b in HISTORY.values())
        role = next((r for r, (a, b) in ranges.items() if a <= i < b), None)
        if not historical_row and role is None:
            continue
        text = row["text"]
        h = hashlib.sha256(text.encode()).hexdigest()
        if historical_row:
            hist_hashes.add(h)
        else:
            selected[role].append((i, text, h))
    tokenizer = Tokenizer.from_file(str(folder / "historical/tokenizer.json"))
    seen = set(hist_hashes)
    story, excluded = {}, {}
    for role in roles:
        story[role], excluded[role] = [], []
        for i, text, h in selected[role]:
            if h in seen:
                excluded[role].append(dict(index=i, sha256=h))
                continue
            seen.add(h)
            ids = tokenizer.encode(text).ids + [1]
            n = runtime["engine"]["training"]["sequence_length"]
            windows = [ids[start : start + n + 1] for start in range(0, len(ids) - n, n)]
            if windows:
                story[role].append(dict(group=i, sha256=h, windows=windows))
        if not story[role]:
            raise RuntimeError("EMPTY_STORY_ROLE:" + role)
    arith, arith_meta = {}, {}
    for role in ("train",) + roles:
        arith[role], arith_meta[role] = generate(role, tokenizer)
    group_proof = isolation(arith)
    manifest = dict(
        config_sha256=canonical(c),
        revision=REVISION,
        source_sha256=digest(source),
        tokenizer_sha256=TOKENIZER,
        generator_sha256=digest(Path(__file__).with_name("arithmetic.py")),
        generator=c["arithmetic"],
        arithmetic=arith_meta,
        group_isolation=group_proof,
        story_excluded=excluded,
        main_report_materialized=main,
        confirmation="SEALED",
    )
    payload = dict(story=story, arithmetic=arith)
    manifest["content_sha256"] = canonical(payload)
    atomic(folder / "research.json", payload)
    atomic(folder / "DATA_BOUNDARY_PROOF.json", proof)
    atomic(folder / "manifest.json", manifest)
    return historical, payload, manifest


def load(folder, c):
    folder = Path(folder)
    meta, payload = read(folder / "manifest.json"), read(folder / "research.json")
    if (
        meta["config_sha256"] != canonical(c)
        or canonical(payload) != meta["content_sha256"]
        or meta["confirmation"] != "SEALED"
    ):
        raise RuntimeError("R3M_ASSET_IDENTITY_CHANGED")
    return payload, meta


def batches(tr, historical, research, environment, c):
    task = task_for(tr.step - c["anchor_step"], environment)
    engine = tr.c["engine"]
    device = next(tr.model.parameters()).device
    if task == "story":
        return train_batches(historical, engine, tr.data_rng, device)
    q = engine["training"]
    records = research["arithmetic"]["train"]
    result = []
    for _ in range(q["effective_batch"] // q["microbatch"]):
        rows = []
        for _ in range(q["microbatch"]):
            ids = []
            while len(ids) < q["sequence_length"] + 1:
                r = records[int(tr.data_rng.integers(len(records)))]
                ids.extend(r["ids"] + [1])
            rows.append(ids[: q["sequence_length"] + 1])
        tokens = torch.tensor(rows, device=device)
        result.append((tokens[:, :-1], tokens[:, 1:]))
    return result
