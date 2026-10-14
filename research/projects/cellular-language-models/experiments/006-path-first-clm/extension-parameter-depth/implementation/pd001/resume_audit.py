"""Bounded synthetic checkpoint rehearsal; never scientific/confirmation training."""

import copy
import json
from pathlib import Path
import numpy as np
import torch
from exp006.reuse import write_json, canonical_sha256, file_sha256
from .configs import ROOT, arm_config
from .runner import fixture_config
from .persistence import export_bundle, import_bundle

SCOPE = "PD001_SYNTHETIC_PERSISTENCE_REHEARSAL"


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def run(c, p, args):
    import subprocess
    from . import training

    if args.device.startswith("cuda") and not args.allow_gpu_microbenchmark:
        raise RuntimeError("explicit Kaggle GPU microbenchmark flag required")
    if args.arm not in ("M0", "P50-F16"):
        raise ValueError("rehearsal supports actual M0/P50-F16 only")
    source = (
        args.source_commit
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    from .configs import BASE, PARENT

    protected = [
        str((BASE / "implementation").relative_to(ROOT)),
        str((PARENT / "implementation").relative_to(ROOT)),
        "shared/python",
        "src",
    ]
    if subprocess.run(["git", "diff", "--quiet", source, "--", *protected], cwd=ROOT).returncode:
        raise RuntimeError("checkpoint rehearsal requires code matching its declared source SHA")
    work = Path(args.output)
    if args.rehearsal_phase == "pause":
        if work.exists():
            raise FileExistsError("fresh rehearsal directory required")
        c = copy.deepcopy(c)
        c["engineering_scope"] = SCOPE
        if args.perf_fixture:
            c["performance_fixture_model"] = fixture_config(c, args.arm)
            c["training"].update(sequence_length=4, effective_batch=4)
        c["training"].update(steps=4, checkpoint_every=1)
        c["training"]["tokens"] = 4 * c["training"]["effective_batch"] * c["training"]["sequence_length"]
        c["performance"].update(pause_after_step=2, max_session_seconds=None)
        c["rehearsal_arm"] = args.arm
        work.mkdir(parents=True)
        (work / "assets").mkdir()
        model_config = c.get("performance_fixture_model") or arm_config(c, args.arm)
        tokens = np.arange(512, dtype=np.int32) % model_config["vocab_size"]
        np.save(work / "assets/synthetic.npy", tokens)
        manifest = {
            "scope": SCOPE,
            "source_sha": source,
            "tokens_sha256": file_sha256(work / "assets/synthetic.npy"),
        }
        write_json(work / "assets/manifest.json", manifest)
        write_json(work / "execution-config.json", c)
        write_json(
            work / "execution-identity.json",
            {
                "source_sha": source,
                "config_sha256": canonical_sha256(c),
                "scientific_decision": False,
                "scope": SCOPE,
            },
        )
    else:
        if not args.restore_from:
            raise ValueError(
                "attach the retained Kaggle Output and specify --restore-from its portable directory"
            )
        c = import_bundle(args.restore_from, work, source)
        if (
            c.get("engineering_scope") != SCOPE
            or c["training"]["steps"] != 4
            or c["rehearsal_arm"] != args.arm
        ):
            raise ValueError("not the matching bounded synthetic rehearsal")
    tokens = np.load(work / "assets/synthetic.npy")
    manifest = json.loads((work / "assets/manifest.json").read_text())
    if file_sha256(work / "assets/synthetic.npy") != manifest["tokens_sha256"]:
        raise ValueError("synthetic rehearsal data hash mismatch")
    starts = list(range(0, 400, c["training"]["sequence_length"]))
    assets = {
        "train": tokens,
        "train_starts": starts,
        "manifest_sha256": file_sha256(work / "assets/manifest.json"),
        "tokenizer_sha256": canonical_sha256({"scope": SCOPE, "synthetic_integer_ids": True}),
    }
    original_loader, original_arm = training.load_bundle, training.arm_config
    training.load_bundle = lambda *a, **kw: {
        "tokens": tokens,
        "starts": starts[:2],
        "groups": [0, 1],
        "evaluation_sha": canonical_sha256(manifest),
    }
    if "performance_fixture_model" in c:
        training.arm_config = lambda *a: c["performance_fixture_model"]
    try:
        folder = work / "runs" / args.arm
        result = training._train_registered(
            c,
            p,
            args.arm,
            17,
            "development",
            source,
            assets,
            "synthetic",
            folder,
            args.device,
            args.rehearsal_phase == "resume",
        )
        if args.rehearsal_phase == "pause":
            export_bundle(work, work / "portable", source)
            result = {
                "status": "PAUSED_RESUMABLE",
                "portable_output": str(work / "portable"),
                "scientific_decision": False,
                "actual_kaggle_persistence_verified": False,
            }
        else:
            control = copy.deepcopy(c)
            control["performance"]["pause_after_step"] = None
            training._train_registered(
                control,
                p,
                args.arm,
                17,
                "development",
                source,
                assets,
                "synthetic",
                work / "uninterrupted",
                args.device,
                False,
            )
            a = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
            b = torch.load(work / "uninterrupted/checkpoint.pt", map_location="cpu", weights_only=False)
            keys = ("model", "optimizer", "step", "numpy_rng", "torch_rng", "python_rng", "cuda_rng")
            parity = all(equal(a[k], b[k]) for k in keys) and equal(
                a["extra"]["path_rngs"], b["extra"]["path_rngs"]
            )
            if not parity:
                raise AssertionError("checkpoint restart differs from uninterrupted training")
            result = {
                "status": "SYNTHETIC_RESUME_PARITY_PASS",
                "source_sha": source,
                "arm": args.arm,
                "device": args.device,
                "scientific_decision": False,
                "actual_kaggle_persistence_verified": False,
                "note": "User must verify imported files came from a retained previous Kaggle cloud session; local path/process parity alone cannot prove this",
            }
        write_json(work / "resume-audit.json", result)
        return result
    finally:
        training.load_bundle, training.arm_config = original_loader, original_arm
