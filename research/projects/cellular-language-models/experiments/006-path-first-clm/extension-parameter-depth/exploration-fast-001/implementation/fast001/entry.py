"""Explicit staged engineering -> optional T4 pilot -> user-approved development."""

import argparse
import json
import hashlib
import time
from pathlib import Path
import torch
from exp006.reuse import canonical_sha256, file_sha256
from .io import write_json
from .config import BASE, ROOT, ARMS, load, counts, search, freeze, source_sha


def audit():
    parent = BASE.parent / "implementation/pd001"
    files = [
        "models.py",
        "router.py",
        "losses.py",
        "training.py",
        "engine.py",
        "performance.py",
        "checkpoint.py",
        "evaluation.py",
        "orchestration.py",
        "accounting.py",
        "diagnostics.py",
    ]
    return {
        "experiment_id": "006-PD1-FAST-001",
        "source_sha": source_sha(),
        "pd001_engine": "pd001-light-v1",
        "reused_files": {name: file_sha256(parent / name) for name in files},
        "historical_protection": "all tracked changes relative to parent 91b1340 must be FAST subtree, FAST tests or FAST workflow",
        "confirmation": "sealed; no confirmation arrays/tokenizers/checkpoints are read",
        "governance_metadata": {
            str(p.relative_to(ROOT)): file_sha256(p)
            for p in (
                BASE.parent.parent / "config/proposed.json",
                BASE.parent / "config/proposed.json",
                BASE.parent.parent.parent / "005-compositional-cell-graph/config/registered.json",
                ROOT / "research/governance/formal_seed_registry.json",
            )
        },
        "scientific_decision": False,
    }


def preflight(c, output):
    from .models import make_model
    from pd001.losses import joint_loss

    result = {
        "experiment_id": c["experiment_id"],
        "source_sha": source_sha(),
        "parameters": counts(c),
        "config_sha256": canonical_sha256(c),
        "arms": {},
        "gpu_used": False,
        "scientific_decision": False,
    }
    torch.set_num_threads(1)
    for arm in ARMS:
        model = make_model(c, arm).train()
        initial = hashlib.sha256()
        for name, value in model.state_dict().items():
            initial.update(name.encode())
            initial.update(value.detach().cpu().numpy().tobytes())
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[2, 3, 4, 5]])
        if arm == "M0-TINY":
            out = model(x)
        else:
            actions = torch.tensor([0, 0, c["model"]["cell_count"], 1] * 4)[: c["arms"][arm]["max_actions"]]
            out = model(x, replay=actions[None, None, :].expand(1, 4, -1), capture=True)
        objective, log = joint_loss([out], y, c["training"])
        objective.backward()
        if not torch.isfinite(objective) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise FloatingPointError("CPU gate nonfinite")
        result["arms"][arm] = {
            "initial_state_sha256": initial.hexdigest(),
            "ce": float(log["ce"]),
            "forward_backward": True,
            "gradient_modules": [
                n for n, p in model.named_parameters() if p.grad is not None and bool(p.grad.abs().sum())
            ],
        }
    from .config import governance

    result["governance"] = governance(c)
    result["scope"] = "SYNTHETIC_CPU_ONLY"
    result["real_asset_hashes_frozen"] = False
    result["gpu_pilot_completed"] = False
    result["status"] = "ENGINEERING_READY"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "gate0.json", result)
    write_json(output / "audit.json", audit())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "audit",
            "params",
            "search",
            "preflight",
            "prepare",
            "benchmark",
            "freeze",
            "train",
            "train-matrix",
            "evaluate",
            "profile",
            "intervene",
            "report",
            "export",
            "restore",
        ],
        default="preflight",
    )
    parser.add_argument("--config")
    parser.add_argument("--arm", choices=ARMS, default="CLM-S8")
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--all-arms", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--assets")
    parser.add_argument("--source-file")
    parser.add_argument("--output", default="fast001-output")
    parser.add_argument("--work")
    parser.add_argument("--bundle")
    parser.add_argument("--pilot")
    parser.add_argument("--allow-gpu-pilot", action="store_true")
    parser.add_argument("--approve-development", action="store_true")
    parser.add_argument("--accept-slower", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-session-seconds", type=float, default=None)
    args = parser.parse_args()
    c = load(args.config)
    arms = list(ARMS) if args.all_arms else args.arms or [args.arm]
    if args.max_session_seconds is not None and args.max_session_seconds <= 0:
        raise ValueError("positive optional safe-pause interval")
    if args.mode in ("audit", "params", "search", "preflight") and args.device != "cpu":
        raise RuntimeError("engineering defaults are CPU-only")
    if args.mode == "audit":
        result = audit()
        write_json(Path(args.output) / "audit.json", result)
    elif args.mode == "params":
        result = counts(c)
        write_json(Path(args.output) / "parameters.json", result)
    elif args.mode == "search":
        config, parameters = search(c)
        write_json(Path(args.output) / "balanced-candidate.json", config)
        result = parameters
    elif args.mode == "preflight":
        result = preflight(c, args.output)
    elif args.mode == "prepare":
        from .data import prepare

        result = prepare(c, args.output, args.source_file)
    elif args.mode == "benchmark":
        from .benchmark import benchmark

        # Benchmark always runs all three on one idle device, never concurrent workers.
        result = benchmark(c, args.output, args.device, allow_gpu=args.allow_gpu_pilot)
    elif args.mode == "freeze":
        if not args.pilot:
            raise RuntimeError("supply the measured pilot.json")
        result = freeze(
            c,
            json.loads(Path(args.pilot).read_text()),
            args.output,
            approve=args.approve_development,
            accept_slow=args.accept_slower,
        )
    elif args.mode == "train":
        from .train import train

        result = train(
            c,
            args.arm,
            args.assets,
            args.output,
            args.device,
            resume=args.resume,
            max_session_seconds=args.max_session_seconds,
        )
    elif args.mode == "train-matrix":
        from .schedule import matrix
        from .config import authorize
        from .data import prepare, load_assets

        for device in args.devices:
            authorize(c, device, "train")
        matrix_started = time.perf_counter()
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        assets = Path(args.assets) if args.assets else output / "assets"
        if not assets.exists():
            prepare(c, assets, args.source_file)
        load_assets(c, assets)
        existing = output / "execution-config.json"
        if existing.exists() and json.loads(existing.read_text()) != c:
            raise ValueError("existing execution config differs")
        write_json(existing, c)
        result = matrix(
            c,
            Path(args.config).resolve(),
            assets.resolve(),
            output.resolve(),
            args.devices,
            arms=args.arms or ["CLM-S16", "CLM-S8", "M0-TINY"],
            resume=args.resume,
            max_session_seconds=args.max_session_seconds,
        )
        # Workers have joined; standalone evaluate/profile/intervene modes also remain available.
        if result["status"] == "DEVELOPMENT_COMPLETED":
            from .evaluate import evaluate, profile
            from .interventions import suite
            from .report import report

            for arm in args.arms or ARMS:
                folder = output / "runs" / arm
                evaluate(c, arm, assets, folder, args.devices[0])
                profile(c, arm, assets, folder, args.devices[0])
                if arm != "M0-TINY":
                    suite(c, arm, assets, folder, args.devices[0])
            result = report(c, output)
        from .resume import export

        if any((output / "runs").glob("*/checkpoint.pt")):
            destination = output / ("portable-" + str(__import__("time").time_ns()))
            result["portable_output"] = str(destination)
            export(output, destination)
        write_json(
            output / "execution-timing.json",
            {
                "end_to_end_seconds": time.perf_counter() - matrix_started,
                "scope": "data preparation, workers, offline evaluation/profile/interventions/report and portable export",
                "source_sha": source_sha(),
                "scientific_decision": False,
            },
        )
    elif args.mode in ("evaluate", "profile", "intervene"):
        from .evaluate import evaluate, profile
        from .interventions import suite

        function = {"evaluate": evaluate, "profile": profile, "intervene": suite}[args.mode]
        result = {
            arm: function(c, arm, args.assets, Path(args.work) / "runs" / arm, args.device) for arm in arms
        }
    elif args.mode == "report":
        from .report import report

        result = report(c, args.work or args.output)
    elif args.mode == "export":
        from .resume import export

        result = export(args.work, args.output)
    else:
        from .resume import restore

        result = restore(args.bundle, args.output)
    # Avoid huge arrays in Notebook console; complete records are in Output.
    if args.mode in ("audit", "params", "search", "freeze", "report", "preflight"):
        print(json.dumps(result, indent=2))
    else:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "status": result.get("status", "COMPLETED"),
                    "output": args.output,
                    "scientific_decision": False,
                }
            )
        )
