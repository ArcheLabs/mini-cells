"""Canonical CPU-only default; every scientific entry is gated before reading runs."""

import argparse
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import torch
from exp006.reuse import file_sha256, write_json
from .configs import PARENT, ROOT, ARMS, load, arm_config, count, blockers, authorize
from .models import make_model
from .losses import joint_loss, sample_paths
from .accounting import Counter
from .diagnostics import routing


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def fixture_config(c, arm):
    m = arm_config(c, arm)
    m.update(
        width=12,
        heads=3,
        cell_count=2,
        cell_hidden=24,
        vocab_size=23,
        max_sequence_length=8,
        router_hidden=8,
        step_width=4,
        edge_rank=3,
        moe_layers=2,
        moe_experts=2,
        moe_hidden=16,
    )
    return m


def dry_run(c, arm, full_size=False):
    torch.set_num_threads(1)
    if full_size:
        print("CPU full-size joint fixture: " + arm, flush=True)
    model = (
        make_model(arm_config(c, arm) if full_size else fixture_config(c, arm), arm, seed=17).cpu().train()
    )
    x = torch.tensor([[1, 2, 3]])
    y = torch.tensor([[2, 3, 4]])
    mechanism = c["arms"][arm]["mechanism"]
    generators = [
        torch.Generator().manual_seed(7100 + i) for i in range(1 if arm == "M0" or mechanism == "J0" else 2)
    ]
    with Counter(model) as counter:
        outputs = sample_paths(model, x, generators, mechanism, capture=True)
        loss, log = joint_loss(outputs, y, c["training"], mechanism if arm != "M0" else "J0")
        loss.backward()
    if not torch.isfinite(loss) or any(
        p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
    ):
        raise FloatingPointError("fixture loss or gradient nonfinite")
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    opt.step()
    return {
        "arm": arm,
        "status": "CPU_FIXTURE_PASS",
        "seed": 17,
        "scientific_decision": False,
        "loss": float(loss.detach()),
        "gradient_norm": norm,
        "diagnostics": {k: float(v) for k, v in log.items()},
        "routing": routing(outputs[0], model.config["cell_count"]),
        "cost": counter.report(x.numel()),
        "parameter_fixture_count": sum(p.numel() for p in model.parameters()),
        "full_size": full_size,
        "gpu_runs": 0,
    }


def audit(c, p):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", head, "--", str(PARENT.relative_to(ROOT))], cwd=ROOT, text=True
    )
    protected = [line for line in tree.splitlines() if "/extension-parameter-depth/" not in line]
    return {
        "experiment_id": "006-PD1",
        "head": head,
        "parent_protocol": json.loads((PARENT / "protocol.json").read_text()),
        "parent_config_sha": file_sha256(PARENT / "config/proposed.json"),
        "protected_parent_git_tree": protected,
        "historical_seeds": "not consumed; collision audit must precede registration",
        "reuse": [
            "exp006.baseline.MoELM",
            "exp006.memory.InputMemory",
            "exp006.cells",
            "exp006.model.initialize",
            "exp006.reuse checkpoint and atomic JSON",
            "exp006.data story/window checks",
            "exp005 ObservedGemmFlops",
        ],
        "gate_blockers": blockers(c, p),
        "gpu_runs": 0,
        "confirmation_read": False,
        "scientific_decision": False,
    }


def scientific(c, p, args):
    authorize(
        c,
        p,
        args.arm,
        args.seed,
        args.role,
        "evaluate" if args.mode == "intervene" else args.mode,
        args.source_commit,
        args.device,
    )
    if args.mode == "train":
        from .training import train

        return train(
            c,
            p,
            args.arm,
            args.seed,
            args.role,
            args.source_commit,
            args.assets,
            args.output,
            args.device,
            args.resume,
            args.confirmation_assets,
        )
    if args.mode == "publish":
        from .publication import publish

        return publish(
            c,
            p,
            args.arm,
            args.seed,
            args.role,
            args.source_commit,
            args.run,
            Path(args.output) / "publication-checkout",
        )
    if not args.run:
        raise ValueError("--run required")
    from .training import identity
    from .assets import load_bundle
    from .evaluation import evaluate_windows, intervention_suite
    from .profile import benchmark
    from exp006.reuse import make_batch

    run = Path(args.run)
    payload = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    if payload["step"] != c["training"]["steps"]:
        raise RuntimeError("only frozen final checkpoints may be evaluated")
    assets = load_bundle(
        args.assets,
        c,
        p,
        args.arm,
        args.seed,
        args.role,
        args.source_commit,
        evaluation=True,
        frozen_checkpoint={
            "sha": file_sha256(run / "checkpoint.pt"),
            "step": payload["step"],
            "complete": True,
        },
    )
    expected = identity(
        c,
        p,
        args.arm,
        args.seed,
        args.role,
        args.source_commit,
        {"manifest_sha256": c["data"]["manifest_sha256"], "tokenizer_sha256": assets["tokenizer_sha256"]},
    )
    if payload["identity"] != expected:
        raise ValueError("checkpoint identity differs from registration")
    model = make_model(arm_config(c, args.arm), args.arm, args.seed).to(args.device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    if args.mode == "evaluate":
        return evaluate_windows(
            model, assets["tokens"], assets["starts"], assets["groups"], c["training"]["sequence_length"]
        )
    x, y = make_batch(
        assets["tokens"],
        np.asarray(assets["starts"][:1]),
        c["training"]["sequence_length"],
        torch.device(args.device),
    )
    if args.mode == "intervene":
        return intervention_suite(
            model,
            assets["tokens"],
            assets["starts"],
            assets["groups"],
            c["training"]["sequence_length"],
            args.intervention,
            step=args.intervention_step,
            seed=c["evaluation"]["intervention_seed"],
            replicates=c["evaluation"]["bootstrap_replicates"],
        )
    rows = []
    for batch in c["evaluation"]["batch_sizes"]:
        prefix = x[:, : min(c["evaluation"]["decode_prefix"], x.shape[1] - 1)].repeat(batch, 1)
        rows.append(
            {
                "batch_size": batch,
                "profile": benchmark(
                    model, prefix, x[:, -1].repeat(batch), repeats=c["evaluation"]["decode_repetitions"]
                ),
            }
        )
    return {
        "profiles": rows,
        "checkpoint_sha": file_sha256(run / "checkpoint.pt"),
        "scientific_decision": False,
    }


def preflight(c, p):
    print("CPU Gate 0: actual parameter inventories and synthetic joint steps for all arms", flush=True)
    counts = count(c)
    if not all(r["ratio_pass"] for r in counts.values()):
        raise ValueError("parameter ratio gate failed")
    result = {
        "status": "ENGINEERING_READY",
        "scientific_decision": False,
        "gpu_runs": 0,
        "parameters": counts,
        "fixtures": [dry_run(c, arm) for arm in ARMS],
        "full_size_single_step_fixtures": [dry_run(c, arm, full_size=True) for arm in ARMS],
        "gate_state": p["status"],
        "unregistered_fields": blockers(c, p),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "note": "tiny nonresearch seed fixtures, not a language experiment",
    }
    return result


def automatic(precheck, approve, execute):
    print("[1/3] Automatic CPU preflight", flush=True)
    result = precheck()
    if result.get("status") != "ENGINEERING_READY":
        raise RuntimeError("Automatic preflight failed; experiment not started")
    print("[2/3] Automatic assets and return-path check", flush=True)
    approve()
    print("[3/3] Running development experiment", flush=True)
    return execute()


def auto_run(c, p, args):
    from .orchestration import matrix
    from .development import configure
    from .publication import preflight_development, publish
    import os
    import time

    source = (
        args.source_commit
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    output = Path(args.output)
    state = {}

    def precheck():
        result = preflight(c, p)
        write(output / "automatic-preflight.json", result)
        return result

    def approve():
        # Existing Kaggle secrets; no new secret labels or token values are logged.
        for name in ("HF_TOKEN", "GITHUB_TOKEN"):
            if os.environ.get(name, "").strip():
                continue
            try:
                from kaggle_secrets import UserSecretsClient

                token = UserSecretsClient().get_secret(name).strip()
                if token:
                    os.environ[name] = token
            except Exception:
                if name == "GITHUB_TOKEN":
                    raise RuntimeError(
                        "Enable the existing GITHUB_TOKEN in this Kaggle notebook; training has not started"
                    ) from None
        if not os.environ.get("GITHUB_TOKEN", "").strip():
            raise RuntimeError("Existing GITHUB_TOKEN is unavailable; training has not started")
        runtime, protocol, config_path, assets, seed = configure(c, p, output, source, args.seed)
        state.update(c=runtime, p=protocol, path=config_path, assets=assets, seed=seed)
        preflight_development(
            runtime, protocol, seed, source, output / ("publication-preflight-" + str(time.time_ns()))
        )

    def execute():
        runtime, protocol = state["c"], state["p"]
        devices = args.devices
        if devices == ["auto"]:
            devices = (
                ["cuda:0", "cuda:1"]
                if torch.cuda.device_count() >= 2
                else ["cuda:0"]
                if torch.cuda.is_available()
                else ["cpu"]
            )
        runs = matrix(
            runtime,
            protocol,
            args.arms,
            state["seed"],
            "development",
            source,
            state["assets"],
            output / "runs",
            state["path"],
            devices,
        )
        # All workers have joined. Profiling cannot overlap another arm's training.
        for run in runs["runs"]:
            folder = Path(run["run"])
            if not (folder / "hardware-profile.json").exists():
                profile_args = argparse.Namespace(
                    mode="profile",
                    arm=run["arm"],
                    seed=state["seed"],
                    role="development",
                    source_commit=source,
                    device=run["device"],
                    run=str(folder),
                    assets=str(state["assets"]),
                )
                write(folder / "hardware-profile.json", scientific(runtime, protocol, profile_args))
        manifest = {
            "experiment_id": "006-PD1",
            "metrics": [str(Path(r["run"]).relative_to(output) / "metrics.json") for r in runs["runs"]],
        }
        write(output / "analysis-manifest.json", manifest)
        from .report import report

        write(output / "analysis.json", report(output / "analysis-manifest.json", runtime))
        for run in runs["runs"]:
            import shutil

            folder = Path(run["run"])
            write(folder / "run-config.json", runtime)
            write(folder / "run-protocol.json", protocol)
            for name in ("manifest.json", "windows.json"):
                shutil.copy2(state["assets"] / name, folder / ("assets-" + name))
            print("[Publish] " + run["arm"], flush=True)
            publish(
                runtime,
                protocol,
                run["arm"],
                state["seed"],
                "development",
                source,
                run["run"],
                output / ("publication-" + run["arm"] + "-" + str(time.time_ns())),
            )
        return {
            **runs,
            "status": "DEVELOPMENT_COMPLETED",
            "automatic_assets": str(state["assets"]),
            "confirmation_access": False,
            "scientific_decision": False,
        }

    try:
        return automatic(precheck, approve, execute)
    except Exception as exc:
        write(
            output / "automatic-failure.json",
            {"status": "FAILED", "reason": str(exc), "scientific_decision": False, "results_preserved": True},
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "auto",
            "audit",
            "params",
            "preflight",
            "dry-run",
            "train-matrix",
            "train",
            "evaluate",
            "profile",
            "report",
            "publish",
            "confirmation",
            "intervene",
        ),
        default="preflight",
    )
    parser.add_argument("--config")
    parser.add_argument("--arm", choices=ARMS, default="P50-A32")
    parser.add_argument("--all-arms", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["M0", "P50-F16", "P50-F32", "P50-A32"])
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source-commit")
    parser.add_argument("--role", choices=("development", "confirmation"), default="development")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--assets")
    parser.add_argument("--confirmation-assets")
    parser.add_argument("--output", default="pd001-cpu-evidence")
    parser.add_argument("--run")
    parser.add_argument("--manifest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--intervention", default="stop_early")
    parser.add_argument("--intervention-step", type=int, default=1)
    args = parser.parse_args()
    c, p = load(args.config)
    if args.mode in ("audit", "params", "preflight", "dry-run") and args.device != "cpu":
        raise RuntimeError("engineering modes are CPU only; CUDA synthetic preflight needs separate approval")
    if args.mode == "auto":
        result = auto_run(c, p, args)
    elif args.mode == "audit":
        result = audit(c, p)
    elif args.mode == "params":
        result = count(c)
    elif args.mode == "dry-run":
        result = dry_run(c, args.arm)
    elif args.mode == "preflight":
        result = preflight(c, p)
    elif args.mode == "train-matrix":
        from .orchestration import matrix

        result = matrix(
            c,
            p,
            args.arms,
            args.seed,
            args.role,
            args.source_commit,
            args.assets,
            args.output,
            args.config,
            args.devices,
        )
    elif args.mode == "report":
        from .report import report

        result = report(args.manifest, c)
    else:
        if args.mode in ("confirmation", "intervene"):
            if args.mode == "confirmation":
                args.role = "confirmation"
            authorize(
                c,
                p,
                args.arm,
                args.seed,
                args.role,
                "confirmation" if args.mode == "confirmation" else "evaluate",
                args.source_commit,
                args.device,
            )
            if args.mode == "confirmation":
                args.mode = "evaluate"
                args.role = "confirmation"
        result = scientific(c, p, args)
    destination = Path(args.output)
    if destination.suffix != ".json":
        destination = destination / (args.mode + ".json")
    if destination.exists() and args.mode not in ("train", "auto"):
        raise FileExistsError("evidence exists; choose a fresh output directory")
    write(destination, result)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": result.get("status", "RECORDED"),
                "output": str(destination),
                "gpu_runs": 0 if args.mode in ("preflight", "dry-run", "audit", "params") else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
