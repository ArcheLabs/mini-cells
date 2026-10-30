import argparse
import json
import platform
import uuid
from pathlib import Path

from .common import BASE, ROOT, digest, source, write


def engineering(args):
    import torch

    from .analysis import analyze
    from .assets import inventory
    from .benchmark import event_matrix, incremental_timings
    from .incremental import PreallocatedIncrementalLM
    from .validation import engineering_model, parity

    root = args.output / uuid.uuid4().hex
    root.mkdir(parents=True)
    config = json.loads((BASE / "protocol.json").read_text())
    matrix = json.loads((BASE / "workloads.json").read_text())
    inputs = {
        str(p.relative_to(ROOT)): digest(p) for p in sorted((BASE / "implementation").rglob("*.py"))
    }
    inputs.update(
        {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                BASE / "protocol.json",
                BASE / "workloads.json",
                BASE / "PROTOCOL_DRAFT.md",
                BASE / "validation_policy.json",
                BASE / "validation_inputs.json",
            )
        }
    )
    manifest = {
        "experiment_id": "EPCLM-001",
        "run_id": root.name,
        "source_sha": source(),
        "config_sha256": digest(BASE / "protocol.json"),
        "scope": "NATIVE_1P9M_SCOPE_CPU_ENGINEERING_ONLY",
        "optimizer_updates": 0,
        "scientific_optimizer_updates": 0,
        "gpu_used": False,
        "confirmation_materialized": False,
        "run_status": "RUNNING",
        "execution_inputs": inputs,
    }
    write(root / "RUN_MANIFEST.json", manifest)
    if args.publish_github:
        from .publication import preflight

        preflight(root)
    try:
        torch.set_num_threads(1)
        assets = inventory(args.assets)
        write(root / "baseline_assets.json", assets)
        results = {}
        for kind in ("X3", "X4", "T1"):
            model = engineering_model(kind)
            ids = torch.randint(0, 32, (1, 128), generator=torch.Generator().manual_seed(1832))
            results[kind] = parity(
                model,
                ids,
                atol=config["engineering_tolerance"]["atol"],
                rtol=config["engineering_tolerance"]["rtol"],
            )
        results["X3_A1_PREALLOCATED"] = parity(
            engineering_model("X3"),
            ids,
            atol=config["engineering_tolerance"]["atol"],
            rtol=config["engineering_tolerance"]["rtol"],
            adapter_class=PreallocatedIncrementalLM,
        )
        write(
            root / "PARITY.json",
            {"scope": "TINY_RANDOM_WEIGHT_FIXTURES_NOT_TRAINED_BASELINES", "models": results},
        )
        if not all(r["passed"] and r["model_unchanged"] for r in results.values()):
            raise RuntimeError("INCREMENTAL_PARITY_FAILED")
        rows = event_matrix(matrix)
        write(
            root / "EVENT_RESULTS.json",
            {"scope": "ENGINEERING_NOT_SCIENTIFIC_PERFORMANCE", "records": rows},
        )
        model = engineering_model("X3")
        write(
            root / "INCREMENTAL_TIMING.json",
            {
                "scope": "TINY_RANDOM_WEIGHT_CPU_SMOKE",
                "records": incremental_timings(
                    model, ids[:, :16], 8, repeats=matrix["repeats"], warmup=matrix["warmup"]
                ),
            },
        )
        plots = analyze(root)
        decisions = {
            k: {"status": "INCOMPLETE", "reason": reason}
            for k, reason in (
                (
                    "H1_INCREMENTAL",
                    "Trained E0-B weights and frozen hardware/precision evidence are missing",
                ),
                (
                    "H2_EVENT_RUNTIME",
                    "Executor parity only; held-out frozen workload/hardware confirmation missing",
                ),
                (
                    "H3_COMPETITIVE",
                    "Bound trained T1/CLM weights, shared quality corpus and competitive measurements missing",
                ),
            )
        }
        write(root / "DECISIONS.json", decisions)
        write(
            root / "RESOURCE_MANIFEST.json",
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "threads": 1,
                "cuda_available": torch.cuda.is_available(),
                "gpu_benchmark": "NOT_RUN",
                "measured_dram_bytes": None,
                "energy_joules": None,
                "kernel_count": None,
                "measurement_status": "UNAVAILABLE",
                "profiler": "NOT_RUN",
                "plots": plots,
                "limitations": [
                    "A1 preallocated cache implemented; hardware performance unvalidated",
                    "Padding masks explicitly unsupported",
                    "30M deferred and excluded from all round-1 execution",
                    "Performance uses boundary timing only; detailed phase timings are from the separate correctness pass",
                ],
            },
        )
        (root / "SUMMARY.md").write_text(
            "# EPCLM-001 engineering result\n\n"
            "Reference X3/X4/T1 incremental parity and four event executors passed on tiny "
            "CPU fixtures. These are randomly initialized engineering models, not recovered E0 models. "
            "No training occurred. H1/H2/H3 remain INCOMPLETE. See baseline_assets.json for asset gaps. "
            "GPU, holdout performance and competitive quality/latency remain unexecuted.\n"
        )
        status = "ENGINEERING_PASS_SCIENTIFIC_INCOMPLETE"
    except Exception as error:
        status = "FAILED_LOCAL_EVIDENCE_RETAINED"
        manifest["failure_type"] = type(error).__name__
        raise
    finally:
        manifest = dict(
            json.loads((root / "RUN_MANIFEST.json").read_text()),
            run_status=locals().get("status", "FAILED_LOCAL_EVIDENCE_RETAINED"),
        )
        manifest["artifact_files"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in root.iterdir()
            if p.is_file() and p.name != "RUN_MANIFEST.json"
        }
        write(root / "RUN_MANIFEST.json", manifest)
        if args.publish_github:
            from .publication import publish

            publish(root)
    return {"status": status, "output": str(root), "optimizer_updates": 0, "gpu_used": False}


def main():
    parser = argparse.ArgumentParser(description="EPCLM-001 inference only; never trains models")
    parser.add_argument(
        "command",
        choices=(
            "engineering",
            "audit-assets",
            "publish-existing",
            "bench-event",
            "bench-incremental",
            "bench-competitive",
            "analyze",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, action="append", default=[])
    parser.add_argument("--publish-github", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--matrix", type=Path, default=BASE / "workloads.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", choices=("E0B_T1", "E0B_X3", "E0B_X4"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "optimizer_updates": 0,
                    "gpu_started": False,
                    "status": "DRY_RUN_VALID",
                    "source_sha": source(),
                }
            )
        )
        return
    if args.command == "engineering":
        result = engineering(args)
    elif args.command == "audit-assets":
        from .assets import inventory

        result = inventory(args.assets)
        write(args.output / "baseline_assets.json", result)
    elif args.command == "publish-existing":
        from .publication import publish

        roots = (
            [args.output]
            if (args.output / "RUN_MANIFEST.json").exists()
            else sorted(p.parent for p in args.output.glob("*/RUN_MANIFEST.json"))
        )
        if not roots:
            parser.error("No completed or failed result manifests found")
        result = [publish(root) for root in roots]
    elif args.command == "analyze":
        from .analysis import analyze

        result = analyze(args.output)
    elif args.command == "bench-event":
        import torch

        from .benchmark import event_matrix

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            parser.error("CUDA_UNAVAILABLE_NO_GPU_PASS")
        matrix = json.loads(args.matrix.read_text())
        result = {
            "scope": "DEVELOPMENT_NOT_FROZEN_SCIENTIFIC",
            "records": event_matrix(matrix, device=args.device),
        }
        write(args.output / "EVENT_RESULTS.json", result)
    elif args.command == "bench-incremental":
        if args.checkpoint is None or args.model is None:
            parser.error(
                "--checkpoint and --model are required; asset audit must establish identity"
            )
        import torch

        from .assets import load_native
        from .benchmark import incremental_timings
        from .validation import parity

        policy = json.loads((BASE / "validation_policy.json").read_text())
        inputs = json.loads((BASE / policy["input_file"]).read_text())
        model, metadata = load_native(args.checkpoint, args.model)
        model.to(args.device)
        dtype = str(next(model.parameters()).dtype).split(".")[-1]
        key = str(next(model.parameters()).device.type) + ":" + dtype
        if key not in policy["criteria"]:
            parser.error("DEVICE_PRECISION_TOLERANCE_NOT_FROZEN")
        ids = torch.tensor(inputs["tokens"], dtype=torch.long, device=args.device)
        if ids.shape[1] != policy["required_length"] or ids.shape[1] != 128:
            parser.error("FULL_BOUNDARY_VALIDATION_INPUT_REQUIRED")
        criterion = policy["criteria"][key]
        check = parity(model, ids, **criterion, checkpoint_sha256=digest(args.checkpoint))
        check.update(
            validation_policy_sha256=digest(BASE / "validation_policy.json"),
            validation_input_sha256=digest(BASE / policy["input_file"]),
            input_scope=inputs["scope"],
            checkpoint_metadata=metadata,
        )
        write(args.output / "PARITY.json", check)
        if not check["passed"]:
            parser.error("PARITY_FAILED_NO_PERFORMANCE_RESULT")
        result = {
            "scope": "DEVELOPMENT_NOT_FROZEN_SCIENTIFIC",
            "source_sha": source(),
            "checkpoint_sha256": digest(args.checkpoint),
            "correctness": check,
            "records": incremental_timings(model, ids, 32),
        }
        write(args.output / "INCREMENTAL_TIMING.json", result)
    else:
        parser.error("H3_BLOCKED_BOUND_T1_CLM_WEIGHTS_AND_SHARED_QUALITY_CORPUS_REQUIRED")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
