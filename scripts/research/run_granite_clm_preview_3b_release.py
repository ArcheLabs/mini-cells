#!/usr/bin/env python3
"""Build, verify, record, and optionally publish Granite-CLM-Preview-3B.

This release wrapper intentionally reuses the already-validated
`scripts/research/moe_conversion_001/run.py` real-model parity runner. It does
not introduce a second conversion implementation.

The 3B release is pinned to the Transformers generation recorded by the
frozen Granite 3.1 3B-A800M checkpoint itself. The release deliberately uses a
single GPU when CUDA is requested: the 3B FP16 model fits on one T4-class GPU,
while the parity workload is dominated by checkpoint materialization, hashing,
and small deterministic inference batches rather than data-parallel throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

DEFAULT_MODEL = "ibm-granite/granite-3.1-3b-a800m-base"
DEFAULT_HF_REPO = "archelabs-org/native-clm-v0"
DEFAULT_HF_SUBDIR = "granite-clm-preview-3b"
DEFAULT_OUTPUT = Path("artifacts/releases/granite-clm-preview-3b")
RUNNER = Path("scripts/research/moe_conversion_001/run.py")
REQUIRED_TRANSFORMERS_VERSION = "4.46.0"
EXECUTION_POLICY = "single_gpu_parity"


def _run(command: list[str], *, log_path: Path | None = None) -> None:
    print("+", " ".join(map(str, command)))
    if log_path is None:
        subprocess.run(list(map(str, command)), check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(map(str, command)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def _require_release_environment(args: argparse.Namespace) -> dict[str, Any]:
    transformers_version = _version("transformers")
    if transformers_version != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Granite-CLM-Preview-3B release requires "
            f"transformers=={REQUIRED_TRANSFORMERS_VERSION}; got {transformers_version!r}. "
            "Use the canonical Kaggle notebook or install the pinned release stack first."
        )

    cuda_devices = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        if args.device not in {"cuda", "cuda:0"}:
            raise RuntimeError(
                "Granite-CLM-Preview-3B release intentionally uses one GPU. "
                "Use --device cuda:0 (or cuda). Multi-GPU sharding is not part of the "
                "validated release path."
            )

    environment = {
        "execution_policy": EXECUTION_POLICY,
        "requested_device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "huggingface_hub_version": _version("huggingface_hub"),
        "accelerate_version": _version("accelerate"),
        "python": sys.version.split()[0],
    }
    print(json.dumps({"release_environment": environment}, indent=2, sort_keys=True))
    return environment


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_release_docs(
    output: Path,
    *,
    metrics: dict[str, Any],
    provenance: dict[str, Any],
    published: bool,
) -> None:
    parity = metrics["parity"]
    gates = metrics["gates"]
    publish_line = (
        f"Published to `{provenance['hf_target']}`."
        if published
        else "Hugging Face publication was not requested for this run."
    )
    preview = [
        "# Granite-CLM-Preview-3B",
        "",
        "Granite-CLM-Preview-3B is a lossless CLM substrate lift of",
        "`ibm-granite/granite-3.1-3b-a800m-base`.",
        "",
        "The upstream Granite MoE execution path and checkpoint bytes are preserved.",
        "The CLM layer adds provenance, tensor identities, and deterministic expert-slice",
        "addresses around the immutable substrate.",
        "",
        "## Preview boundary",
        "",
        "- No training or weight mutation is performed.",
        "- Expert addresses are canonical mutation coordinates only.",
        "- This release does **not** claim that pretrained MoE experts are Cells.",
        "- It does **not** claim safe model evolution, composability, or replay-free learning.",
        "",
        "## Identity",
        "",
        f"- Upstream model: `{provenance['source']['model_id']}`",
        f"- Upstream revision: `{provenance['source']['revision']}`",
        f"- CLM manifest identity: `{provenance['manifest_identity_sha256']}`",
        "",
        "## Release environment",
        "",
        f"- Transformers: `{metrics['environment']['transformers_version']}` (pinned)",
        f"- Execution policy: `{metrics['environment']['execution_policy']}`",
        f"- Requested device: `{metrics['environment']['device']}`",
        f"- Visible CUDA devices: **{metrics['environment']['cuda_device_count']}**",
        "",
        "The release intentionally uses a single GPU even when two GPUs are visible.",
        "The 3B FP16 checkpoint fits on one T4-class GPU, and this parity workload is",
        "primarily checkpoint-loading/hashing plus small deterministic inference batches;",
        "multi-GPU model sharding is therefore kept out of the validated release path.",
        "",
        "## Release gates",
        "",
        f"- Bundle integrity: **{gates['bundle_integrity']}**",
        f"- Checkpoint byte identity: **{gates['checkpoint_byte_identity']}**",
        f"- Forward parity: **{gates['forward_parity']}**",
        f"- Router Top-K identity: **{parity['gates']['router_topk_identity']}**",
        f"- Greedy token identity: **{parity['gates']['greedy_token_identity']}**",
        "",
        f"Release status: **{metrics['status']}**.",
        "",
        "The exact upstream Hugging Face checkpoint is stored under `substrate/` so the",
        "CLM bundle remains independently verifiable with `verify_clm_moe_bundle`.",
        "Machine-readable evidence is in `metrics.json`, `parity_report.json`,",
        "`provenance.json`, and `clm_moe_manifest.json`.",
    ]
    (output / "CLM_PREVIEW.md").write_text("\n".join(preview) + "\n", encoding="utf-8")

    results = [
        "# Granite-CLM-Preview-3B Release",
        "",
        f"Status: **{metrics['status']}**",
        "",
        "> Release-engineering artifact; not a new continual-learning or safe-evolution decision.",
        "",
        "## Source and structure",
        "",
        f"- Source: `{provenance['source']['model_id']}`",
        f"- Revision: `{provenance['source']['revision']}`",
        f"- Hidden layers: **{metrics['model']['num_hidden_layers']}**",
        f"- Local experts: **{metrics['model']['num_local_experts']}**",
        f"- Experts per token: **{metrics['model']['num_experts_per_tok']}**",
        f"- Safetensors bytes: **{metrics['storage']['safetensors_bytes']:,}**",
        f"- Tensor records: **{metrics['conversion']['tensor_count']}**",
        f"- Expert address records: **{metrics['conversion']['expert_address_count']}**",
        "",
        "## Parity",
        "",
        f"- Max absolute logit error: `{parity['max_abs_logit_error']}`",
        f"- Max absolute router error: `{parity['max_abs_router_error']}`",
        f"- Router outputs compared: **{parity['router_outputs_compared']}**",
        f"- Logit argmax identity: **{parity['gates']['logit_argmax_identity']}**",
        f"- Router Top-K identity: **{parity['gates']['router_topk_identity']}**",
        f"- Greedy token identity: **{parity['gates']['greedy_token_identity']}**",
        "",
        "## Runtime",
        "",
        f"- Device: `{metrics['environment']['device']}`",
        f"- GPU: `{metrics['environment']['gpu_name']}`",
        f"- Visible CUDA devices: **{metrics['environment']['cuda_device_count']}**",
        f"- Execution policy: `{metrics['environment']['execution_policy']}`",
        f"- Transformers: `{metrics['environment']['transformers_version']}`",
        f"- Dtype: `{metrics['environment']['dtype']}`",
        f"- End-to-end conversion/parity seconds: `{metrics['runtime']['runner_seconds']:.3f}`",
        "",
        publish_line,
    ]
    (output / "RESULTS.md").write_text("\n".join(results) + "\n", encoding="utf-8")


def _stage_bundle(bundle: Path, output: Path) -> None:
    for name in (
        "metrics.json",
        "parity_report.json",
        "provenance.json",
        "CLM_PREVIEW.md",
        "RESULTS.md",
    ):
        shutil.copy2(output / name, bundle / name)
    shutil.copy2(output / "CLM_PREVIEW.md", bundle / "README.md")


def _publish_bundle(bundle: Path, *, repo_id: str, subdir: str, token: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi(token=token).upload_folder(
        folder_path=bundle,
        path_in_repo=subdir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Granite-CLM-Preview-3B: publish verified CLM bundle ({subdir})",
    )
    return {
        "published": True,
        "repo_id": repo_id,
        "subdir": subdir,
        "commit_oid": getattr(info, "oid", None),
        "commit_url": str(getattr(info, "commit_url", "")) or None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    release_environment = _require_release_environment(args)

    output = args.output_dir.resolve()
    work = args.work_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if work.exists() and args.clean:
        shutil.rmtree(work)
    work.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        RUNNER,
        "--stage",
        "real",
        "--model-id",
        args.model_id,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--tolerance",
        str(args.tolerance),
        "--copy-mode",
        args.copy_mode,
        "--work-dir",
        work,
    ]
    if args.revision:
        command.extend(["--revision", args.revision])
    if args.clean:
        command.append("--clean")
    else:
        command.append("--no-clean")

    started = time.perf_counter()
    runner_log = output / "runner.log"
    try:
        _run(command, log_path=runner_log)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"3B conversion/parity runner exited with {exc.returncode}. "
            f"Full child stdout/stderr is preserved at {runner_log}."
        ) from exc
    runner_seconds = time.perf_counter() - started

    runner_result = _read_json(work / "result.json")
    if runner_result.get("status") != "PASS":
        raise RuntimeError("3B conversion/parity runner did not PASS; refusing release")

    bundle = work / "clm-bundle"
    manifest = _read_json(bundle / "clm_moe_manifest.json")
    parity = runner_result["forward"]
    gates = runner_result["gates"]
    safetensors_bytes = sum(
        int(item["bytes"])
        for item in manifest["files"]
        if str(item["path"]).endswith(".safetensors")
    )
    snapshot_bytes = sum(int(item["bytes"]) for item in manifest["files"])

    metrics = {
        "release": "Granite-CLM-Preview-3B",
        "status": runner_result["status"],
        "gates": gates,
        "model": manifest["config"],
        "storage": {
            "snapshot_bytes": snapshot_bytes,
            "safetensors_bytes": safetensors_bytes,
            "source_file_count": len(manifest["files"]),
        },
        "conversion": {
            "schema_version": manifest["schema_version"],
            "kind": manifest["conversion"]["kind"],
            "execution_semantics": manifest["conversion"]["execution_semantics"],
            "tensor_count": runner_result["tensor_count"],
            "expert_address_count": runner_result["expert_address_count"],
            "manifest_identity_sha256": runner_result["manifest_identity_sha256"],
        },
        "parity": parity,
        "environment": {
            **release_environment,
            "device": runner_result["device"],
            "dtype": runner_result["dtype"],
            "gpu_name": (
                torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
            ),
        },
        "runtime": {"runner_seconds": runner_seconds},
    }
    provenance = {
        "release": "Granite-CLM-Preview-3B",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_source_commit": _git_head(),
        "source": {
            "model_id": runner_result["source_model_id"],
            "requested_revision": args.revision,
            "revision": runner_result["source_revision"],
        },
        "hf_target": f"{args.hf_repo}/{args.hf_subdir}",
        "manifest_identity_sha256": runner_result["manifest_identity_sha256"],
        "conversion_contract": manifest["conversion"],
        "release_environment_contract": {
            "transformers_version": REQUIRED_TRANSFORMERS_VERSION,
            "execution_policy": EXECUTION_POLICY,
            "multi_gpu_sharding": False,
        },
        "claim_boundary": {
            "preserves_pretrained_behavior": True,
            "expert_addresses_are_mutation_coordinates": True,
            "claims_expert_is_cell": False,
            "claims_safe_model_evolution": False,
            "claims_replay_free_learning": False,
        },
    }

    _write_json(output / "clm_moe_manifest.json", manifest)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "parity_report.json", parity)
    _write_json(output / "provenance.json", provenance)
    _write_release_docs(output, metrics=metrics, provenance=provenance, published=False)

    hf_publish: dict[str, Any] | None = None
    if args.publish_hf:
        token = os.environ.get(args.hf_token_env)
        if not token:
            raise RuntimeError(f"missing environment variable {args.hf_token_env}")
        _stage_bundle(bundle, output)
        hf_publish = _publish_bundle(
            bundle,
            repo_id=args.hf_repo,
            subdir=args.hf_subdir,
            token=token,
        )
        _write_json(output / "hf_publish.json", hf_publish)
        _write_release_docs(output, metrics=metrics, provenance=provenance, published=True)

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        refresh = {
            "README.md": output / "CLM_PREVIEW.md",
            "CLM_PREVIEW.md": output / "CLM_PREVIEW.md",
            "RESULTS.md": output / "RESULTS.md",
            "hf_publish.json": output / "hf_publish.json",
        }
        for remote_name, local_path in refresh.items():
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"{args.hf_subdir}/{remote_name}",
                repo_id=args.hf_repo,
                repo_type="model",
                commit_message="Granite-CLM-Preview-3B: record release receipt",
            )

    result = {
        "status": runner_result["status"],
        "source_revision": runner_result["source_revision"],
        "manifest_identity_sha256": runner_result["manifest_identity_sha256"],
        "hf_publish": hf_publish,
        "metrics_path": str(output / "metrics.json"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Granite-CLM-Preview-3B lossless lift, metrics, and Hugging Face release"
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-subdir", default=DEFAULT_HF_SUBDIR)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("MINICELLS_RELEASE_WORK_DIR", "/tmp/granite-clm-preview-3b")),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="hardlink")
    parser.add_argument("--publish-hf", action="store_true")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    outcome = run(parse_args())
    raise SystemExit(0 if outcome["status"] == "PASS" else 1)
