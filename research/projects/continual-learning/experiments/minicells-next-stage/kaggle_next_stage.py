"""Publish separate M0 and M1-001 receipts from one pinned checkout."""
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

SOURCE = "https://github.com/ArcheLabs/mini-cells.git"
BRANCH = "codex/minicells-m0-m1"
ROOT = "research/projects/continual-learning/experiments/minicells-next-stage"
APP = "apps/mom-native-cells"
OUTPUT = Path("/kaggle/working")
JOBS = {
    "m0": ("minicells-m0-developer-preview", "m0_acceptance.py", "minicells.m0.developer-preview.v1"),
    "m1": ("minicells-m1-001-value-swap", "m1_value_swap.py", "minicells.m1.value-swap.v1"),
}


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def t4_uuid():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,compute_cap", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    for line in output.splitlines():
        uuid, device, capability = (part.strip() for part in line.split(","))
        if device == "Tesla T4" and capability == "7.5":
            return uuid
    raise RuntimeError("One visible Tesla T4 is required")


def full_suite(repo, out):
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", "tests"]
    with (out / "full-suite.log").open("w", encoding="utf-8") as stream:
        try:
            proc = subprocess.run(command, cwd=repo, text=True, stdout=stream,
                                  stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                  timeout=600)
            receipt = dict(status="PASS" if proc.returncode == 0 else "FAIL", returncode=proc.returncode)
        except subprocess.TimeoutExpired:
            receipt = dict(status="TIMEOUT", timeout_seconds=600)
    (out / "test-receipt.json").write_text(json.dumps(dict(receipt, command=command)) + "\n",
                                          encoding="utf-8")


def main():
    mode = os.getenv("MINICELLS_NEXT_STAGE", "both")
    if mode not in ("m0", "m1", "both"):
        raise ValueError("MINICELLS_NEXT_STAGE must be m0, m1, or both")
    jobs = JOBS if mode == "both" else {mode: JOBS[mode]}
    run_id = uuid4().hex[:12]
    with TemporaryDirectory(prefix="minicells-next-stage-") as temp:
        temp = Path(temp)
        repo = temp / "source"
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "--branch", BRANCH, SOURCE, str(repo), timeout=180)
        # The SDK wheel and both experiment entry points need only these paths.
        run("git", "sparse-checkout", "set", ROOT, APP, "scripts/research", "src",
            cwd=repo, timeout=180)
        sys.path.insert(0, str(repo))
        from scripts.research.publish_experiment_results import (
            ExperimentSpec, load_github_token, prepare_artifacts, push_results,
        )

        load_github_token("GITHUB_TOKEN")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        version_id = os.getenv("KAGGLE_SCRIPT_VERSION_ID")
        setup_error = None
        wheel_hash = None
        try:
            gpu = t4_uuid()
            constraints = temp / "constraints.txt"
            constraints.write_text(f"torch=={version('torch')}\ntriton=={version('triton')}\n",
                                   encoding="utf-8")
            run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                str(repo / APP), "--wheel-dir", str(temp), timeout=180)
            wheel, = temp.glob("minicells_native_sdk-0.1.0-*.whl")
            wheel_hash = sha256(wheel.read_bytes()).hexdigest()
            run(sys.executable, "-m", "pip", "install", "-c", str(constraints), str(wheel), timeout=300)
            run(sys.executable, str(repo / ROOT / "test_next_stage.py"), cwd=temp, timeout=120)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            setup_error = f"{type(exc).__name__}: {exc}"

        failures, publications = [], []
        for key, (name, script, fmt) in jobs.items():
            label = f"{name}-{run_id}"
            out = OUTPUT / label
            action_error = setup_error
            if action_error is None:
                try:
                    if key == "m0":
                        try:
                            # Only M0 needs the full repository test surface. Keep M1 independent.
                            run("git", "sparse-checkout", "set", "tests", "configs", "docs", "shared",
                                "research", "scripts", "apps", "src", "crates", cwd=repo, timeout=240)
                            run("git", "checkout", "--ignore-skip-worktree-bits", "HEAD", "--",
                                "pyproject.toml", cwd=repo, timeout=90)
                            full_suite(repo, out)
                        except (OSError, subprocess.SubprocessError) as exc:
                            out.mkdir(parents=True, exist_ok=True)
                            (out / "test-receipt.json").write_text(json.dumps({
                                "status": "BLOCKED", "reason": str(exc)
                            }) + "\n", encoding="utf-8")
                    run(sys.executable, "-u", str(repo / ROOT / script), str(out), cwd=temp,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, timeout=1800)
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    action_error = f"{type(exc).__name__}: {exc}"
            receipt = out / "decision.json"
            if not receipt.exists():
                out.mkdir(parents=True, exist_ok=True)
                receipt.write_text(json.dumps(dict(format=fmt, status="INVALID", engineering_status="FAIL",
                                                   diagnosis=action_error)) + "\n", encoding="utf-8")
            decision = json.loads(receipt.read_text(encoding="utf-8"))
            decision.update(source_commit=commit, script_version_id=version_id, wheel_sha256=wheel_hash)
            receipt.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
            manifest = out / "manifest.json"
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                data["artifacts_sha256"][receipt.name] = sha256(receipt.read_bytes()).hexdigest()
                manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            if decision["engineering_status"] != "PASS":
                failures.append(label)

            publications.append((name, label, out, fmt, decision["status"]))

        # The shared publisher switches the checkout to a result branch.
        # Finish every requested GPU lane while the candidate still owns HEAD.
        for name, label, out, fmt, status in publications:
            relative = f"results/{label}"
            artifact = f"artifacts/experiments/{name}/{run_id}"
            branch = f"kaggle/{label}"
            try:
                shutil.copytree(out, repo / relative, ignore=shutil.ignore_patterns("work", "*.tmp"))
                run("git", "sparse-checkout", "add", artifact, cwd=repo, timeout=120)
                files = tuple(sorted(path.name for path in (repo / relative).iterdir() if path.is_file()))
                spec = ExperimentSpec(relative, artifact, files, branch, fmt)
                bundle = prepare_artifacts(repo, label, spec, version_id)
                push_results(repo, bundle, label, branch, "GITHUB_TOKEN", "origin/main")
                print(f"Published: {branch} ({status})", flush=True)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                failures.append(label)
                print(f"Publication failed for {label}: {type(exc).__name__}: {exc}", flush=True)
        if failures:
            raise SystemExit(f"Engineering or publication failures: {', '.join(failures)}")


if __name__ == "__main__":
    main()
