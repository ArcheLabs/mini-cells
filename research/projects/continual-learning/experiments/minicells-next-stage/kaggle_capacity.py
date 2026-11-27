"""Run M1-002 on one Tesla T4 and publish its independent receipt."""
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
NAME = "minicells-m1-002-capacity"
FORMAT = "minicells.m1.capacity.v1"
REQUIRED = {"cases.jsonl", "dataset.json", "decision.json", "manifest.json",
            "report.md", "result.json", "statistics.json"}


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


def main():
    run_id = uuid4().hex[:12]
    label = f"{NAME}-{run_id}"
    out = OUTPUT / label
    with TemporaryDirectory(prefix=f"{NAME}-") as temp:
        temp = Path(temp)
        repo = temp / "source"
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "--branch", BRANCH, SOURCE, str(repo), timeout=180)
        run("git", "sparse-checkout", "set", ROOT, APP, "scripts/research",
            cwd=repo, timeout=180)
        sys.path.insert(0, str(repo))
        from scripts.research.publish_experiment_results import (
            ExperimentSpec, load_github_token, prepare_artifacts, push_results,
        )

        load_github_token("GITHUB_TOKEN")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        version_id = os.getenv("KAGGLE_SCRIPT_VERSION_ID")
        wheel_hash = None
        action_error = None
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
            run(sys.executable, str(repo / ROOT / "test_capacity.py"), cwd=temp,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=120)
            run(sys.executable, "-u", str(repo / ROOT / "m1_capacity.py"), str(out), cwd=temp,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, timeout=7200)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            action_error = f"{type(exc).__name__}: {exc}"

        receipt = out / "decision.json"
        if receipt.exists():
            decision = json.loads(receipt.read_text(encoding="utf-8"))
        else:
            out.mkdir(parents=True, exist_ok=True)
            decision = dict(format=FORMAT, status="INVALID", engineering_status="FAIL",
                            diagnosis=action_error or "Missing experiment decision")
        if decision["engineering_status"] == "PASS":
            missing = sorted(name for name in REQUIRED if not (out / name).is_file())
            if missing:
                decision.update(status="INVALID", engineering_status="FAIL",
                                diagnosis=f"Missing artifacts: {', '.join(missing)}")
        decision.update(source_commit=commit, script_version_id=version_id, wheel_sha256=wheel_hash)
        receipt.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
        manifest = out / "manifest.json"
        digest = sha256(receipt.read_bytes()).hexdigest()
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
        else:
            data = {"artifacts_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                        for p in out.iterdir() if p.is_file()}}
        data["artifacts_sha256"][receipt.name] = digest
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

        relative = f"results/{label}"
        artifact = f"artifacts/experiments/{NAME}/{run_id}"
        branch = f"kaggle/{label}"
        shutil.copytree(out, repo / relative, ignore=shutil.ignore_patterns("work", "*.tmp"))
        run("git", "sparse-checkout", "add", artifact, cwd=repo, timeout=120)
        files = tuple(sorted(path.name for path in (repo / relative).iterdir() if path.is_file()))
        spec = ExperimentSpec(relative, artifact, files, branch, FORMAT)
        bundle = prepare_artifacts(repo, label, spec, version_id)
        push_results(repo, bundle, label, branch, "GITHUB_TOKEN", "origin/main")
        print(f"Published: {branch} ({decision['status']})", flush=True)
        if action_error or decision["engineering_status"] != "PASS":
            raise SystemExit("M1-002 engineering check failed. See the published decision.")


if __name__ == "__main__":
    main()
