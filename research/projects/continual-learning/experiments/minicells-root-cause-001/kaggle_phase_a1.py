"""Single Kaggle Python Script entry for the Phase A.1 component diagnostic."""
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

SOURCE = "https://github.com/ArcheLabs/mini-cells.git"
BRANCH = "codex/minicells-root-cause-001"
APP = "apps/mom-native-cells"
NAME = "minicells-root-cause-001"
EXPERIMENT = f"research/projects/continual-learning/experiments/{NAME}"
FORMAT = "minicells.root-cause.001.phase-a1.v1"


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def main():
    from importlib.metadata import version

    retry_id = os.getenv("MINICELLS_ROOT_CAUSE_A1_PUBLISH_ID")
    if retry_id and not re.fullmatch(r"[0-9a-f]{12}", retry_id):
        raise ValueError("Invalid publication retry ID")
    name = f"{NAME}-a1-{retry_id or uuid4().hex[:12]}"
    output = Path("/kaggle/working") / name
    receipt = output / "decision.json"
    with TemporaryDirectory(prefix="minicells-root-cause-") as temp:
        repo = Path(temp) / "source"
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "--branch", BRANCH, SOURCE, str(repo), timeout=180)
        if retry_id:
            original = json.loads(receipt.read_text(encoding="utf-8"))["source_commit"]
            run("git", "fetch", "--depth=1", "origin", original, cwd=repo, timeout=180)
            run("git", "checkout", "--detach", "FETCH_HEAD", cwd=repo, timeout=120)
        run("git", "sparse-checkout", "set", "scripts/research", APP, EXPERIMENT, "src",
            cwd=repo, timeout=120)
        sys.path[:0] = [str(repo), str(repo / APP)]
        from kaggle_t4_1b_capacity import t4_uuid
        from scripts.research.publish_experiment_results import (
            ExperimentSpec, load_github_token, prepare_artifacts, push_results,
        )

        load_github_token("GITHUB_TOKEN")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        version_id = os.getenv("KAGGLE_SCRIPT_VERSION_ID")
        if not retry_id:
            try:
                constraints = Path(temp) / "constraints.txt"
                constraints.write_text(f"torch=={version('torch')}\ntriton=={version('triton')}\n")
                run("git", "checkout", "--ignore-skip-worktree-bits", "HEAD", "--",
                    "pyproject.toml", "README.md", "LICENSE", cwd=repo, timeout=120)
                run(sys.executable, "-m", "pip", "install", "-c", str(constraints),
                    str(repo) + "[mom]", timeout=240)
                visible_gpu = t4_uuid()
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible_gpu}
                run(sys.executable, "-u", str(repo / EXPERIMENT / "run_a1.py"),
                    "--output", str(output), timeout=900, env=env)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                if not receipt.exists():
                    output.mkdir(parents=True, exist_ok=True)
                    receipt.write_text(json.dumps(dict(format=FORMAT, status="INVALID",
                        engineering_status="FAIL", diagnosis=f"{type(exc).__name__}: {exc}")) + "\n",
                        encoding="utf-8")
            evidence = json.loads(receipt.read_text(encoding="utf-8"))
            evidence.update(source_commit=commit, script_version_id=version_id)
            receipt.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["artifacts_sha256"][receipt.name] = hashlib.sha256(receipt.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        else:
            evidence = json.loads(receipt.read_text(encoding="utf-8"))
            version_id = evidence.get("script_version_id")

        relative = f"results/{name}"
        artifact = f"artifacts/experiments/{NAME}/{name.removeprefix(NAME + '-')}"
        branch = f"kaggle/{name}"
        shutil.copytree(output, repo / relative, ignore=shutil.ignore_patterns("work", "*.tmp"))
        run("git", "sparse-checkout", "add", artifact, cwd=repo, timeout=120)
        files = tuple(sorted(p.name for p in (repo / relative).iterdir() if p.is_file()))
        spec = ExperimentSpec(relative, artifact, files, branch, FORMAT)
        bundle = prepare_artifacts(repo, name, spec, version_id)
        push_results(repo, bundle, name, branch, "GITHUB_TOKEN", "origin/main")
        print(f"Published: {branch} ({evidence['status']})", flush=True)
        if evidence["engineering_status"] != "PASS":
            raise SystemExit("Phase A.1 was invalid; see the published report.")


if __name__ == "__main__":
    main()
