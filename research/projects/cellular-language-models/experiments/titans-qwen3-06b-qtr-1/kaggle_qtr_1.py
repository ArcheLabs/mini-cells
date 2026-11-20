"""Kaggle Python script. Enable Internet and select Tesla T4."""
from pathlib import Path
import json
import os
import shutil
import uuid
import subprocess
import sys

SOURCE = "https://github.com/ArcheLabs/mini-cells.git"
BRANCH = "codex/qtr-1"
MODE = "engineering"
ENGINEERING_RESULT_COMMIT = ""
ENGINEERING_RESULT_RUN = ""
ARM = "E0"
SEED = 42
STOP_AFTER = 100
RESUME_ASSET_ID = None
RESUME_CHECKPOINT_SHA256 = ""
SCRIPT_VERSION_ID = ""
UPSTREAM = "https://github.com/sarahpannn/titans-pytorch.git"
REVISION = "c60b4efaa3a3c6df4deebd57cbccf75947b87cad"


def run(*args, cwd=None, timeout=None):
    subprocess.run(args, cwd=cwd, check=True, timeout=timeout)


def main():
    script_version = os.getenv("KAGGLE_SCRIPT_VERSION_ID") or SCRIPT_VERSION_ID
    if not (script_version.isdecimal() and int(script_version) > 0):
        script_version = None
        print("KAGGLE_SCRIPT_VERSION_ID unavailable; recording exact Git source commit", flush=True)
    if script_version:
        os.environ["KAGGLE_SCRIPT_VERSION_ID"] = script_version
    os.environ["PYTHONUNBUFFERED"] = "1"
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="p14-", dir="/kaggle/working"))
    repo, upstream = root / "layer-location-source", root / "layer-location-titans"
    print("SOURCE_FETCH_START", flush=True)
    run("git", "clone", "--progress", "--depth", "1", "--single-branch",
        "--filter=blob:none", "--sparse", "--branch", BRANCH, SOURCE, str(repo),
        timeout=180)
    run("git", "sparse-checkout", "set", "scripts/research",
        "research/titans-qwen3-06b/gate1", "research/titans-qwen3-06b/qtr-1", cwd=repo, timeout=180)
    print("SOURCE_FETCH_COMPLETE", flush=True)
    sys.path.insert(0, str(repo))
    from scripts.research.publish_experiment_results import (
        ExperimentSpec, load_github_token, prepare_artifacts, push_results,
    )
    token = load_github_token("GITHUB_TOKEN")
    upstream.mkdir()
    run("git", "init", str(upstream))
    run("git", "fetch", "--depth", "1", UPSTREAM, REVISION, cwd=upstream, timeout=180)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=upstream)
    # Use the pinned checkout directly. Preserve Kaggle's matched Torch stack.
    import tomllib
    from importlib.metadata import version

    dependencies = tomllib.loads((upstream / "pyproject.toml").read_text())["project"]["dependencies"]
    constraints = root / "layer-location-constraints.txt"
    constraints.write_text("torch==" + version("torch") + "\n")
    run(sys.executable, "-m", "pip", "install", "-c", str(constraints),
        "transformers==4.57.6", "datasets==4.4.1", "assoc-scan==0.0.6", "accelerated-scan==0.3.1",
        *(item for item in dependencies if not item.startswith("torch>=")))
    directory = repo / "research/titans-qwen3-06b/qtr-1"
    output = Path("/kaggle/working") / ("qtr-1-" + uuid.uuid4().hex[:12])
    output.mkdir()
    log = root / "run.log"
    os.environ.update(HAM_TITANS_UPSTREAM=str(upstream), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    import hashlib
    import requests
    import tarfile
    commands = []
    if MODE == "engineering":
        commands.append([sys.executable, "engineering.py", str(upstream), str(output / "formal")])
    elif MODE == "formal":
        import re
        if not re.fullmatch(r"[0-9a-f]{40}", ENGINEERING_RESULT_COMMIT) or not re.fullmatch(
                r"qtr-1-[0-9a-f]{12}", ENGINEERING_RESULT_RUN):
            raise ValueError("Formal mode requires a published engineering result commit and run ID")
        evidence_url = (f"https://raw.githubusercontent.com/ArcheLabs/mini-cells/{ENGINEERING_RESULT_COMMIT}"
                        f"/artifacts/experiments/{ENGINEERING_RESULT_RUN}/decision.json")
        response = requests.get(evidence_url, timeout=30)
        response.raise_for_status()
        evidence = response.json()
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if (evidence.get("format") != "minicells.qtr-1.v1" or evidence.get("status") != "ENGINEERING_ONLY"
            or evidence.get("source_commit") != source or set(evidence.get("arms", {})) != {"E0","E1","E2","E3"}
            or not all(row["status"] == "PASS" for row in evidence["arms"].values())):
            raise ValueError("Engineering evidence does not validate this source")
        cmd = [sys.executable, "train.py", str(upstream), str(output / "formal"),
               "--arm", ARM, "--seed", str(SEED), "--stop-after", str(STOP_AFTER)]
        if RESUME_ASSET_ID:
            url = f"https://api.github.com/repos/ArcheLabs/mini-cells/releases/assets/{int(RESUME_ASSET_ID)}"
            with requests.get(url, headers={"Authorization": "Bearer " + token,
                    "Accept": "application/octet-stream"}, timeout=120, stream=True) as response:
                response.raise_for_status()
                bundle = root / "resume.tar.gz"
                with bundle.open("wb") as f:
                    for chunk in response.iter_content(1024*1024):
                        f.write(chunk)
            incoming = root / "resume.pt"
            with tarfile.open(bundle) as archive, archive.extractfile("resume.pt") as stream, incoming.open("wb") as f:
                shutil.copyfileobj(stream,f)
            cmd.extend(["--resume-from",str(incoming),"--resume-sha256",RESUME_CHECKPOINT_SHA256])
        commands.append(cmd)
    else:
        raise ValueError("Invalid QTR mode")
    with log.open("w") as stream:
        for command in commands:
            with subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                completed = subprocess.CompletedProcess(command, process.wait())
            if completed.returncode:
                break
    output.mkdir(exist_ok=True)
    shutil.move(log, output / "run.log")
    kind = "minicells.qtr-1.v1"
    decision_path = output / "decision.json"
    decision = json.loads((output / "formal" / "decision.json").read_text()) if completed.returncode == 0 else {
        "format": kind, "status": "RUN_FAILED", "engineering_status": "FAIL",
        "scientific_status": "DIAGNOSTIC_ONLY", "diagnostic_status": "INCOMPLETE", "returncode": completed.returncode,
    }
    if script_version is None and completed.returncode == 0:
        decision["diagnostic_status"] = "INCOMPLETE_PROVENANCE"
        decision["kaggle_version_provenance"] = "UNAVAILABLE"
    decision["run_id"] = output.name
    decision["kaggle_script_version_id"] = script_version
    decision["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    decision_path.write_text(json.dumps(decision, indent=2))
    sys.path.insert(0, str(directory))
    import hashlib
    (output / "sha256.json").write_text(json.dumps({
        f.relative_to(output).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(output.rglob("*")) if f.is_file() and f.suffix != ".pt"
    }, indent=2))
    if completed.returncode == 0 and MODE == "formal":
        import requests
        try:
            import tarfile
            bundle = root / (output.name + ".tar.gz")
            with tarfile.open(bundle, "w:gz") as archive:
                for checkpoint in sorted((output / "formal").glob("*.pt")):
                    archive.add(checkpoint, arcname=checkpoint.name)
            headers = {"Authorization": "Bearer " + token,
                       "Accept": "application/vnd.github+json"}
            endpoint = "https://api.github.com/repos/ArcheLabs/mini-cells/releases"
            release = requests.post(endpoint, headers=headers, timeout=60,
                                    json={"tag_name": output.name, "name": output.name})
            release.raise_for_status()
            upload = release.json()["upload_url"].split("{")[0]
            with bundle.open("rb") as stream:
                asset = requests.post(upload, headers={**headers, "Content-Type": "application/gzip"},
                                      params={"name": "checkpoints.tar.gz"}, data=stream, timeout=300)
            asset.raise_for_status()
            view = asset.json()
            with requests.get(view["url"], headers={**headers, "Accept": "application/octet-stream"},
                              stream=True, timeout=120) as download:
                download.raise_for_status()
                digest = hashlib.sha256()
                for chunk in download.iter_content(1024*1024):
                    digest.update(chunk)
            with bundle.open("rb") as stream:
                local = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest.hexdigest() != local:
                raise RuntimeError("Checkpoint bundle readback hash mismatch")
            decision["checkpoint_archive"] = {"status": "VERIFIED", "sha256": local,
                                              "url": view["browser_download_url"], "asset_id": view["id"]}
        except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
            decision["checkpoint_archive"] = {"status": "FAILED", "error_type": type(error).__name__}
            decision["diagnostic_status"] = "INCOMPLETE_PUBLICATION"
        decision_path.write_text(json.dumps(decision, indent=2))
        (output / "sha256.json").write_text(json.dumps({
            f.relative_to(output).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in output.rglob("*") if f.is_file() and f.suffix != ".pt" and f.name != "sha256.json"
        }, indent=2))
    source = f"results/{output.name}"
    shutil.copytree(output, repo / source, ignore=shutil.ignore_patterns("*.pt"))
    branch = f"kaggle/{output.name}"
    spec = ExperimentSpec(source, f"artifacts/experiments/{output.name}",
                          tuple(sorted(file.relative_to(output).as_posix()
                                       for file in output.rglob("*") if file.is_file() and file.suffix != ".pt")),
                          branch, kind)
    run("git", "sparse-checkout", "add", spec.artifact_dir, cwd=repo, timeout=180)
    folder = prepare_artifacts(repo, "qtr-1", spec,
                               script_version or None)
    push_results(repo, folder, "qtr-1", branch,
                 "GITHUB_TOKEN", "origin/main")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    print({"status": "REMOTE_SHA_VERIFIED", "branch": branch, "commit": sha}, flush=True)
    completed.check_returncode()
    if MODE == "formal" and decision.get("checkpoint_archive", {}).get("status") != "VERIFIED":
        raise RuntimeError("Checkpoint archive verification failed; see published decision.json")


if __name__ == "__main__":
    main()
