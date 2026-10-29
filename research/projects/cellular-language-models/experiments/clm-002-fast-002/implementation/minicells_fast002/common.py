import hashlib
import subprocess
from pathlib import Path

from minicells_fast001.common import atomic, digest, read, sha

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[4]
ID = "CLM-002-FAST-002"


def config():
    c = read(BASE / "config/registered.json")
    if sha(c) != (BASE / "config/registered.sha256").read_text().strip():
        raise RuntimeError("FAST002_CONFIG_CHANGED")
    if digest(BASE / "PROTOCOL.zh-CN.md") != c["protocol_sha256"]:
        raise RuntimeError("FAST002_PROTOCOL_CHANGED")
    return c


def source():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def historical_reports(c, destination):
    """One pinned Git fetch, batched object reading; never stage or overwrite old reports."""
    destination = Path(destination)
    ref, prefix = c["historical"]["results_commit"], c["historical"]["report_directory"]
    destination.mkdir(parents=True, exist_ok=True)
    cached = destination / "SOURCE.json"
    if cached.exists():
        proof = read(cached)
        if proof["commit"] != ref:
            raise RuntimeError("FAST002_REPORT_CACHE_IDENTITY_CONFLICT")
        for name, evidence in proof["files"].items():
            if digest(destination / name) != evidence["sha256"]:
                raise RuntimeError("FAST002_HISTORICAL_REPORT_CHANGED")
        return destination
    repo = destination / ".objects"
    if not repo.exists():
        subprocess.run(["git", "init", "--bare", "--quiet", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "config",
            "remote.origin.url",
            "https://github.com/ArcheLabs/mini-cells.git",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "remote.origin.promisor", "true"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "remote.origin.partialclonefilter", "blob:none"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "fetch",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            ref,
        ],
        check=True,
        timeout=60,
    )
    paths = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", ref, "--", prefix], text=True
    ).splitlines()
    # Fetch only this report package, in one request rather than one lazy fetch per job.
    objects = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", ref, "--", prefix], text=True
    ).splitlines()
    blob_ids = sorted({row.split()[2] for row in objects})
    if blob_ids:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "fetch",
                "--quiet",
                "--no-tags",
                "origin",
                *blob_ids,
            ],
            check=True,
            timeout=60,
        )
    files = {}
    for path in paths:
        name = Path(path).relative_to(prefix).as_posix()
        if not (
            name
            in {
                "RUN_MANIFEST.json",
                "STAGE_A_DECISION.json",
                "FINAL_ANALYSIS.json",
                "BUDGET.json",
                "DATASET_MANIFEST.json",
                "PARENT_FORK_ACCEPTANCE.json",
            }
            or name.startswith("jobs/")
            and name.endswith(".json")
        ):
            continue
        content = subprocess.check_output(["git", "-C", str(repo), "show", ref + ":" + path])
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    atomic(cached, {"commit": ref, "files": files})
    return destination


def require_ci(head):
    import json
    import urllib.request

    url = (
        "https://api.github.com/repos/ArcheLabs/mini-cells/actions/workflows/"
        "clm002-fast002.yml/runs?head_sha=" + head + "&per_page=20"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "MiniCells-FAST002"})
    with urllib.request.urlopen(request, timeout=20) as response:
        rows = json.load(response)["workflow_runs"]
    matches = [r for r in rows if r["head_sha"] == head]
    if not matches or matches[0]["conclusion"] != "success":
        raise RuntimeError("FAST002_CURRENT_SOURCE_CI_NOT_GREEN_NO_GPU_STARTED")
    return {
        "source_sha": head,
        "run_id": matches[0]["id"],
        "conclusion": "success",
        "url": matches[0]["html_url"],
    }
