"""Metadata-only publication using existing authentication on a current-main base."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
from exp006.reuse import file_sha256, canonical_sha256, write_json
from .configs import ROOT, BASE, authorize


def publish(c, p, arm, seed, role, source, run, checkout):
    authorize(c, p, arm, seed, role, "publish", source)
    run = Path(run)
    metrics = json.loads((run / "metrics.json").read_text())
    if metrics["identity"]["experiment_id"] != "006-PD1" or metrics["identity"]["source_sha"] != source:
        raise ValueError("run identity mismatch")
    if metrics["checkpoint_sha"] != file_sha256(run / "checkpoint.pt"):
        raise ValueError("checkpoint hash mismatch")
    checkout = Path(checkout)
    if checkout.exists():
        raise FileExistsError("preserve earlier publication checkout; use a fresh destination")
    checkout.mkdir(parents=True)

    def git(*args, check=True):
        r = subprocess.run(
            ["git", *args], cwd=checkout, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if check and r.returncode:
            raise RuntimeError(r.stderr or r.stdout)
        return r

    git("init")
    git("remote", "add", "origin", "https://github.com/ArcheLabs/mini-cells.git")
    git("config", "remote.origin.promisor", "true")
    git("config", "remote.origin.partialclonefilter", "blob:none")
    git("sparse-checkout", "init", "--no-cone")
    relative = BASE.relative_to(ROOT) / "artifacts"
    git("sparse-checkout", "set", "--no-cone", "/" + str(relative) + "/")
    git("fetch", "--depth=1", "--filter=blob:none", "origin", "main")
    main = git("rev-parse", "FETCH_HEAD").stdout.strip()
    branch = "results/pd001-" + source[:12]
    remote = git("ls-remote", "--exit-code", "--heads", "origin", branch, check=False)
    if remote.returncode == 0:
        git("fetch", "--depth=1", "--filter=blob:none", "origin", "refs/heads/" + branch)
        git("checkout", "-b", branch, "FETCH_HEAD")
        if git("diff", "--quiet", main, "HEAD", "--", ".github/workflows", check=False).returncode:
            raise RuntimeError(
                "result branch workflows differ from current main; do not push code or force-rewrite"
            )
    elif remote.returncode == 2:
        git("checkout", "-b", branch, main)
    else:
        raise RuntimeError(remote.stderr)
    for folder in (
        "research/projects/cellular-language-models/context/08-hybrid-clm/scripts/research",
        "research/projects/continual-learning/context/04-continual-learning/scripts/research",
    ):
        sys.path.insert(0, str(ROOT / folder))
    import publish_core_validation_007 as publisher

    publisher.ROOT = checkout
    publisher.preflight_push(branch, "GITHUB_TOKEN")
    files = {
        str(f.relative_to(run)): file_sha256(f)
        for f in sorted(run.rglob("*"))
        if f.is_file() and f.suffix in (".json", ".jsonl", ".csv", ".png", ".md", ".log")
    }
    if any((run / name).stat().st_size > 20_000_000 for name in files):
        raise ValueError("metadata too large; weights/raw data cannot be published")
    target = checkout / relative / source / canonical_sha256(files)
    if target.exists():
        if json.loads((target / "evidence.json").read_text())["files"] != files:
            raise ValueError("immutable archive changed")
    else:
        target.mkdir(parents=True)
        for name in files:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / name, dest)
        write_json(
            target / "evidence.json",
            {"identity": metrics["identity"], "files": files, "weights": "external checkpoint hash only"},
        )
    git("config", "user.name", "MiniCells Research")
    git("config", "user.email", "research@minicells.invalid")
    git("add", "--", str(target.relative_to(checkout)))
    if git("diff", "--cached", "--quiet", check=False).returncode:
        git("commit", "-m", "research: archive PD-001 metadata")
    publisher.push_branch(branch, "GITHUB_TOKEN")
    return {"branch": branch, "archive": str(target), "scientific_decision": False}
